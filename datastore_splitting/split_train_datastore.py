import argparse
import os
import numpy as np
import faiss
import time
import ctypes
import pickle

# the implementation refers to knnlm

parser = argparse.ArgumentParser()
parser.add_argument('--dstore_mmap', type=str, help='memmap where keys and vals are stored')
parser.add_argument('--dstore_size', type=int, help='number of items saved in the datastore memmap')
parser.add_argument('--dimension', type=int, default=1024, help='Size of each key')
parser.add_argument('--dstore-fp16', default=False, action='store_true')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed for sampling the subset of vectors to train the cache')
parser.add_argument('--ncentroids', type=int, default=4096, help='number of centroids faiss should learn')
parser.add_argument('--code_size', type=int, default=64, help='size of quantized vectors')
parser.add_argument('--probe', type=int, default=32, help='number of clusters to query')
parser.add_argument('--faiss_index', type=str, help='file to write the faiss index')
parser.add_argument('--num_keys_to_add_at_a_time', default=1000000, type=int,
                    help='can only load a certain amount of data to memory at a time.')
parser.add_argument('--starting_point', type=int, default=0, help='index to start adding keys at')
parser.add_argument('--use_gpu', default=False, action='store_true')
parser.add_argument("--pca", default=0, type=int)
parser.add_argument("--n_datastores", default=16, type=int)
parser.add_argument("--n_examples_train_kmeans", default=10000000, type=int)
parser.add_argument("--kmeans_iter", default=50, type=int)

args = parser.parse_args()

print(args)

res = faiss.StandardGpuResources()

# load the saved keys and values
if args.dstore_fp16:
    print('load dstore fp16', args.dstore_size, args.dimension)
    keys = np.memmap(args.dstore_mmap + 'keys.npy', dtype=np.float16, mode='r', shape=(args.dstore_size, args.dimension))
    vals = np.memmap(args.dstore_mmap + 'vals.npy', dtype=np.int, mode='r', shape=(args.dstore_size, 1))
else:
    print(args.dstore_mmap + 'keys.npy')
    keys = np.memmap(args.dstore_mmap + 'keys.npy', dtype=np.float32, mode='r', shape=(args.dstore_size, args.dimension))
    vals = np.memmap(args.dstore_mmap + 'vals.npy', dtype=np.int, mode='r', shape=(args.dstore_size, 1))

print('done.')

# to speed up access to np.memmap
madvise = ctypes.CDLL("libc.so.6").madvise
madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
madvise.restype = ctypes.c_int
assert madvise(keys.ctypes.data, keys.size * keys.dtype.itemsize, 1) == 0, "MADVISE FAILED" # 2 means MADV_SEQUENTIAL

np.random.seed(args.seed)  

index_dim = args.pca if args.pca > 0 else args.dimension

random_sample = np.random.choice(np.arange(vals.shape[0]), size=[min(args.n_examples_train_kmeans, vals.shape[0])], replace=False)
kmeans = faiss.Kmeans(index_dim, args.n_datastores, niter=args.kmeans_iter, verbose=True)

kmeans.train(keys[random_sample].astype(np.float32)) # keys.astype(np.float32))
#kmeans.train(keys.astype(np.float32))

print('finished training kmeans')

_, I = kmeans.index.search(keys, 1)

print('finish searching')


centroids=kmeans.centroids



dstore_sizes=[]
quantizer = faiss.IndexFlatL2(index_dim)

if args.pca>0:
    pca_matrix = faiss.PCAMatrix(args.dimension, args.pca, 0, True)

indexes = {}
j=0
aux=0
for i in range(args.n_datastores):

    print('\n')
    print('Training Index', i)

    idx = (I==i).nonzero()[0]

    if len(idx)>4096:

        indexes[j] = faiss.IndexIVFPQ(quantizer, index_dim, args.ncentroids, args.code_size, 8)
        indexes[j].nprobe = args.probe

        if args.pca > 0:
            indexes[j] = faiss.IndexPreTransform(pca_matrix, indexes[j])

        keys_ = keys[idx]
        vals_ = vals[idx]

        dstore_sizes.append(vals_.shape[0])

        dstore_vals = np.memmap(args.faiss_index + '/vals_' + str(j) + '.npy', dtype=np.int, mode='w+', shape=(vals_.shape[0], 1))
        dstore_vals[:] = vals_
        dstore_vals.flush()

        random_sample = np.random.choice(np.arange(vals_.shape[0]), size=[min(1000000, vals_.shape[0])], replace=False)
        start = time.time()

        indexes[j].train(keys_[random_sample].astype(np.float32))

        print('Training took {} s'.format(time.time() - start))

        print('Writing index after training')
        start = time.time()

        faiss.write_index(indexes[j], args.faiss_index + str(j) + "_knn_index.trained")
        
        print('Writing index took {} s'.format(time.time() - start))        

        print('Adding Keys')
        index = faiss.read_index(args.faiss_index + str(j) + "_knn_index.trained")

        

        start = args.starting_point
        start_time = time.time()
        while start < vals_.shape[0]:
            end = min(vals_.shape[0], start + args.num_keys_to_add_at_a_time)
            to_add = keys_[start:end].copy()

            indexes[j].add_with_ids(to_add.astype(np.float32), np.arange(start, end))

            start += args.num_keys_to_add_at_a_time

            if (start % 1000000) == 0:
                print('Added %d tokens so far' % start)
                print('Writing Index', start)
                faiss.write_index(indexes[j], args.faiss_index+ str(j) + "_knn_index")

            print("Adding total %d keys" % end)
            print('Adding took {} s'.format(time.time() - start_time))
            print('Writing Index')
            start = time.time()

            faiss.write_index(indexes[j], args.faiss_index+ str(j) + "_knn_index")

            print('Writing index took {} s'.format(time.time() - start))

        j+=1

    else:
        centroids=np.delete(centroids, i-aux, axis=0)
        aux+=1


np.save(args.faiss_index+'centroids', centroids)
            
            
with open(args.faiss_index + 'dstore_sizes', 'wb') as f:
    pickle.dump(dstore_sizes, f)