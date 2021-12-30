import argparse
import os
import numpy as np
import faiss
import time
import ctypes

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
    keys = np.memmap(args.dstore_mmap + 'keys.npy', dtype=np.float16, mode='r', 
                        shape=(args.dstore_size, args.dimension))
    vals = np.memmap(args.dstore_mmap + 'vals.npy', dtype=np.int, mode='r', 
                        shape=(args.dstore_size, 1))
else:
    print(args.dstore_mmap + 'keys.npy')
    keys = np.memmap(args.dstore_mmap + 'keys.npy', dtype=np.float32, mode='r', 
                        shape=(args.dstore_size, args.dimension))
    vals = np.memmap(args.dstore_mmap + 'vals.npy', dtype=np.int, mode='r', 
                        shape=(args.dstore_size, 1))

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

kmeans.train(keys[random_sample].astype(np.float32))

D, _ = kmeans.index.search(keys, 1)

print(D.shape)

log_file = open(args.faiss_index+'_log', 'w')
if not os.path.exists(args.faiss_index + ".trained"):
    quantizer = faiss.IndexFlatL2(index_dim)

    if args.pca>0:
        pca_matrix = faiss.PCAMatrix(args.dimension, args.pca, 0, True)
    
    indexes = {}
    # Initialize faiss index
    for i in range(args.n_datastores):
        indexes[i] = faiss.IndexIVFPQ(quantizer, index_dim, args.ncentroids, args.code_size, 8)
        indexes[i].nprobe = args.probe

        if args.pca > 0:
            indexes[i] = faiss.IndexPreTransform(pca_matrix, indexes[i])

        print('Training Index', i)

        random_sample = np.random.choice(np.arange(multi_vals[i].shape[0]), size=[min(1000000, multi_vals[i].shape[0])], replace=False)
        start = time.time()

        indexes[i].train(multi_keys[i][random_sample].astype(np.float32))

        print('Training took {} s'.format(time.time() - start))

        print('Writing index after training')
        start = time.time()

        faiss.write_index(indexes[i], args.faiss_index + '_' + str(i) + ".trained")
        
        print('Writing index took {} s'.format(time.time() - start))        

        print('Adding Keys')
        index = faiss.read_index(args.faiss_index + '_' + str(i) + ".trained")

        start = args.starting_point
        start_time = time.time()
        while start < multi_vals[i].shape[0]:
            end = min(multi_vals[i].shape[0], start + args.num_keys_to_add_at_a_time)
            to_add = keys[start:end].copy()

            indexes[i].add_with_ids(to_add.astype(np.float32), np.arange(start, end))

            start += args.num_keys_to_add_at_a_time

            if (start % 1000000) == 0:
                print('Added %d tokens so far' % start)
                print('Writing Index', start)
                faiss.write_index(indexes[i], args.faiss_index+ '_' + str(i))

            print("Adding total %d keys" % end)
            print('Adding took {} s'.format(time.time() - start_time))
            print('Writing Index')
            start = time.time()

            faiss.write_index(indexes[i], args.faiss_index+ '_' + str(i))

            print('Writing index took {} s'.format(time.time() - start))
            
            log_file.write('index ' + str(i) + ':' + str(multi_vals[i].shape[0]) + '\n\n')