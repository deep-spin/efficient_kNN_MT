import argparse
import os
import numpy as np
import faiss
import time
import ctypes
import random

parser = argparse.ArgumentParser()
parser.add_argument('--dstore_mmap', type=str, help='memmap where keys and vals are stored')
parser.add_argument('--dstore_size', type=int, help='number of items saved in the datastore memmap')
parser.add_argument('--dimension', type=int, default=1024, help='Size of each key')
parser.add_argument('--dstore-fp16', default=False, action='store_true')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--retrieval-dir', type=str)
parser.add_argument('--save-dir', type=str)
parser.add_argument('--k', type=int, default=5, help='the number of nearest neighbors to probe')


args = parser.parse_args()

print(args)

res = faiss.StandardGpuResources()

# load the saved keys and values
if args.dstore_fp16:
    print('load dstore fp16', args.dstore_size, args.dimension)
    keys = np.memmap(args.dstore_mmap + '/keys.npy', dtype=np.float16, mode='r', shape=(args.dstore_size, args.dimension))
    vals = np.memmap(args.dstore_mmap + '/vals.npy', dtype=int, mode='r', shape=(args.dstore_size, 1))
else:
    keys = np.memmap(args.dstore_mmap + '/keys.npy', dtype=np.float32, mode='r', shape=(args.dstore_size, args.dimension))
    vals = np.memmap(args.dstore_mmap + '/vals.npy', dtype=int, mode='r', shape=(args.dstore_size, 1))

def parse_retrieve_fname(fname):
    offset = size = nk = None
    x = fname.split('_')
    for s in x:
        if s.startswith('start'):
            offset = int(s.split('start')[-1])

        if s.startswith('size'):
            size = int(s.split('size')[-1])

        if s.startswith('k'):
            nk = int(s.split('k')[-1])


    if offset is not None and size is not None and nk is not None:
        return offset, size, nk

    raise ValueError(f"parsing error for {fname}")

weights = np.ones(args.dstore_size, dtype=np.int)

def merge_knn(fname):
    print(f'start processing {fname}', flush=True)
    offset, size, nk = parse_retrieve_fname(fname)
    print(f'offset: {offset}, size{size}, k{nk}', flush=True)
    scores = np.zeros(args.dstore_size, dtype=np.float32)

    ret = np.memmap(os.path.join(args.retrieval_dir, fname), dtype=np.int32, mode='r', shape=(size, nk))

    t = time.time()
    ret_mem = np.zeros((size, args.k+1), dtype=np.int32)
    ret_mem[:] = ret[:, :args.k+1]
    print(f'reading index into memory costs {time.time() - t} seconds', flush=True)

    # traverse with random order
    random_order = list(range(size))
    random.shuffle(random_order)

    for i, id_ in enumerate(random_order):
        if i % 100000 == 0:
            print(f'processing {i} rows', flush=True)

        cur_id = offset + id_

        # already removed
        if weights[cur_id] <= 0:
            continue

        for k, v in enumerate(ret_mem[id_]):
            if cur_id != v and weights[v] == 1 and vals[v] == vals[cur_id]:
                # select one to drop
                weights[v] = 0
                weights[cur_id] += 1

    del ret_mem

# from https://github.com/numpy/numpy/issues/13172
# to speed up access to np.memmap
madvise = ctypes.CDLL("libc.so.6").madvise
madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
madvise.restype = ctypes.c_int
assert madvise(keys.ctypes.data, keys.size * keys.dtype.itemsize, 1) == 0, "MADVISE FAILED" # 2 means MADV_SEQUENTIAL

fnames = []

for f in os.listdir(args.retrieval_dir):
    if f.startswith('retrieve') and f.endswith('npy'):
        fnames.append(f)

# import pdb; pdb.set_trace()
random.shuffle(fnames)
for fname in fnames:
    merge_knn(fname)

num = (weights > 0).sum()
print(f'the new datastore has {num} entries')

if args.dstore_fp16:
    new_key = np.memmap(os.path.join(args.save_dir,
            f'dstore_merge{args.k}_size{num}_keys.npy'), mode='w+', dtype=np.float16, shape=(num, args.dimension))
    new_val = np.memmap(os.path.join(args.save_dir,
            f'dstore_merge{args.k}_size{num}_vals.npy'), mode='w+', dtype=np.int, shape=(num, 1))
    new_weight = np.memmap(os.path.join(args.save_dir,
            f'dstore_merge{args.k}_size{num}_weights.npy'), mode='w+', dtype=np.int, shape=(num, 1))
else:
    new_key = np.memmap(os.path.join(args.save_dir,
            f'dstore_merge{args.k}_size{num}_keys.npy'), mode='w+', dtype=np.float32, shape=(num, args.dimension))
    new_val = np.memmap(os.path.join(args.save_dir,
            f'dstore_merge{args.k}_size{num}_vals.npy'), mode='w+', dtype=np.int, shape=(num, 1))
    new_weight = np.memmap(os.path.join(args.save_dir,
            f'dstore_merge{args.k}_size{num}_weights.npy'), mode='w+', dtype=np.int, shape=(num, 1))

cnt = 0
for i, v in enumerate(weights):
    if i % 500000 == 0:
        print(f'writing {i} tokens', flush=True)

    if v > 0:
        new_key[cnt] = keys[i]
        new_val[cnt] = vals[i]
        new_weight[cnt] = v
        cnt += 1

        