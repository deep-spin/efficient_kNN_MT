import argparse
import os
import numpy as np
import faiss
import time
from collections import defaultdict
import ctypes


parser = argparse.ArgumentParser()
parser.add_argument('--dstore_mmap', type=str, help='memmap where keys and vals are stored')
parser.add_argument('--dstore_size', type=int, help='number of items saved in the datastore memmap')
parser.add_argument('--dimension', type=int, default=1024, help='Size of each key')
parser.add_argument('--dstore-fp16', default=False, action='store_true')
parser.add_argument('--probe', type=int, default=32, help='number of clusters to query')
parser.add_argument('--faiss_index', type=str)
parser.add_argument('--starting_point', type=int, default=0, help='index to start adding keys at')

parser.add_argument('--k', type=int, default=1024, help='the number of nearest neighbors')
parser.add_argument('--save-dir', type=str)
parser.add_argument('--num', type=int, default=1e12,help='number of points to traverse')
parser.add_argument('--batch-size', type=int, default=3072)

args = parser.parse_args()

print(args)

res = faiss.StandardGpuResources()

# load the saved keys and values
if args.dstore_fp16:
    print('load dstore fp16', args.dstore_size, args.dimension)
    keys = np.memmap(args.dstore_mmap + '/keys.npy', dtype=np.float16, mode='r', shape=(args.dstore_size, args.dimension))
    vals = np.memmap(args.dstore_mmap + '/vals.npy', dtype=np.int, mode='r', shape=(args.dstore_size, 1))
else:
    keys = np.memmap(args.dstore_mmap + '/keys.npy', dtype=np.float32, mode='r', shape=(args.dstore_size, args.dimension))
    vals = np.memmap(args.dstore_mmap + '/vals.npy', dtype=np.int, mode='r', shape=(args.dstore_size, 1))

save_size = min(args.num, args.dstore_size - args.starting_point)

retrieve =  np.memmap(args.save_dir + f'_size{save_size}_k{args.k}_int32.npy', dtype=np.int32, mode='w+', shape=(save_size, args.k))

index = faiss.read_index(args.faiss_index, faiss.IO_FLAG_ONDISK_SAME_DIR)

# from https://github.com/numpy/numpy/issues/13172
# to speed up access to np.memmap
madvise = ctypes.CDLL("libc.so.6").madvise
madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
madvise.restype = ctypes.c_int
assert madvise(keys.ctypes.data, keys.size * keys.dtype.itemsize, 1) == 0, "MADVISE FAILED" # 2 means MADV_SEQUENTIAL

batches = []
cnt = 0
offset = 0
bsz = args.batch_size

score = defaultdict(float)
t = time.time()

for id_, i in enumerate(range(args.starting_point, min(args.starting_point + args.num, args.dstore_size))):
    if i % 10000 == 0:
        print(f'processing {i}th entries', flush='True')
        
    batches.append(keys[i])
    cnt += 1

    if cnt % bsz == 0:
        dists, knns = index.search(np.array(batches).astype(np.float32), args.k)
        assert knns.shape[0] == bsz

        retrieve[offset:offset + knns.shape[0]] = knns

        cnt = 0
        batches = []

        offset += knns.shape[0]

if len(batches) > 0:
    dists, knns = index.search(np.array(batches).astype(np.float32), args.k)
    retrieve[offset:offset + knns.shape[0]] = knns