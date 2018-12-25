"""load pdbbind into ani dataset in hdf5.
"""
import sys
import h5py
import argparse
import numpy as np
from ase.io import read
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-i', '--index', required=True,
    help="index of pdbbind.")
parser.add_argument('-o', '--output', default='data',
    help="prefix of output. default is data")
parser.add_argument('-s', '--split_seed', type=int, default=123,
    help="split seed, default is 123, will split in 80/20")
args = parser.parse_args()

def load_index(index_file):
    index = []
    with open(index_file) as f:
        for i in f:
            if i.startswith("#"): continue
            fileds = i.split()
            pdb_id, resolution, year, logK, K, cluster_id  = fileds[:6]
            index.append((pdb_id, float(logK)))
    return index


index = load_index(args.index)
# print(index)

h5 = h5py.File(args.output + '.h5')
print("All data save in " + h5.filename)
for pdb_id, logK in index:
    print(pdb_id, logK)
    ligand = read("v2015/{}/{}_ligand.pdb".format(pdb_id, pdb_id))
    ligandS = [i.lower() for i in ligand.get_chemical_symbols()]
    ligandC = ligand.positions
    pocket = read("v2015/{}/{}_pocket.pdb".format(pdb_id, pdb_id))
    pocketS = pocket.get_chemical_symbols()
    pocketC = pocket.positions
    S = np.concatenate((ligandS, pocketS))
    C = np.array([np.concatenate((ligandC, pocketC))])
    noH = np.logical_and(S != 'H', S != 'h')
    S = S[noH]
    C = C[:,noH]
    S = [i.encode('utf8') for i in S]
    g = h5.create_group(pdb_id)
    g.create_dataset('energies', data=[logK])
    g.create_dataset('species', data=S)
    g.create_dataset('coordinates', data=C)

np.random.seed(args.split_seed)
pdb_ids = np.array([pdb_id for (pdb_id, logK) in index])
n = len(pdb_ids)
perm = np.random.permutation(n)
train_ids = pdb_ids[perm[:int(0.8*n)]]
test_ids = pdb_ids[perm[int(0.8*n):]]

with h5py.File(args.output+'.train.h5') as f:
  print("train data save in " + f.filename)
  for pdb_id in train_ids:
      #f.create_group(pdb_id)
      h5.copy(pdb_id, f)

with h5py.File(args.output+'.test.h5') as f:
  print("test data save in " + f.filename)
  for pdb_id in test_ids:
      #f.create_group(pdb_id)
      h5.copy(pdb_id, f)

h5.close()
