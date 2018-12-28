"""load pdbbind into ani dataset in hdf5.
"""
import sys
import h5py
import argparse
import numpy as np
from biopandas.mol2 import split_multimol2
from biopandas.mol2 import PandasMol2

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-i', '--input_mol2', required=True, nargs='+',
    help="input mol2 files")
parser.add_argument('-o', '--output', default='data',
    help="prefix of output. default is data")
parser.add_argument('-s', '--split_seed', type=int, default=123,
    help="split seed, default is 123, will split in 80/20")
args = parser.parse_args()


alltype = set()
h5 = h5py.File(args.output + '.h5')
print("All data save in " + h5.filename)
mol = PandasMol2()
ids = []
for file_name in args.input_mol2:
    if 'decoy' in file_name:
      energy = 0.
    if 'ligand' in file_name:
      energy = 1.
    for (mol2_code, mol2_lines) in split_multimol2(file_name):
        mol.read_mol2_from_list(mol2_lines, mol2_code)
        S = np.array([i[0].lower() for i in mol.df['atom_type']])
        C = np.array(mol.df[['x','y','z']])
        C = np.array([C])
        metal = ('Na', 'Mg', 'Mn', 'Ca', 'Zn', 'Co', 'Cd', 'Cs', 'Cu', 'Fe', 'Hg', 'Ni', 'Sr', 'K')
        halogen = ('f', 'cl', 'br', 'i')
        S[np.isin(S, metal)] = 'M'
        S[np.isin(S, halogen)] = 'x'
        S[S=='Se'] = 'S'
        alltype.update(S)
        noH = np.logical_and(S != 'H', S != 'h')
        S = S[noH]
        C = C[:,noH]
        S = [i.encode('utf8') for i in S]
        if mol2_code in h5:
            continue
        ids.append(mol2_code)
        g = h5.create_group(mol2_code)
        g.create_dataset('energies', data=[energy])
        g.create_dataset('species', data=S)
        g.create_dataset('coordinates', data=C)
print("alltype: {}".format(sorted(alltype)))
np.random.seed(args.split_seed)
ids = np.array(ids)
n = len(ids)
perm = np.random.permutation(n)
train_ids = ids[perm[:int(0.8*n)]]
test_ids = ids[perm[int(0.8*n):]]

with h5py.File(args.output+'.train.h5') as f:
  print("train data save in " + f.filename)
  for id_ in train_ids:
      h5.copy(id_, f)

with h5py.File(args.output+'.test.h5') as f:
  print("test data save in " + f.filename)
  for id_ in test_ids:
      h5.copy(id_, f)

h5.close()
