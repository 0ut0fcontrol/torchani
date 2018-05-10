import pkg_resources
import torch

buildin_const_file = pkg_resources.resource_filename(
    __name__, 'data/rHCNO-4.6R_16-3.1A_a4-8_3.params')

buildin_sae_file = pkg_resources.resource_filename(
    __name__, 'data/sae_linfit.dat')

buildin_network_dir = pkg_resources.resource_filename(
    __name__, 'data/networks/')

default_dtype = torch.float32
default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from .energyshifter import EnergyShifter
from .nn import ModelOnAEV, PerSpeciesFromNeuroChem
from .aev import AEV
from .dataset import Dataset
import logging

__all__ = ['AEV', 'EnergyShifter', 'ModelOnAEV', 'PerSpeciesFromNeuroChem', 'Dataset',
           'buildin_const_file', 'buildin_sae_file', 'buildin_network_dir', 'default_dtype', 'default_device']

try:
    from .neurochem_aev import NeuroChemAEV
    __all__.append('NeuroChemAEV')
except ImportError:
    logging.log(logging.WARNING,
                'Unable to import NeuroChemAEV, please check your pyNeuroChem installation.')
