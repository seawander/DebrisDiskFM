import numpy as np
from astropy.io import fits
import subprocess
from emcee import PTSampler, EnsembleSampler
from . import lnlike

# def main_hd191089():
#     """Main function for MCMC modeling of HD191089 debris disk system."""
#     var_names = ['inc', 'PA', 'm_disk',
#                  'Rc', 'R_in', 'alpha_in', 'alpha_out', 'porosity',
#                  'fmass_0', 'fmass_1',
#                  'a_min', 'Q_powerlaw']
#
#     n_dim = len(var_names)  # dimension (size) of the variables
#     n_walkers = 1e3         # number of walkers
#
#     sampler = EnsembleSampler(nwalkers = n_walkers, dim = n_dim, lnpostfn = lnlike.lnlike_hd191089)

from . import mcfostRun

# mcmc_wrapper_hd191089.mcmc_wrapper_hd191089()