# This code can be run outside the DebrisDiskFM package
# Temaplte for MCMC on a cluster
import matplotlib
matplotlib.use('agg')
import sys
import debrisdiskfm                             # to import debrisdiskfm, make sure you setup the code in the DebrisDiskFM package using "python3 setup.py develop"
from debrisdiskfm import lnpost_hd191089
import numpy as np
from schwimmbad import MPIPool
import time
import os
#import matplotlib.pyplot as plt


path_obs='./data_observation/'                  # Where the observed data are, check/modify the lnlike.lnlike_hd191089() function, 
                                                # especially the "#Observations" section for your own adjustion.
                                                
path_model='./mcfost_models/'                   # Where you'd like to store the models duing MCMC
var_names = np.array(['inc', 'PA', 'm_disk', 'Rc']) # Parameters of interest, the following commented line is all the parameters
# var_names = np.array(['inc', 'PA', 'm_disk', 'Rc', 'R_in', 'alpha_in', 'alpha_out', 'porosity', 'fmass_0', 'fmass_1', 'a_min', 'Q_powerlaw'])
var_values_init = np.array([59.7, 70, -7, 45.3])    # Initial guesses for the above parameters, the following line is for all the parameters
# var_values_init = np.array([59.7, 70, 1e-7, 45.3, 20, 3.5,  -3.5, 0.1, 0.05, 0.9, 1.0, 3.5])

#lnpost_initial = lnpost_hd191089(var_values=var_values_init, var_names=var_names, path_obs=path_obs, path_model=path_model, calcSED=True, hash_address = False)# The above line calculates the SED to make sure MCMC can run in the "image-only" mode, as in the following line
lnpost_mcmc = lnpost_hd191089(var_values=var_values_init, var_names=var_names, path_obs=path_obs, path_model=path_model, hash_address=True)#, calcSED=False)

n_dim = len(var_values_init)    # number of variables
n_walkers = int(2*n_dim)              # an even number (>= 2*n_dim)
step = 10                       # how many steps are expected for MCMC to run
# CAUTION: Approximated Time for Running:
# time_expected = n_walkers * step * 10 seconds. In the setup of this code, 8*10*10s = 800s = 13 minitues is expected
# where the 10s is extimated from the MCFOST generation and forward modeling of STIS, NICMOS, and GPI images of HD191089

########################################################################################################
############################                MCMC               #########################################
########################################################################################################
import emcee
filename = "state.h5"
backend = emcee.backends.HDFBackend(filename)   # the backend file is used to store the status

with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
    start = time.time()
    if not os.path.exists(filename):  #initial run, no backend file existed
        sampler = emcee.EnsembleSampler(nwalkers = n_walkers, ndim = n_dim, log_prob_fn=lnpost_hd191089, args=[var_names, path_obs, path_model], pool = pool, backend=backend)
        values_ball = [var_values_init + 1e-1*np.random.randn(n_dim) for i in range(n_walkers)] # Initialize the walkers using different values 
                                                                                            # around the initial guess (var_values_init)
        sampler.run_mcmc(values_ball, step)
    else:    #load the data directly from the backend file
        sampler = emcee.EnsembleSampler(nwalkers = n_walkers, ndim = n_dim, log_prob_fn = lnpost_hd191089, args = [var_names, path_obs, path_model], pool = pool, backend = backend)
        sampler.run_mcmc(None, nsteps = step)
    end = time.time()
    serial_time = end - start
    print("2 nodes * 6 tasks * 4 cores with MPI took {0:.1f} seconds".format(serial_time))

#import corner
#trunc = 0                                        # A step number you'd like to truncate at (aim: get rid of the burrning stage)
#samples = sampler.chain[:, trunc:, :].reshape((-1, n_dim))

#plt.figure()
#labels = ['inc', 'PA', '$log_{10}m_{disk}$', '$R_c$']
#fig = corner.corner(samples, labels=labels, quantiles =[.16, .50, .84])         # corner plots with the quantiles  (-1sigma, median, +1sigma)
#plt.savefig('./corner.pdf')                      # save the file

# The numbers for median+/-1sigma are calculated with the following line
#inc, pa, m_disk, rc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))
