############ This code is adapted from
###### http://dfm.io/emcee/current/user/line/
###### to demonstrate the MPI capability of EMCEE with line fitting

import matplotlib
matplotlib.use('Agg')
#import corner
import emcee
import mpi4py
from schwimmbad import MPIPool
#from emcee.utils import MPIPool
import sys
import numpy as np
import time

# Choose the "true" parameters.
m_true = -0.9594
b_true = 4.294
f_true = 0.534

# Generate some synthetic data from the model.
N = 50
x = np.sort(10*np.random.rand(N))
yerr = 0.1+0.5*np.random.rand(N)
y = m_true*x+b_true
y += np.abs(f_true*y) * np.random.randn(N)
y += yerr * np.random.randn(N)

A = np.vstack((np.ones_like(x), x)).T
C = np.diag(yerr * yerr)
cov = np.linalg.inv(np.dot(A.T, np.linalg.solve(C, A)))
b_ls, m_ls = np.dot(cov, np.dot(A.T, np.linalg.solve(C, y)))


def lnlike(theta, x, y, yerr):
    m, b, lnf = theta
    model = m * x + b
    inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))


import scipy.optimize as op
nll = lambda *args: -lnlike(*args)
result = op.minimize(nll, [m_true, b_true, np.log(f_true)], args=(x, y, yerr))
m_ml, b_ml, lnf_ml = result["x"]


def lnprior(theta):
    m, b, lnf = theta
    if -5.0 < m < 0.5 and 0.0 < b < 10.0 and -10.0 < lnf < 1.0:
        return 0.0
    return -np.inf


def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    time.sleep(1.0)                              # pause for 1 second, this is for demonstration to make sure that MPI is faster than a singlenode
                                                 # caution: please delete this command in your own calculation
                                                 # MPI is working great unless the loglikelihood procedure takes more time than the communication among nodes
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)


ndim, nwalkers = 3, 10
pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
with MPIPool() as pool:
    #pool = MPIPool()
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr), pool = pool)

    #import time
    start = time.time()
    sampler.run_mcmc(pos, 50)

    #pool.close()
    end = time.time()
    serial_time = end - start
    print("2 nodes * 6 tasks * 4 cores with MPI took {0:.1f} seconds".format(serial_time))

#samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

#import matplotlib.pyplot as plt
#plt.figure()
#corner.corner(samples, labels=["$m$", "$b$", "$\ln\,f$"],
#                      truths=[m_true, b_true, np.log(f_true)])
#plt.savefig('./test.png')

