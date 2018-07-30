############ This code is adapted from
###### http://dfm.io/emcee/current/user/line/
###### to demonstrate the MPI capability of EMCEE

import matplotlib
matplotlib.use('Agg')
#import corner
import time
import numpy as np

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
    time.sleep(1.0)                                               # sleep 1 second just to make sure the loglikelihood procedure takes relatively more
                                                                  # time than the communication among nodes.
                                                                  # delete the sleep command in your own calculation!
    if -5.0 < m < 0.5 and 0.0 < b < 10.0 and -10.0 < lnf < 1.0:
        return 0.0
    return -np.inf


def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)

pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]


import emcee
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))


result["x"] 

pos

import time
start = time.time()
sampler.run_mcmc(pos, 50)
end = time.time()
serial_time = end - start
print("1 core on 1 node  took {0:.1f} seconds".format(serial_time))

samples = sampler.chain[:, 50:, :].reshape((-1, ndim))


#import matplotlib.pyplot as plt
#plt.figure()
#corner.corner(samples, labels=["$m$", "$b$", "$\ln\,f$"],
#                      truths=[m_true, b_true, np.log(f_true)])
#plt.savefig('./test.png')

