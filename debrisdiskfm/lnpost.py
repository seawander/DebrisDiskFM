from . import lnprior
from . import lnlike
from . import mcfostRun
import numpy as np
import shutil

def lnpost_hd191089(var_values = None, var_names = None, path_obs = None, path_model = None, calcSED = False, hash_address = True, STIS = True, NICMOS = True, GPI = True, Fe_composition = False, pit = False, pit_input = None):
    """Returns the log-posterior probability (post = prior * likelihood, thus lnpost = lnprior + lnlike)
    for a given parameter combination.
    Input:  var_values: number array, values for var_names. Refer to mcfostRun() for details. 
                'It is important that the first argument of the probability function is the position of a single walker (a N dimensional numpy array).' (http://dfm.io/emcee/current/user/quickstart/)
            var_names: string array, names of variables. Refer to mcfostRun() for details.
            path_obs: string, address where the observed values are stored.
            path_model: string, address where you would like to store the MCFOST raw models (not forwarded ones).
            calcSED: boolean, whether to calculate the SED of the system.
            hash_address: boolean, "True" strongly suggested for parallel computation efficiency--folders with different names will be created and visited.
            STIS: boolean, whether to calculate the STIS data?
            NICMOS: boolean, wheter to calculate the NICMOS data?
            GPI: boolean, whether to calculate the GPI data?
            Fe_composition: boolean, default is False (i.e., use amorphous Silicates, amorphous Carbon, and water Ice);
                                    if True, water ice will be switched to Fe-Posch.
            pit: boolean, whether to use Probability Integral Transform (PIT) to sample from the posteriors from the previous MCMC run?
                If True, then `pit_input` cannot be None
            pit_input: 2D array/matrix, input MCMC posterior from last run, if not None, only when `pit == True` will it be considered
    Output: log-posterior probability."""
    if pit:
        var_values_percentiles = np.copy(var_values)
        for percentile in var_values_percentiles:
            if not (5 <= percentile <= 95):
                return -np.inf                  #only accept percentiles ranging from 0 to 100 (PIT requirement)
        for i, percentile in enumerate(var_values_percentiles):
            var_values[i] = np.nanpercentile(pit_input[:, i], percentile)
        
    ln_prior = lnprior.lnprior_hd191089(var_names = var_names, var_values = var_values)
    
    if not np.isfinite(ln_prior):
        return -np.inf
        
    run_flag = 1
    try:
        if hash_address:
            run_flag, hash_string = mcfostRun.run_hd191089(var_names = var_names, var_values = var_values, paraPath = path_model, calcSED = calcSED, calcImage = True, hash_address = hash_address, STIS = STIS, NICMOS = NICMOS, GPI = GPI, Fe_composition = Fe_composition)
        else:
            run_flag = mcfostRun.run_hd191089(var_names = var_names, var_values = var_values, paraPath = path_model, calcSED = calcSED, calcImage = True, hash_address = hash_address, STIS = STIS, NICMOS = NICMOS, GPI = GPI, Fe_composition = Fe_composition)
    except:
        pass
        
    if not (run_flag == 0):             # if run is not successful, remove the folders
        try:
            if hash_address:
                shutil.rmtree(path_model[:-1] + hash_string + '/')
            else:
                shutil.rmtree(path_model)
        except:
            print('This folder is not successfully removed.')
        return -np.inf
    try:                                # if run is successful, calculate the posterior
        if hash_address:
            ln_likelihood = lnlike.lnlike_hd191089(path_obs = path_obs, path_model = path_model, hash_address = hash_address, hash_string = hash_string, STIS = STIS, NICMOS = NICMOS, GPI = GPI)
        else:
            ln_likelihood = lnlike.lnlike_hd191089(path_obs = path_obs, path_model = path_model, hash_address = hash_address, STIS = STIS, NICMOS = NICMOS, GPI = GPI)
        
        return ln_prior + ln_likelihood
    except:
        if hash_address:               
            shutil.rmtree(path_model[:-1] + hash_string + '/')
        return -np.inf                  #loglikelihood calculation is not sucessful
    