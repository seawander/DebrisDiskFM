from . import lnprior
from . import lnlike
from . import mcfostRun
import numpy as np
import shutil

def lnpost_hd191089(var_values = None, var_names = None, path_obs = None, path_model = None, calcSED = False, hash_address = True):
    """Returns the log-posterior probability (post = prior * likelihood, thus lnpost = lnprior + lnlike)
    for a given parameter combination.
    Input:  var_values: number array, values for var_names. Refer to mcfostRun() for details. 
                'It is important that the first argument of the probability function is the position of a single walker (a N dimensional numpy array).' (http://dfm.io/emcee/current/user/quickstart/)
            var_names: string array, names of variables. Refer to mcfostRun() for details.
            path_obs: string, address where the observed values are stored.
            path_model: string, address where you would like to store the MCFOST raw models (not forwarded ones).
            calcSED: boolean, whether to calculate the SED of the system.
    Output: log-posterior probability."""
    ln_prior = lnprior.lnprior_hd191089(var_names = var_names, var_values = var_values)
    if not np.isfinite(ln_prior):
        return -np.inf
        
    run_flag = 1
    try:
        if hash_address:
            run_flag, hash_string = mcfostRun.run_hd191089(var_names = var_names, var_values = var_values, paraPath = path_model, calcSED = calcSED, calcImage = True, hash_address = hash_address)
        else:
            run_flag = mcfostRun.run_hd191089(var_names = var_names, var_values = var_values, paraPath = path_model, calcSED = calcSED, calcImage = True, hash_address = hash_address)
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
            ln_likelihood = lnlike.lnlike_hd191089(path_obs = path_obs, path_model = path_model, hash_address = hash_address, hash_string = hash_string)
        else:
            ln_likelihood = lnlike.lnlike_hd191089(path_obs = path_obs, path_model = path_model, hash_address = hash_address)
        
        return ln_prior + ln_likelihood
    except:
        if hash_address:               
            shutil.rmtree(path_model[:-1] + hash_string + '/')
        return -np.inf                  #loglikelihood calculation is not sucessful
    