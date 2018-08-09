import numpy as np

def lnprior_hd191089(var_names = None, var_values = None):
    """This code sets the prior for the MCMC modeling of the HD191089 system."""
    if var_names is None:
        var_names = ['inc', 'PA', 'm_disk', 
                     'Rc', 'R_in', 'alpha_in', 'alpha_out', 'porosity', 
                     'fmass_0', 'fmass_1', 
                     'a_min', 'Q_powerlaw']
    if var_values is None:    
        var_values = [59.7, 70, -7, 
                         45.3, 20, 3.5,  -3.5, 0.1,
                        0.05, 0.9,
                        1.0, 3.5]
    var_values = list(np.round(var_values, 3)) #round to 3 decimal digits
                        
    # The MCFOST definition of inclination and position angle is not what we have been using.

    theta = dict(zip(var_names, var_values))
    for var_name in var_names:
        if var_name == 'inc':
            if not (-20 < (theta['inc'] - 59.7) < 20):
                return -np.inf
        elif var_name == 'PA':
            if not (-20 < (theta['PA'] - 70) < 20):
                return -np.inf
        elif var_name == 'm_disk':
            if not (-10 < theta['m_disk'] < -4):
                return -np.inf
        elif var_name == 'Rc':
            if not (-30 < (theta['Rc'] - 45.3) < 30):
                return -np.inf
        elif var_name == 'R_in':
            if not (0 < theta['R_in'] < 100):
                return -np.inf
        elif var_name == 'alpha_in':
            if not (0 < theta['alpha_in'] < 10):
                return -np.inf
        elif var_name == 'alpha_out':
            if not (-15 < theta['alpha_out'] < 0):
                return -np.inf
        elif var_name == 'porosity':
            if not (0 < theta['porosity'] < 1):
                return -np.inf
        elif var_name == 'fmass_0':
            if not (0 < theta['fmass_0'] < 1) or (not (0 < theta['fmass_1'] < 1)) or (not (0 < (theta['fmass_0'] + theta['fmass_1']) <= 1)):
                return -np.inf
        elif var_name == 'a_min':
            if not (0 < theta['a_min'] < 50):
                return -np.inf
        elif var_name == 'Q_powerlaw':
            if not (0 < theta['Q_powerlaw'] < 10):
                return -np.inf
    return 0
    # if ((-10 < (theta['inc'] - 59.7) < 10) and \
    #    (-10 < (theta['PA'] - 70) < 10) and \
    #    (1e-10 < theta['m_disk'] < 1e-5) and \
    #    (-20 < (theta['Rc'] - 45.3) < 20) and \
    #    (0 < theta['R_in'] < 65.3) and \
    #    (0 < theta['alpha_in'] < 5) and \
    #    (-15 < theta['alpha_out'] < 0) and \
    #    (0 < theta['porosity'] < 1) and \
    #    (0 < theta['fmass_0'] < 1) and \
    #    (0 < theta['fmass_1'] < 1) and \
    #    (0 < (theta['fmass_0'] + theta['fmass_1']) <= 1) and \
    #    (0 < theta['a_min'] < 5) and \
    #    (0 < theta['Q_powerlaw'] < 5)):
    #     return 0
    # else:
    #     return -np.inf
           
    
    
