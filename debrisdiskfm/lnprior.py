import numpy as np

def lnprior_hd191089(var_names = None, var_value = None):
    """This code sets the prior for the MCMC modeling of the HD191089 system."""
    if var_names is None:
        var_names = ['inc', 'PA', 'm_disk', 
                     'Rc', 'R_in', 'alpha_in', 'alpha_out', 'porosity', 
                     'fmass_0', 'fmass_1', 
                     'a_min', 'Q_powerlaw']
    if var_value is None:    
        var_value = [59.7, 70, 1e-7, 
                         45.3, 20, 3.5,  -3.5, 0.1,
                        0.05, 0.9,
                        1.0, 3.5]
                        
    # The MCFOST definition of inclination and position angle is not what we have been using.

    theta = dict(zip(var_names, var_value))
    
    if ((-10 < (theta['inc'] - 59.7) < 10) and \
       (-10 < (theta['PA'] - 70) < 10) and \
       (1e-10 < theta['m_disk'] < 1e-5) and \
       (-20 < (theta['Rc'] - 45.3) < 20) and \
       (0 < theta['R_in'] < 65.3) and \
       (0 < theta['alpha_in'] < 5) and \
       (-15 < theta['alpha_out'] < 0) and \
       (0 < theta['porosity'] < 1) and \
       (0 < theta['fmass_0'] < 1) and \
       (0 < theta['fmass_1'] < 1) and \
       (0 < (theta['fmass_0'] + theta['fmass_1']) <= 1) and \
       (0 < theta['a_min'] < 5) and \
       (0 < theta['Q_powerlaw'] < 5)):
        return 0
    else:
        return -np.inf
           
    
    
