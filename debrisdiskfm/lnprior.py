import numpy as np

def lnprior_hd191089(var_names = None, var_values = None):
    """This code sets the prior for the MCMC modeling of the HD191089 system."""
    if var_names is None:
        var_names = ['inc', 'PA', 'm_disk', 
                     'Rc', 'R_in', 'alpha_in', 'alpha_out', 'porosity', 
                     'fmass_0', 'fmass_1', 
                     'a_min', 'Q_powerlaw', 'Vmax']
    if var_values is None:    
        var_values = [59.5, 70.3, -7, 
                         43.6, 20, 5.9,  -5.1, 0.1,
                        0.0, 0.0,
                        1.0, 3.5, 0.7]
    var_values = list(np.round(var_values, 3)) #round to 3 decimal digits
                        
    # The MCFOST definition of inclination and position angle is not what we have been using.

    theta = dict(zip(var_names, var_values))
    for var_name in var_names:
        if var_name == 'inc':
            if not (-5 < (theta['inc'] - 59.7) < 5):
                return -np.inf
        elif var_name == 'PA':
            if not (-5 < (theta['PA'] - 70) < 5):
                return -np.inf
        elif var_name == 'm_disk':
            if not (-12 < theta['m_disk'] < -4):
                return -np.inf
        elif var_name == 'Rc':
            if not (-10 < (theta['Rc'] - 45.3) < 10):
                return -np.inf
        elif var_name == 'R_in':
            if not (0 < theta['R_in'] < 45) or not (theta['R_in'] < theta['Rc']):
                return -np.inf
        elif var_name == 'alpha_in':
            if not (0 < theta['alpha_in'] < 5):
                return -np.inf
        elif var_name == 'alpha_out':
            if not (-15 < theta['alpha_out'] < 0):
                return -np.inf
        elif var_name == 'porosity':
            if not (0 <= theta['porosity'] <= 1):
                return -np.inf
        elif var_name == 'fmass_0':
            if not (0 <= theta['fmass_0'] <= 1) or (not (0 <= theta['fmass_1'] <= 1)) or (not (0 <= (theta['fmass_0'] + theta['fmass_1']) <= 1)):
                return -np.inf
        elif var_name == 'a_min':
            if not (-0.3 < theta['a_min'] < 2): #find minimum grain size > 0.5µm
                return -np.inf
        elif var_name == 'Q_powerlaw':
            if not (3 < theta['Q_powerlaw'] < 6):
                return -np.inf
        elif var_name == 'Vmax':
            if not (0 <= theta['Vmax'] <= 1):
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
           
    
    
def lnprior_hr4796aH2spf(var_names = None, var_values = None):
    """This code sets the prior for the MCMC modeling of the HR 4796A H2 SPF."""
    if var_names is None:
        var_names = ['inc', 'PA', 'm_disk', 
                     'Rc', 'R_in', 'alpha_in', 'R_out', 'alpha_out', 'porosity', 
                     'fmass_0', 'fmass_1', 
                     'a_min', 'Q_powerlaw', 'scale height', 'Vmax']
    if var_values is None:    
        var_values = [76.45, 27.1, -6, 
                     76.7, 72.2, 5.25,  91.7, -6.8, 0.2,
                    0.6, 0.2,
                    1.0, 3.5, 3.07, 0.6]
    var_values = list(np.round(var_values, 3)) #round to 3 decimal digits
                        
    # The MCFOST definition of inclination and position angle is not what we have been using.

    theta = dict(zip(var_names, var_values))
    for var_name in var_names:
        if var_name == 'inc':
            if not (-5 < (theta['inc'] - 76.45) < 5):
                return -np.inf
        elif var_name == 'PA':
            if not (-5 < (theta['PA'] - 27.1) < 5):
                return -np.inf
        elif var_name == 'm_disk':
            if not (-12 < theta['m_disk'] < -4):
                return -np.inf
        elif var_name == 'Rc':
            if not (-10 < (theta['Rc'] - 76.7) < 10):
                return -np.inf
        elif var_name == 'R_in':
            if not (0 < theta['R_in'] < 76.7) or not (theta['R_in'] < theta['Rc']):
                return -np.inf
        elif var_name == 'alpha_in':
            if not (0 < theta['alpha_in'] < 7):
                return -np.inf
        elif var_name == 'alpha_out':
            if not (-15 < theta['alpha_out'] < 0):
                return -np.inf
        elif var_name == 'porosity':
            if not (0 <= theta['porosity'] <= 1):
                return -np.inf
        elif var_name == 'fmass_0':
            if not (0 <= theta['fmass_0'] <= 1) or (not (0 <= theta['fmass_1'] <= 1)) or (not (0 <= (theta['fmass_0'] + theta['fmass_1']) <= 1)):
                return -np.inf
        elif var_name == 'a_min':
            if not (-0.3 < theta['a_min'] < 2): #find minimum dust size between 0.5µm and 100µm
                return -np.inf
        elif var_name == 'Q_powerlaw':
            if not (3 < theta['Q_powerlaw'] < 6):
                return -np.inf
        elif var_name == 'Vmax':
            if not (0 <= theta['Vmax'] <= 1):
                return -np.inf
    return 0

def lnprior_pds70keck(var_names = None, var_values = None):
    """This code sets the prior for the MCMC modeling of the Keck Lp image (3.8 micron)."""
    if var_names is None:
        var_names = ['inc', 'PA', 'm_disk', 
                     'Rc', 'R_in', 'alpha_in', 'R_out', 'alpha_out', 'porosity', 
                     'fmass_0', 'fmass_1', 
                     'a_min', 'Q_powerlaw', 'scale height', 'flaring exp']
    if var_values is None:    
        var_values = [49.7, -21.4, -7, 
                     67.8, 60, 2,  76, -2, 0.0,
                     0.0, 0.0,
                    -2.0, 3.5, 1.812, 1.0]
    var_values = list(np.round(var_values, 3)) #round to 3 decimal digits
                        
    # The MCFOST definition of inclination and position angle is not what we have been using.

    theta = dict(zip(var_names, var_values))
    for var_name in var_names:
        if var_name == 'inc':
            if not (-20 < (theta['inc'] - 65) < 20):
                return -np.inf
        elif var_name == 'PA':
            if not (-10 < (theta['PA'] + 21.4) < 10):
                return -np.inf
        elif var_name == 'm_disk':
            if not (-12 < theta['m_disk'] < -4):
                return -np.inf
        elif var_name == 'Rc':
            if not (-30 < (theta['Rc'] - 67.8) < 200):
                return -np.inf
        elif var_name == 'R_in':
            if not (theta['R_in'] < theta['Rc']): #not (0 < theta['R_in'] < 67.8) or 
                return -np.inf
        elif var_name == 'R_out':
            if not (theta['R_out'] > theta['Rc']): #not (67.8 < theta['R_out'] < 90) or 
                return -np.inf
        elif var_name == 'alpha_in':
            if not (0 < theta['alpha_in'] < 7):
                return -np.inf
        elif var_name == 'alpha_out':
            if not (-15 < theta['alpha_out'] < 0):
                return -np.inf
        elif var_name == 'porosity':
            if not (0 <= theta['porosity'] <= 1):
                return -np.inf
        elif var_name == 'fmass_0':
            if not (0 <= theta['fmass_0'] <= 1) or (not (0 <= theta['fmass_1'] <= 1)) or (not (0 <= (theta['fmass_0'] + theta['fmass_1']) <= 1)):
                return -np.inf
        elif var_name == 'a_min':
            if not (-3 < theta['a_min'] < 2): #find minimum dust size between 0.5µm and 100µm
                return -np.inf
        elif var_name == 'Q_powerlaw':
            if not (1 < theta['Q_powerlaw'] < 10):
                return -np.inf
        elif var_name == 'flaring exp':
            if not (0 < theta['flaring exp'] < 5):
                return -np.inf
    return 0