from astropy.io import fits
import numpy as np

# returns the Qr model

def radialStokes(modeldata = None, mcfostGenerated = True, q = None, u = None):
    if mcfostGenerated:
        #if modeldata has the structure of MCFOST image output
        q = modeldata[1, 0, 0]
        u = modeldata[2, 0, 0]
        
    x_cen = (q.shape[1] - 1)/2.0
    y_cen = (q.shape[0] - 1)/2.0
    
    x_range, y_range = np.arange(q.shape[1]), np.arange(q.shape[0])
    x = np.ones(q.shape)
    y = np.ones(q.shape)
    
    x[:] = x_range
    y[:] = y_range
    y = y.T
    
    phi = np.arctan((y-y_cen)*1.0/(x-x_cen))
    qr = q * np.cos(2 * phi) + u * np.sin(2 * phi)
    ur = -q * np.sin(2 * phi) + u * np.cos(2 * phi)
    
    return qr, ur

def diskmodeling_Qr_main(path = './test/'):
    mcfost_disk_pol = fits.getdata(path + 'data_1.65/RT.fits.gz')
    Qr, Ur = radialStokes(mcfost_disk_pol)
    del Ur
    return Qr

# test = diskmodeling_Qr_main()
