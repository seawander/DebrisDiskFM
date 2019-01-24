from astropy.io import fits
import numpy as np

# returns the Qr model

def radialStokes(modeldata = None, mcfostGenerated = True, q = None, u = None):
    if mcfostGenerated:
        #if modeldata has the structure of MCFOST image output
        q = modeldata[1, 0, 0]
        u = modeldata[2, 0, 0] # load MCFOST Polarimetry data
    
    x_cen = (q.shape[1] - 1)/2.0
    y_cen = (q.shape[0] - 1)/2.0
    
    x_range, y_range = np.arange(q.shape[1]), np.arange(q.shape[0])
    x = np.ones(q.shape)
    y = np.ones(q.shape)
    
    x[:] = x_range
    y[:] = y_range
    y = y.T
    
    phi = np.arctan((y-y_cen)*1.0/(x-x_cen))        # This angle is the counterclockwise angle from x-axis, 
                                                    # which is 90-deg behind the definition in Monnier et al. (2019) https://arxiv.org/pdf/1901.02467.pdf
    qr = q * np.cos(2 * phi) + u * np.sin(2 * phi)  # The following two lines are the Q_phi and U_phi after the adjustment of 90 degree.
    ur = -q * np.sin(2 * phi) + u * np.cos(2 * phi) # They are not changed from the previous version.
    qr[np.where(np.isnan(qr))] = 0
    ur[np.where(np.isnan(ur))] = 0
    return qr, ur

def diskmodeling_Qr_main(path = './test/', fwhm = None):
    """"""
    
    mcfost_disk_pol = fits.getdata(path + 'data_1.65/RT.fits.gz')
    Qr, Ur = radialStokes(mcfost_disk_pol)
    del Ur
    if fwhm is not None:
        import scipy
        sigma = fwhm/2.355 # Convert FWHM to sigma for Gaussian/Normal distribution. https://en.wikipedia.org/wiki/Full_width_at_half_maximum
        Qr_convolved = scipy.ndimage.filters.gaussian_filter(Qr, sigma)
        Qr = Qr_convolved
        del Qr_convolved
    return Qr

# test = diskmodeling_Qr_main()
