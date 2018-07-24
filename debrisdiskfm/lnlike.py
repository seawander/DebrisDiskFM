from astropy.io import fits
import numpy as np
from . import fm_klip
from . import diskmodeling_Qr


def lnlike_hd191089(path_obs = None, path_model = None):
    
    ### Observations:
    if path_obs is None:
        path_obs = './data_observation/'
    stis_obs = fits.getdata(path_obs + 'STIS/HD-191089.fits')
    stis_obs_unc = fits.getdata(path_obs + 'STIS/HD-191089_NoiseMap.fits')
    
    nicmos_obs = fits.getdata(path_obs + 'NICMOS/HD-191089_NICMOS_F110W_Lib-84_KL-19_Signal.fits')
    nicmos_obs_unc = fits.getdata(path_obs + 'NICMOS/HD-191089_NICMOS_F110W_Lib-84_KL-19_NoiseMap.fits')
    
    gpi_obs = fits.getdata(path_obs + 'GPI/hd191089_gpi_smooth_mJy_arcsec2.fits')
    gpi_obs_unc = fits.getdata(path_obs + 'GPI/hd191089_gpi_smooth_mJy_arcsec2_noisemap.fits')
    
    ### (Forwarded) Models:
    if path_model is None:
        path_model = './test/'
    stis_model = fits.getdata(path_model + 'data_0.5852/RT.fits.gz')
    nicmos_model = fm_klip.klip_fm_main(path = path_model, angles= None)
    gpi_model = diskmodeling_Qr.diskmodeling_Qr_main(path = path_model)
    
    ### Chi-squared values
    chi2_stis = np.nansum(((stis_obs - stis_model)/stis_obs_unc)**2)
    chi2_nicmos = np.nansum(((nicmos_obs - nicmos_model)/nicmos_obs_unc)**2)
    chi2_gpi = np.nansum(((gpi_obs - gpi_model)/gpi_obs_unc)**2)
    
    return -0.5*(chi2_stis+chi2_nicmos+chi2_gpi)
