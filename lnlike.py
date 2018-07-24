import fm_klip
import diskmodeling_Qr
from astropy.io import fits
import numpy as np
import dependences
import image_registration

def lnlike_hd191089(path_obs = None, path_model = None, psfs = None, psf_cut_hw = None):
    """Return the log-likelihood for observed data and modelled data.
    Input:  path_obs: the path to the observed data
            path_model: the path to the (forwarded) models
            psfs: the point spread functions for forward modeling to simulate instrument response
            psf_cut_hw: the half-width of the PSFs if you would like to cut them to smaller sizes (size = 2*hw + 1)
            """
    
    
    ### Observations:
    if path_obs is None:
        path_obs = './data_observation/'
    stis_obs = fits.getdata(path_obs + 'STIS/calibrated/HD-191089_Signal_Jy_arcsec-2_oddSize.fits')
    stis_obs_unc = fits.getdata(path_obs + 'STIS/calibrated/HD-191089_NoiseMap_Jy_arcsec-2_oddSize.fits')
    stis_obs_unc[np.where(stis_obs_unc <=0)] = np.nan
    
    nicmos_obs = fits.getdata(path_obs + 'NICMOS/calibrated/HD-191089_NICMOS_F110W_Lib-84_KL-19_Signal.fits')
    nicmos_obs_unc = fits.getdata(path_obs + 'NICMOS/calibrated/HD-191089_NICMOS_F110W_Lib-84_KL-19_NoiseMap.fits')
    nicmos_obs_unc[np.where(nicmos_obs_unc <=0)] = np.nan
    
    gpi_obs = fits.getdata(path_obs + 'GPI/hd191089_gpi_smooth_mJy_arcsec2.fits')*1e3 #Turn it to Jy/arcsec^2
    gpi_obs_unc = fits.getdata(path_obs + 'GPI/hd191089_gpi_smooth_mJy_arcsec2_noisemap.fits')*1e3 #Turn it to Jy/arcsec^2
    gpi_obs_unc[np.where(gpi_obs_unc <=0)] = np.nan
    
    ### (Forwarded) Models:
    if path_model is None:
        path_model = './test/'
    if psfs is None:
        psf_stis = fits.getdata(path_obs + 'STIS/calibrated/STIS_6440K_tinyTIM_oddSize.fits')
        psf_nicmos = fits.getdata(path_obs + 'NICMOS/calibrated/')
        if psf_cut_hw is not None:
            psfs[0] = dependences.cutImage(psf_stis, psf_cut_hw) # a 7*7 PSF would need psf_cut_hw = 3 (then 3*2+1 = 7).
            psfs[1] = dependences.cutImage(psf_nicmos, psf_cut_hw) # a 7*7 PSF would need psf_cut_hw = 3
        else:
            psfs[0] = psf_stis
            psfs[1] = psf_nicmos
        
    stis_model = fits.getdata(path_model + 'data_0.5852/RT.fits.gz')
    stis_convolved = image_registration.fft_tools.convolve_nd.convolvend(stis_model, psfs[0])
    
    nicmos_model_forwarded = fm_klip.klip_fm_main(path = path_model, angles= None, psf = psfs[1])
    
    gpi_model = diskmodeling_Qr.diskmodeling_Qr_main(path = path_model, fwhm = 3.8) 
    #FWHM = 3.8 for GPI, as provided in Tom Esposito's HD35841 paper (Section: MCMC Modeling Procedure)
    
    ### Chi-squared values
    chi2_stis = np.nansum(((stis_obs - stis_model)/stis_obs_unc)**2)
    chi2_nicmos = np.nansum(((nicmos_obs - nicmos_model_forwarded)/nicmos_obs_unc)**2)
    chi2_gpi = np.nansum(((gpi_obs - gpi_model)/gpi_obs_unc)**2)
    
    return -0.5*(chi2_stis+chi2_nicmos+chi2_gpi)
