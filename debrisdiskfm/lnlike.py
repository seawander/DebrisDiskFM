from astropy.io import fits
import numpy as np
from . import fm_klip
from . import diskmodeling_Qr
from . import dependencies
import image_registration
import astropy.units as units
from . import lnprior

def convertMCFOSTdataToJy(data, wavelength, spatialUnit = 'arcsec', spatialResolution = None):
    """Convert data in MCFOST units into Jansky/pixel or Jansky/arcsec^2:
    Input:  data: 2D array, MCFOST-generated data.
            wavelength: float, wavelength of the data to be converted in micron.
            spatialUnit: string, default = 'arcsec', which will convert it to arcsec^{-2}; 'pixel' will be pixel^{-1}.
            spatialResolution: float, unit is arcsec/pixel. This will be used to convert the units to arcsec^{-2}.
    Output: converted data.
    """
    data_with_units = data*units.W/units.m**2/units.pixel
    frequency = ((3e8*units.m/units.s)/(wavelength*units.micron)).to(units.hertz)
    data_with_units_jy = (data_with_units/frequency).to(units.Jansky/units.pixel)   # convert to Jansky/pixel
    if spatialUnit == 'arcsec':
        data_with_units_jy /= spatialResolution**2                                  # convert to Jansky/arcsec^2
    return data_with_units_jy.value
    
def chi2(data, data_unc, model, lnlike = True):
    """Calculate the chi-squared value or log-likelihood for given data and model. 
    Note: if data_unc has values <= 0, they will be ignored and replaced by NaN.
    Input:  data: 2D array, observed data.
            data_unc: 2D array, uncertainty/noise map of the observed data.
            lnlike: boolean, if True, then the log-likelihood is returned.
    Output: chi2: float, chi-squared or log-likelihood value."""
    data_unc[np.where(data_unc <= 0)] = np.nan
    chi2 = np.nansum(((data-model)/data_unc)**2)
    if lnlike:
        loglikelihood = -0.5*np.log(2*np.pi)*np.count_nonzero(~np.isnan(data_unc)) - 0.5*chi2 - np.nansum(np.log(data_unc))
        # -n/2*log(2pi) - 1/2 * chi2 - sum_i(log sigma_i) 
        return loglikelihood
    return chi2

def lnlike_hd191089(path_obs = None, path_model = None, psfs = None, psf_cut_hw = None):
    """Return the log-likelihood for observed data and modelled data.
    Input:  path_obs: the path to the observed data
            path_model: the path to the (forwarded) models
            psfs: the point spread functions for forward modeling to simulate instrument response
            psf_cut_hw: the half-width of the PSFs if you would like to cut them to smaller sizes (size = 2*hw + 1)
    Output: log-likelihood
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
    
    gpi_obs = fits.getdata(path_obs + 'GPI/calibrated/hd191089_gpi_smooth_mJy_arcsec2.fits')/1e3 #Turn it to Jy/arcsec^2
    gpi_obs_unc = fits.getdata(path_obs + 'GPI/calibrated/hd191089_gpi_smooth_mJy_arcsec2_noisemap.fits')/1e3 #Turn it to Jy/arcsec^2
    gpi_obs_unc[np.where(gpi_obs_unc <=0)] = np.nan
    
    resolution_stis = 0.05078 # arcsec/pixel
    resolution_gpi = 14.166e-3
    resolution_nicmos = 0.07565
    
    ### (Forwarded) Models:
    if path_model is None:
        path_model = './test/'
    if psfs is None:
        psf_stis = fits.getdata(path_obs + 'STIS/calibrated/STIS_6440K_tinyTIM_oddSize.fits')
        psf_nicmos = fits.getdata(path_obs + 'NICMOS/calibrated/NICMOS_Era2_F110W_oddSize.fits')
        psfs = [psf_stis, psf_nicmos]
        
        if psf_cut_hw is not None:
            psfs[0] = dependencies.cutImage(psf_stis, psf_cut_hw)   # a 7*7 PSF would need psf_cut_hw = 3 (then 3*2+1 = 7).
            psfs[1] = dependencies.cutImage(psf_nicmos, psf_cut_hw) # a 7*7 PSF would need psf_cut_hw = 3
            psfs[0] /= np.nansum(psfs[0])
            psfs[1] /= np.nansum(psfs[1])
        else:
            psfs[0] = psf_stis
            psfs[1] = psf_nicmos
            psfs[0] /= np.nansum(psfs[0])
            psfs[1] /= np.nansum(psfs[1])
        
        
    # convert the MCFOST units to Jy/arcsec^2, and calculate individual chi2
    stis_model = fits.getdata(path_model + 'data_0.5852/RT.fits.gz')[0, 0, 0]
    stis_model[int((stis_model.shape[0]-1)/2)-2:int((stis_model.shape[0]-1)/2)+3, int((stis_model.shape[1]-1)/2)-2:int((stis_model.shape[1]-1)/2)+3] = 0
    stis_convolved = image_registration.fft_tools.convolve_nd.convolvend(stis_model, psfs[0])
    stis_model = convertMCFOSTdataToJy(stis_convolved, wavelength = 0.5852, spatialResolution = resolution_stis) #convert to Jansky/arscec^2    
    chi2_stis = chi2(stis_obs, stis_obs_unc, stis_model, lnlike = True) #return loglikelihood value for STIS
    fits.writeto('/Users/binren/Desktop/test2.fits', stis_model, clobber = True)
    
    
    nicmos_model_forwarded = fm_klip.klip_fm_main(path = path_model, angles= None, psf = psfs[1]) # already convolved
    nicmos_model = convertMCFOSTdataToJy(nicmos_model_forwarded, wavelength = 1.12347, spatialResolution = resolution_nicmos) #convert to Jansky/arscec^2
    chi2_nicmos = chi2(nicmos_obs_unc, nicmos_obs_unc, nicmos_model, lnlike = True) #return loglikelihood value for NICMOS   
    fits.writeto('/Users/binren/Desktop/test2.fits', nicmos_model, clobber = True)
    
    gpi_model = diskmodeling_Qr.diskmodeling_Qr_main(path = path_model, fwhm = 3.8)
    # FWHM = 3.8 for GPI, as provided in Tom Esposito's HD35841 paper (Section: MCMC Modeling Procedure)
    gpi_model = convertMCFOSTdataToJy(gpi_model, wavelength = 1.65, spatialResolution = resolution_gpi) #convert to Jansky/arscec^2
    chi2_gpi = chi2(gpi_obs, gpi_obs_unc, gpi_model, lnlike = True) #return loglikelihood value for GPI
    fits.writeto('/Users/binren/Desktop/test3.fits', gpi_model, clobber = True)
    
    
    
    return (chi2_stis+chi2_nicmos+chi2_gpi) #Returns the loglikelihood
