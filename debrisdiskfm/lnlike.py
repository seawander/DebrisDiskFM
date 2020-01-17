from astropy.io import fits
import numpy as np
from . import fm_klip
from . import diskmodeling_Qr
from . import dependencies
import image_registration
import astropy.units as units
from . import lnprior
import shutil

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
    
def chi2_1dinterp(angles, data, data_unc, model, lnlike = True):
    """Calculate the chi-squared value or log-likelihood for given data and model. 
    Note: if data_unc has values <= 0, they will be ignored and replaced by NaN.
    Input:  
            angles: 1D array, observed SPF angles.
            data: 1D array, observed SPF (normalized).
            data_unc: 1D array, uncertainty of the observed SPF.
            model: 1D array, MCFOST model array.
            lnlike: boolean, if True, then the log-likelihood is returned.
    Output: 
            chi2: float, chi-squared or log-likelihood value."""
    from scipy.interpolate import interp1d as interp1d
    model_mcfost = np.copy(model)
    func_spf = interp1d(np.arange(0, 181, 1), model_mcfost[:, 0])
    model = func_spf(angles)
    
    data_unc[np.where(data_unc <= 0)] = np.nan
    chi2 = np.nansum(((data-model)/data_unc)**2)
    
    if lnlike:
        loglikelihood = -0.5*np.log(2*np.pi)*np.count_nonzero(~np.isnan(data_unc)) - 0.5*chi2 - np.nansum(np.log(data_unc))
        # -n/2*log(2pi) - 1/2 * chi2 - sum_i(log sigma_i) 
        return loglikelihood
    return chi2

def lnlike_hd191089(path_obs = None, path_model = None, psfs = None, psf_cut_hw = None, hash_address = False, delete_model = True, hash_string = None, return_model_only = False, STIS = True, NICMOS = True, GPI = True):
    """Return the log-likelihood for observed data and modelled data.
    Input:  path_obs: the path to the observed data
            path_model: the path to the (forwarded) models
            psfs: the point spread functions for forward modeling to simulate instrument response
            psf_cut_hw: the half-width of the PSFs if you would like to cut them to smaller sizes (size = 2*hw + 1)
            hash_address: whether to hash the address based on the values, if True, then the address should be provided by `hash_string'
            delete_model: whether to delete the models. True by default.
            return_model_only: only return the forwarded models for debug/grid-modeling purpose
    Output: log-likelihood
            """
    ### Observations:
    if path_obs is None:
        path_obs = './data_observation/'
    if STIS:
        stis_obs = fits.getdata(path_obs + 'STIS/calibrated/HD-191089_Signal_Jy_arcsec-2_oddSize.fits')
        stis_obs_unc = fits.getdata(path_obs + 'STIS/calibrated/HD-191089_NoiseMap_Jy_arcsec-2_oddSize.fits')
        stis_obs_unc[np.where(stis_obs_unc <=0)] = np.nan
        mask_stis = fits.getdata(path_obs + 'STIS/calibrated/mask_stis.fits')
    if NICMOS:
        nicmos_obs = fits.getdata(path_obs + 'NICMOS/calibrated/HD-191089_NICMOS_F110W_Lib-84_KL-19_Signal-Jy_arcsec-2.fits')
        nicmos_obs_unc = fits.getdata(path_obs + 'NICMOS/calibrated/HD-191089_NICMOS_F110W_Lib-84_KL-19_NoiseMap-Jy_arcsec-2.fits')
        nicmos_obs_unc[np.where(nicmos_obs_unc <=0)] = np.nan
        mask_nicmos = fits.getdata(path_obs + 'NICMOS/calibrated/mask_nicmos.fits')
    if GPI:
        gpi_obs = fits.getdata(path_obs + 'GPI/calibrated/hd191089_gpi_smooth_mJy_arcsec2.fits')/1e3 #Turn it to Jy/arcsec^2
        gpi_obs_unc = fits.getdata(path_obs + 'GPI/calibrated/hd191089_gpi_smooth_mJy_arcsec2_noisemap.fits')/1e3 #Turn it to Jy/arcsec^2
        gpi_obs_unc[np.where(gpi_obs_unc <=0)] = np.nan
        mask_gpi = fits.getdata(path_obs + 'GPI/calibrated/mask_gpi.fits')
    
    resolution_stis = 0.05078 # arcsec/pixel
    resolution_gpi = 14.166e-3
    resolution_nicmos = 0.07565
    
    ### (Forwarded) Models:
    if path_model is None:
        path_model = './test/'
    if hash_address:
        if hash_string is None:
            print('Please provide the hash string if you set hash_address = True!')
            return -np.inf     
        path_model = path_model[:-1] + hash_string + '/'
    try:    
        if psfs is None:
            psfs = [None, None]
            psf_stis_raw = fits.getdata(path_obs + 'STIS/calibrated/STIS_6440K_tinyTIM_oddSize.fits')
            psf_stis = np.zeros(psf_stis_raw.shape)
            psf_stis[148:167, 148:167] = psf_stis_raw[148:167, 148:167] #focus only on the 19x19 PSF region as done in calculating the STIS BAR5 contrast.
            psf_nicmos_raw = fits.getdata(path_obs + 'NICMOS/calibrated/NICMOS_Era2_F110W_oddSize.fits')
            psf_nicmos = np.zeros(psf_nicmos_raw.shape)
            psf_nicmos[60:79, 60:79] = psf_nicmos_raw[60:79, 60:79] #focus only on the 19x19 PSF region as for the STIS data.
        
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
    except:
        pass        
    # convert the MCFOST units to Jy/arcsec^2, and calculate individual chi2

    if STIS:
        stis_model = fits.getdata(path_model + 'data_0.58/RT.fits.gz')[0, 0, 0]
        if np.nansum(np.isnan(stis_model)) != 0:
            chi2_stis = -np.inf
        else:
            stis_model[int((stis_model.shape[0]-1)/2)-2:int((stis_model.shape[0]-1)/2)+3, int((stis_model.shape[1]-1)/2)-2:int((stis_model.shape[1]-1)/2)+3] = 0
            stis_convolved = image_registration.fft_tools.convolve_nd.convolvend(stis_model, psfs[0])
            stis_model = convertMCFOSTdataToJy(stis_convolved, wavelength = 0.58, spatialResolution = resolution_stis) #convert to Jansky/arscec^2
            # mask_stis = dependencies.annulusMask(stis_model.shape[0], r_in = 0, r_out=30) #define your own mask here
            mask_stis[np.isnan(stis_obs_unc)] = 0
            chi2_stis = chi2(stis_obs, stis_obs_unc*mask_stis, stis_model, lnlike = True) #return loglikelihood value for STIS
    else:
        chi2_stis = 0
    if NICMOS:
        nicmos_model_forwarded = fm_klip.klip_fm_main(path = path_model, path_obs = path_obs, angles= None, psf = psfs[1]) # already convolved
        nicmos_model = convertMCFOSTdataToJy(nicmos_model_forwarded, wavelength = 1.12, spatialResolution = resolution_nicmos) #convert to Jansky/arscec^2
        # mask_nicmos = dependencies.annulusMask(nicmos_model.shape[0], r_in = 0, r_out = 20) #define your own mask here
        mask_nicmos[np.isnan(nicmos_obs_unc)] = 0
        
        chi2_nicmos = chi2(nicmos_obs, nicmos_obs_unc*mask_nicmos, nicmos_model, lnlike = True) #return loglikelihood value for NICMOS       
    else:
        chi2_nicmos = 0
    if GPI:
        gpi_model = diskmodeling_Qr.diskmodeling_Qr_main(path = path_model, fwhm = 3.8)
        # mask_gpi = dependencies.annulusMask(gpi_model.shape[0], r_in = 15, r_out = 85)   #define your own mask here
        if np.nansum(np.isnan(gpi_model)) != 0:
            chi2_gpi = -np.inf
        else:
            # FWHM = 3.8 for GPI, as provided in Tom Esposito's HD35841 paper (Section: MCMC Modeling Procedure)
            gpi_model = convertMCFOSTdataToJy(gpi_model, wavelength = 1.65, spatialResolution = resolution_gpi) #convert to Jansky/arscec^2
            chi2_gpi = chi2(gpi_obs*mask_gpi, gpi_obs_unc*mask_gpi, gpi_model, lnlike = True) #return loglikelihood value for GPI             #NOTE: Magic number of 5 to boost the SNR is used!
    else:
        chi2_gpi = 0

        
    if hash_address and delete_model:    #delete the temporary MCFOST models
        shutil.rmtree(path_model)
    
    lnlike_total = chi2_stis+chi2_nicmos+chi2_gpi
    
    if np.isfinite(lnlike_total):
        if return_model_only:
            if STIS and NICMOS and GPI:
                return stis_model, nicmos_model, gpi_model*mask_gpi
            if STIS and NICMOS:
                return stis_model, nicmos_model
            if GPI:
                return gpi_model*mask_gpi
    
        return  lnlike_total #Returns the loglikelihood
    else:
        return -np.inf


def lnlike_hr4796aH2spf(path_obs = None, path_model = None, hash_address = False, delete_model = True, hash_string = None, return_model_only = False):
    """Return the log-likelihood for observed data and modelled data.
    Input:  path_obs: the path to the observed data
            path_model: the path to the (forwarded) models
            hash_address: whether to hash the address based on the values, if True, then the address should be provided by `hash_string'
            delete_model: whether to delete the models. True by default.
            return_model_only: only return the forwarded models for debug/grid-modeling purpose
    Output: log-likelihood
            """
    ### Observations:
    if path_obs is None:
        path_obs = './data_spf/'
    
    data_spf = fits.getdata(path_obs + 'best_spf_sphereh2.fits')    
    spf_angles = np.copy(data_spf[0])
    spf_obs = np.copy(data_spf[1])
    spf_obs_unc = np.copy(data_spf[2])
    factor_norm = spf_obs[np.where(spf_angles == 90)] #normalization at 90 degree
    spf_obs_unc /= factor_norm
    spf_obs /= factor_norm
    
    ### (Forwarded) Models:
    if path_model is None:
        path_model = './mcfost_models/'
    if hash_address:
        if hash_string is None:
            print('Please provide the hash string if you set hash_address = True!')
            return -np.inf     
        path_model = path_model[:-1] + hash_string + '/'
        
    spf_model_mcfost = fits.getdata(path_model + 'data_dust/phase_function.fits.gz')
    spf_model_mcfost /= spf_model_mcfost[90] # normalized at 90 degree angle
    
    chi2_spf = chi2_1dinterp(spf_angles, spf_obs, spf_obs_unc, spf_model_mcfost, lnlike = True)
    
        
    if hash_address and delete_model:    #delete the temporary MCFOST models
        shutil.rmtree(path_model)
    
    return  chi2_spf #Returns the loglikelihood
    
    
def lnlike_pds70keck(path_obs = None, path_model = None, hash_address = False, delete_model = True, hash_string = None, return_model_only = False, data_input_info = None, writemodel = False):
    """Return the log-likelihood for observed data and modelled data.
    Input:  path_obs: the path to the observed data
            path_model: the path to the (forwarded) models
            hash_address: whether to hash the address based on the values, if True, then the address should be provided by `hash_string'
            delete_model: whether to delete the models. True by default.
            return_model_only: only return the forwarded models for debug/grid-modeling purpose
            data_input_info: a class object containign input data, uncertainty, mask, etc. See row 270 for a setup example. 
            writemodel: write model in the model folder for easy comparison
    Output: log-likelihood
            """
    ### Observations:
    if data_input_info is None: 
        print("Reading the observation each time, this might be redundant. See the else sentences in this block...")
        if path_obs is None:
            path_obs = './reduction-bren/'
        
        data_obs = fits.getdata(path_obs + 'result_median_3components.fits')
        unc_obs = fits.getdata(path_obs + 'result_std_3components.fits')
        components_klip_obs = fits.getdata(path_obs + 'components3_0to2.fits')

        mask_obs = fits.getdata(path_obs + 'mask_161x161_in16_out80.fits')

        mask_planet = fits.getdata(path_obs + 'mask_161x161_in16_out80_plus_planets.fits')
        mask_calc = np.copy(mask_planet)
        mask_calc[mask_calc < 1] = np.nan

        angles = fits.getdata(path_obs + 'pyklip_parangs.fits')

        psf_keck = fits.getdata(path_obs + 'psf_noscale.fits')
        if np.nansum(psf_keck) > 1:
            psf_keck /= np.nansum(psf_keck)
    else:
        data_obs = np.copy(data_input_info.data_obs)
        unc_obs = np.copy(data_input_info.unc_obs)
        components_klip_obs = np.copy(data_input_info.components_klip)
        mask_obs = np.copy(data_input_info.mask_obs)
        mask_planet = np.copy(data_input_info.mask_planet)
        mask_calc = np.copy(data_input_info.mask_calc)
        angles = np.copy(data_input_info.angles)
        psf_keck = np.copy(data_input_info.psf)
        # See the following for a sample setup with the class object --- do it before calling the posterior function
        # This is designed to speed up the calculations by reading the observations for only once.
        # >>> code start 1
        # class data_input:
        #     def __init__(self, path_obs = None, data_obs = None, unc_obs = None, mask_obs = None, mask_planet = None,
        #                 psf = None, components_klip = None, angles = None):
        #         if path_obs is not None:
        #             self.data_obs = fits.getdata(path_obs + 'result_median_3components.fits')
        #             self.unc_obs = fits.getdata(path_obs + 'result_std_3components.fits')
        #             self.components_klip = fits.getdata(path_obs + 'components3_0to2.fits')
        #
        #             self.mask_obs = fits.getdata(path_obs + 'mask_161x161_in16_out80.fits')
        #
        #             self.mask_planet = fits.getdata(path_obs + 'mask_161x161_in16_out80_plus_planets.fits')
        #             self.mask_calc = np.copy(self.mask_planet)
        #             self.mask_calc[np.where(self.mask_calc < 1)] = np.nan
        #
        #             self.angles = fits.getdata(path_obs + 'pyklip_parangs.fits')
        #
        #             self.psf = fits.getdata(path_obs + 'psf_noscale.fits')
        #             self.psf /= np.nansum(self.psf)
        # <<< code end 1
        # Then set the object up as:
        # >>> code start 2
        # data_input_info = data_input(path_obs = './reduction-bren/')
        # <<< code end 2

    ### (Forwarded) Models:
    if path_model is None:
        path_model = './mcfost_models/'
    if hash_address:
        if hash_string is None:
            print('Please provide the hash string if you set hash_address = True!')
            return -np.inf     
        path_model = path_model[:-1] + hash_string + '/'

    model_mcfost = fits.getdata(path_model + 'data_3.8/RT.fits.gz')*8e20

    if len(model_mcfost.shape) == 5:
        model_mcfost = model_mcfost[0, 0, 0]

    model_mcfost[(model_mcfost.shape[0] - 1)//2, (model_mcfost.shape[1] - 1)//2] = 0

    cube = dependencies.rotateCube(model_mcfost, angle=-angles, mask = mask_obs, maskedNaN=True, outputMask=False)
    cube_convoled = np.array([image_registration.fft_tools.convolve_nd.convolvend(cube[i], psf_keck) for i in range(cube.shape[0])])
    cube_reduced = np.array([fm_klip.klip(cube_convoled[i], components_klip_obs, mask = mask_obs, cube=False) for i in range(cube.shape[0])])
    reduced_derotated =  dependencies.rotateCube(cube_reduced, angle=angles, mask = mask_obs, maskedNaN=True, outputMask=False)
    model_fm = np.nanmedian(reduced_derotated, axis = 0)
        
    lnlike_value = chi2(data_obs*mask_calc, unc_obs, model_fm, lnlike=True)
    
    if writemodel:
        fits.writeto(path_model + 'model_fm.fits', model_fm, overwrite = True)
        
    if hash_address and delete_model:    #delete the temporary MCFOST models only when the string is hashed
        shutil.rmtree(path_model)
    
    return  lnlike_value #Returns the loglikelihood
    
def lnlike_pds70keck_ADI(path_obs = None, path_model = None, hash_address = False, delete_model = True, hash_string = None, return_model_only = False, data_input_info = None, writemodel = False):
    """Return the log-likelihood for observed data and modelled data.
    Input:  path_obs: the path to the observed data
            path_model: the path to the (forwarded) models
            hash_address: whether to hash the address based on the values, if True, then the address should be provided by `hash_string'
            delete_model: whether to delete the models. True by default.
            return_model_only: only return the forwarded models for debug/grid-modeling purpose
            data_input_info: a class object containign input data, uncertainty, mask, etc. See row 270 for a setup example. 
            writemodel: write model in the model folder for easy comparison
    Output: log-likelihood
            """
    ### Observations:
    if data_input_info is None: 
        print("Reading the observation each time, this might be redundant. See the else sentences in this block...")
        if path_obs is None:
            path_obs = './reduction-bren/'
        
        data_obs = fits.getdata(path_obs + 'result_median_3components.fits')
        unc_obs = fits.getdata(path_obs + 'result_std_3components.fits')
        obs_raw = fits.getdata(path_obs + 'data_cut.fits')

        mask_obs = fits.getdata(path_obs + 'mask_161x161_in16_out80.fits')

        mask_planet = fits.getdata(path_obs + 'mask_161x161_in16_out80_plus_planets.fits')
        mask_disk = fits.getdata(path_obs + 'mask_disk.fits')
        mask_calc = np.copy(mask_planet)*mask_disk
        mask_calc[mask_calc < 1] = np.nan

        angles = fits.getdata(path_obs + 'pyklip_parangs.fits')

        psf_keck = fits.getdata(path_obs + 'psf_noscale.fits')
        if np.nansum(psf_keck) > 1:
            psf_keck /= np.nansum(psf_keck)
    else:
        data_obs = np.copy(data_input_info.data_obs)
        unc_obs = np.copy(data_input_info.unc_obs)
        obs_raw = np.copy(data_input_info.obs_raw)
        mask_obs = np.copy(data_input_info.mask_obs)
        mask_planet = np.copy(data_input_info.mask_planet)
        mask_calc = np.copy(data_input_info.mask_calc)
        mask_calc[mask_calc < 1] = np.nan
        angles = np.copy(data_input_info.angles)
        psf_keck = np.copy(data_input_info.psf)
        map_transmission = np.copy(data_input_info.map_transmission)
        
        # See the following for a sample setup with the class object --- do it before calling the posterior function
        # This is designed to speed up the calculations by reading the observations for only once.
        # >>> code start 1
        # class data_input:
        #     def __init__(self, path_obs = None, data_obs = None, unc_obs = None, mask_obs = None, mask_planet = None,
        #                 psf = None, components_klip = None, angles = None):
        #         if path_obs is not None:
        #             self.data_obs = fits.getdata(path_obs + 'result_median_3components.fits')
        #             self.unc_obs = fits.getdata(path_obs + 'result_std_3components.fits')
        #             self.obs_raw = fits.getdata(path_obs + 'data_cut.fits')
        #
        #             self.mask_obs = fits.getdata(path_obs + 'mask_161x161_in16_out80.fits')
        #
        #             self.mask_planet = fits.getdata(path_obs + 'mask_161x161_in16_out80_plus_planets.fits')
        #             self.mask_disk = fits.getdata(path_obs + 'mask_disk.fits')
        #             self.mask_calc = np.copy(self.mask_planet) * mask_disk
        #             self.mask_calc[np.where(self.mask_calc < 1)] = np.nan
        #
        #             self.map_transmission = fits.getdata(path_obs + 'NIRC2transmissionmap_161x161.fits')
        #
        #             self.angles = fits.getdata(path_obs + 'pyklip_parangs.fits')
        #
        #             self.psf = fits.getdata(path_obs + 'psf_noscale.fits')
        #             self.psf /= np.nansum(self.psf)
        # <<< code end 1
        # Then set the object up as:
        # >>> code start 2
        # data_input_info = data_input(path_obs = './reduction-bren/')
        # <<< code end 2

    ### (Forwarded) Models:
    if path_model is None:
        path_model = './mcfost_models/'
    if hash_address:
        if hash_string is None:
            print('Please provide the hash string if you set hash_address = True!')
            return -np.inf     
        path_model = path_model[:-1] + hash_string + '/'

    model_mcfost = fits.getdata(path_model + 'data_3.8/RT.fits.gz')*8e20

    if len(model_mcfost.shape) == 5:
        model_mcfost = model_mcfost[0, 0, 0]

    model_mcfost[(model_mcfost.shape[0] - 1)//2, (model_mcfost.shape[1] - 1)//2] = 0

    cube = dependencies.rotateCube(model_mcfost, angle=-angles, mask = mask_obs, maskedNaN=True, outputMask=False)
    cube_convoled = np.array([image_registration.fft_tools.convolve_nd.convolvend(cube[i], psf_keck) for i in range(cube.shape[0])])
    
    cube_convoled *= map_transmission #multiply the transmission map
    
    #negative injection
    obs_neg_injected = obs_raw - cube_convoled
    # PCA for negative injected observation
    components_neg_injected = fm_klip.pcaImageCube(obs_neg_injected, mask_obs, pcNum = 3)
    klipped_neg_injected = np.zeros_like(obs_neg_injected)
    for i in range(klipped_neg_injected.shape[0]):
        klipped_neg_injected[i] = fm_klip.klip(obs_neg_injected[i], components_neg_injected[:3], mask_obs, cube = False)
    
    reduced_derotated =  dependencies.rotateCube(klipped_neg_injected, angle=angles, mask = mask_obs, maskedNaN=True, outputMask=False)
    
    result_neg_inj = np.nanmedian(reduced_derotated, axis = 0)
    unc_neg_inj = np.nanstd(reduced_derotated, axis = 0)
        
    lnlike_value = chi2(result_neg_inj*mask_calc, unc_neg_inj*mask_calc, np.zeros_like(result_neg_inj), lnlike=True)
    
    if writemodel:
        fits.writeto(path_model + 'model_fm.fits', result_neg_inj, overwrite = True)
        fits.writeto(path_model + 'model_convolved.fits', image_registration.fft_tools.convolve_nd.convolvend(model_mcfost, psf_keck), overwrite = True)
        fits.writeto(path_model + 'model_convolved_times_transmission.fits', image_registration.fft_tools.convolve_nd.convolvend(model_mcfost, psf_keck)*map_transmission, overwrite = True)
        fits.writeto(path_model + 'model_fm_snr.fits', result_neg_inj/unc_neg_inj, overwrite = True)
        
        
    if hash_address and delete_model:    #delete the temporary MCFOST models only when the string is hashed
        shutil.rmtree(path_model)
    
    return  lnlike_value #Returns the loglikelihood