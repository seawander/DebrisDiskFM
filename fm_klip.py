from astropy.io import fits
import numpy as np

import dependences
import image_registration

# returns the KLIPped model

def flattenAndNormalize(image, mask = None, onlyMasked = True):
    """Flattend and Normalize the image (for KLIP).
    Input:
        image: image;
        mask: 0-1 mask;
        onlyMasked: True, then only pixels with maskvalue 1 will be outputed.
    Output: result, std
        if onlyMasked == True:
            only the mask==1 values, and the standard deviation
        else:
            all the values, and the standard deviation
    
    """
    
    if np.size(image.shape) == 2:
        if mask is None:
            mask = np.ones(image.shape, dtype = 'int')
        
        mask_flat = mask.flatten()
        
        result = np.zeros(np.where(mask_flat == 1)[0].shape[0])

        result = image.flatten()[np.where(mask_flat == 1)]*1.0 # multiply by 1.0 to convert possible integers to floats
        result -= np.nanmean(result)
        std = np.nanstd(result)
        result /= std
        if onlyMasked == True:
            return result, std
        else:
            mask_flat[np.where(mask_flat==1)] = result
            return mask_flat, std
    
    elif np.size(image.shape) == 3:
        if mask is None:
            mask = np.ones(image[0].shape, dtype = 'int')
        
        mask_flat = mask.flatten()
        
        images = np.copy(image)
        result = np.zeros((images.shape[0], np.where(mask_flat == 1)[0].shape[0]))
        std = np.zeros(images.shape[0])
        for i in range(images.shape[0]):
            image_slice = images[i]
            result[i], std[i] = flattenAndNormalize(image_slice, mask = mask, onlyMasked = onlyMasked)
        return result, std
    
def klip(trg, pcs, mask = None, klipK = None, cube = True, trg2D=True):
    """KLIP Algorithm. 
    Input:
        trg: target image, 2D; 
            if trg2D==False, then it is 1D 
            (Flattend&Normalized. std=1 and the result should be multiplied by original std!)
        pcs: principal components from PCA, 3D cube or 2D cube;
            Requirement: For the 2D cube, components are on rows.
        klipK: the truncation value.
        trg2D: is the target a 2D image?
    Output:
        Image, if cube == False;
        Cube Image of all the slices, if cube == True.
    """
    if mask is None:
        mask = np.ones(trg.shape)
    if klipK is None:
        klipK = pcs.shape[0]
        
    width = mask.shape[0]         # Number of rows, width_y
    width_x = mask.shape[1]       # The above two lines are used to reconstruct a 1D image back to 2D.
    mask_flat = mask.flatten()
    
    if trg2D==True:
        trg_flat, std = flattenAndNormalize(trg, mask, onlyMasked = False)
    else:
        trg_flat = np.zeros(mask_flat.shape)
        trg_flat[np.where(mask_flat == 1)] = trg
        std = 1
    
    
    if np.array(pcs.shape).shape[0] == 3:
        #3D cube, convert to 2D
        pcs_flat = np.zeros((klipK, mask_flat.shape[0]))#Masked region included
        for i in range(klipK):
            pcs_flat[i] = pcs[i].flatten()                 #[np.where(mask_flat == 1)]
    else:
        #2D cube
        pcs_flat = pcs[:klipK]

    coef = np.transpose(np.dot(pcs_flat, np.transpose(trg_flat))[np.newaxis])
    result_flat = np.dot(np.ones((klipK, 1)), 
                    trg_flat[np.newaxis]) - np.dot(np.tril(np.ones((klipK, klipK))),
                                               coef * pcs_flat)
    if cube == False:
        temp_result = result_flat[klipK-1]
        return temp_result.reshape(width, width_x) * std
    else:
        result = np.zeros((klipK, width, width_x))
        for i in range(klipK):
            temp_result =  result_flat[i]
            result[i] = temp_result.reshape(width, width_x)
        return result*std


def klip_fm_main(path = './test/', angles = None, psf = None):
    disk_model = fits.getdata(path + 'data_1.12347/RT.fits.gz')[0, 0, 0]
    disk_model[int((disk_model.shape[0]-1)/2)-2:int((disk_model.shape[0]-1)/2)+3, int((disk_model.shape[0]-1)/2)-2:int((disk_model.shape[0]-1)/2)+3] = 0
    # Exclude the star in the above line
    if psf is not None:
        if len(psf.shape) != 2:
            raise  valueError('The input PSF is not 2D, please pass a 2D one here!')
        psf /= np.nansum(psf)               #Normalize the PSF (planet PSF) in case the input is not equal to 1
        convolved0 = image_registration.fft_tools.convolve_nd.convolvend(disk_model, psf)
        disk_model = convolved0
    
    
    components = fits.getdata('./data_observation/NICMOS/HD-191089_NICMOS_F110W_Lib-84_KL-19_KLmodes.fits')
    mask = fits.getdata('./data_observation/NICMOS/HD-191089_NICMOS_F110W_Lib-84_KL-19_Mask.fits')
    if angles is None:
        angles = np.concatenate([[19.5699]*4, [49.5699]*4]) # The values are hard coded for HD 191089 NICMOS observations, pelase change it for other targets.

    disk_rotated = dependences.rotateCube(disk_model, angle = angles, maskedNaN=True, outputMask=False)
    masks_rotated = np.ones(disk_rotated.shape)
    masks_rotated[np.where(np.isnan(disk_rotated))] = 0
    masks_rotated *= mask

    results_rotated = np.zeros(disk_rotated.shape)

    for i, data_slice in enumerate(disk_rotated):
        results_rotated[i] = klip(data_slice, pcs = components[i], mask = masks_rotated[i], cube=False)

    mask_rotated_nan = np.ones(masks_rotated.shape)    
    mask_rotated_nan[np.where(masks_rotated==0)] = np.nan

    results = dependences.rotateCube(results_rotated*mask_rotated_nan, mask = None, angle = -angles, maskedNaN=True, outputMask=False)

    mask_detorated_nan = np.ones(results.shape)
    mask_detorated_nan[np.where(np.isnan(results))] = np.nan

    result_klip = np.nansum(results, axis = 0)/np.nansum(mask_detorated_nan, axis = 0)
    
    return result_klip


# test = klip_fm_main()


