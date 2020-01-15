from astropy.io import fits
import numpy as np

from . import dependencies
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

def pcaImageCube(ref, mask = None, pcNum = None, cube=True, ref3D=True, outputEval = False):
    """Principal Component Analysis, 
    Input: 
        ref: Cube of references, 3D; 
            if ref3D==False, 2D (Flattened and Normalized, with maksked region excluded.) 
        mask: mask, 2D or 1D;
        pcNum: how many principal components are needed;
        cube: output as a cube? Otherwise a flattend 2D component array will be returned.
        ref3D: Ture by default.
        outputEval: whether to return the eigen values, False by default.
    Output:
        The principal components, either cube (3D) or flattend (2D)."""
    if mask is None:
        mask = np.ones(ref[0].shape)
    if pcNum is None:
        pcNum = ref.shape[0]
    if ref3D:
        mask_flat = mask.flatten()
        ref_flat = np.zeros((ref.shape[0], np.where(mask_flat == 1)[0].shape[0]))
        for i in range(ref_flat.shape[0]):
            ref_flat[i], std = flattenAndNormalize(ref[i], mask)
    else:
        ref_flat = ref
        if np.shape(mask.shape)[0] == 1: #1D mask, already flattened
            mask_flat = mask
        elif np.shape(mask.shape)[0] == 2: #2D mask, need flatten
            mask_flat = mask.flatten()
        
    covMatrix = np.dot(ref_flat, np.transpose(ref_flat))
    eVal, eVec = np.linalg.eig(covMatrix)
    index = (-eVal).argsort()[:pcNum]
    eVec = eVec[:,index]
    components_flatten = np.dot(np.transpose(eVec), ref_flat)
    
    pc_flat = np.zeros((pcNum, mask_flat.shape[0]))
    
    for i in range(pc_flat.shape[0]):
        pc_flat[i][np.where(mask_flat==1)] = components_flatten[i]/np.sqrt(np.dot(components_flatten[i], np.transpose(components_flatten[i])))
    if cube == False:
        return pc_flat
    
    pc_cube = np.zeros((pcNum, mask.shape[0], mask.shape[1]))
    width = mask.shape[0]
    for i in range(pc_flat.shape[0]):
        pc_cube[i] = np.array(np.split(pc_flat[i], width))
        
    if not outputEval:
        return pc_cube
    else:
        return pc_cube, eVal[index]
           
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
        
    mask[np.isnan(trg)] = 0
        
    width = mask.shape[0]         # Number of rows, width_y
    width_x = mask.shape[1]       # The above two lines are used to reconstruct a 1D image back to 2D.
    mask_flat = mask.flatten()
    
    if trg2D is True:
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

    if cube == False:
        result_flat = trg_flat - np.sum(coef * pcs_flat, axis = 0)
        temp_result = result_flat
        return temp_result.reshape(width, width_x) * std
    else:
        result_flat = np.dot(np.ones((klipK, 1)), 
                        trg_flat[np.newaxis]) - np.dot(np.tril(np.ones((klipK, klipK))),
                                                   coef * pcs_flat)
        result = np.zeros((klipK, width, width_x))
        for i in range(klipK):
            temp_result =  result_flat[i]
            result[i] = temp_result.reshape(width, width_x)
        return result*std


def klip_fm_main(path = './test/', path_obs = None, angles = None, psf = None, pipeline_input = 'ALICE', alice_size = None):
    disk_model = fits.getdata(path + 'data_1.12/RT.fits.gz')[0, 0, 0]
    disk_model[int((disk_model.shape[0]-1)/2)-2:int((disk_model.shape[0]-1)/2)+3, int((disk_model.shape[0]-1)/2)-2:int((disk_model.shape[0]-1)/2)+3] = 0
    # Exclude the star in the above line
    if psf is not None:
        if len(psf.shape) != 2:
            raise  valueError('The input PSF is not 2D, please pass a 2D one here!')
        psf /= np.nansum(psf)               #Normalize the PSF (planet PSF) in case the input is not equal to 1
        convolved0 = image_registration.fft_tools.convolve_nd.convolvend(disk_model, psf)
        disk_model = convolved0
    
    if path_obs is None:
        path_obs = './data_observation/'
    components = fits.getdata(path_obs + 'NICMOS/HD-191089_NICMOS_F110W_Lib-84_KL-19_KLmodes.fits')
    mask = fits.getdata(path_obs + 'NICMOS/HD-191089_NICMOS_F110W_Lib-84_KL-19_Mask.fits')
    if angles is None:
        angles = np.concatenate([[19.5699]*4, [49.5699]*4]) # The values are hard coded for HD 191089 NICMOS observations, pelase change it for other targets.

    disk_rotated = dependencies.rotateCube(disk_model, angle = angles, maskedNaN=True, outputMask=False)
    masks_rotated = np.ones(disk_rotated.shape)
    masks_rotated[np.where(np.isnan(disk_rotated))] = 0
    if pipeline_input == 'ALICE':
        mask = mask[1:, 1:]
    masks_rotated *= mask
    
    if pipeline_input == 'ALICE':
        if alice_size is None:
            alice_size = 140
        # The ALICE pipeline has image of even size, and the center of the star is at the center of the image with the 1st row and 1st column cropped
        # Solution as follows (create 140*140 or 80*80 images, with the 1st row and 1st column set to be all 0's)
        disk_rotated_140 = np.zeros((disk_rotated.shape[0], alice_size, alice_size)) # make size = 140x140 images for KLIP
        disk_rotated_140[:, 1:, 1:] = disk_rotated
        
        mask_rotated_140 = np.zeros(disk_rotated_140.shape)
        mask_rotated_140[:, 1:, 1:] = masks_rotated
        
        disk_rotated = disk_rotated_140
        del disk_rotated_140
        masks_rotated = mask_rotated_140
        del mask_rotated_140
        
    results_rotated = np.zeros(disk_rotated.shape)    
    
    for i, data_slice in enumerate(disk_rotated):
        results_rotated[i] = klip(data_slice, pcs = components[i], mask = masks_rotated[i], cube=False)

    mask_rotated_nan = np.ones(masks_rotated.shape)    
    mask_rotated_nan[np.where(masks_rotated==0)] = np.nan
    
    if pipeline_input == 'ALICE':
        results_rotated_old = results_rotated[:, 1:, 1:]
        results_rotated = results_rotated_old
        del results_rotated_old
        mask_rotated_nan_old = mask_rotated_nan[:, 1:, 1:]
        mask_rotated_nan = mask_rotated_nan_old
        del mask_rotated_nan_old

    results = dependencies.rotateCube(results_rotated*mask_rotated_nan, mask = None, angle = -angles, maskedNaN=True, outputMask=False)

    mask_detorated_nan = np.ones(results.shape)
    mask_detorated_nan[np.where(np.isnan(results))] = np.nan

    result_klip = np.nansum(results, axis = 0)/np.nansum(mask_detorated_nan, axis = 0)
    
    if pipeline_input == 'ALICE':
        if alice_size is None:
            alice_size = 140
        result_klip_alice = np.zeros((alice_size, alice_size))
        result_klip_alice[1:, 1:] = result_klip
        result_klip = result_klip_alice
        del result_klip_alice   
    
    return result_klip


# test = klip_fm_main()


