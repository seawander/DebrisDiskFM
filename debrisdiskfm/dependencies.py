import numpy as np
from scipy.ndimage.interpolation import rotate
from scipy.interpolate import interp2d

def annulusMask(width, r_in, r_out = None, width_x = None, cen_y = None, cen_x = None):
    """Creat a width*width all-0 mask; for r_in <= r <= r_out, 1.
    If r_out = 0, it means you are not constraining r_out, i.e., if r >= r_int, all ones.
    Default is a square mask centering at the center.
    Input:
        width: number of rows, a.k.a., width_y;
        width_x: number of columns. Default is None, i.e., width_x = width;
        r_in: where 1 starts;
        r_out: where 1 ends. Default is None, i.e., r_out = +infinity;
        cen_y: center of the annulus in y-direction. Default is None, i.e., cen_y = (width-1)/2.0;
        cen_x: center of the annulus in x-direction. Default is None, i.e., cen_x = (width_x-1)/2.0;
    Output:
        result.
    """
    if width_x is None:
        width_x = width
    if cen_y is None:
        cen_y = (width-1)/2.0
    if cen_x is None:
        cen_x = (width_x-1)/2.0
    
    result = np.zeros((width, width_x))
    
    r_in = np.max([0, r_in])
    
    for i in range(width):
        for j in range(width_x):
            if r_out is None:
                if (i - cen_y)**2 + (j - cen_x)**2 >= r_in**2:
                    result[i, j] = 1
            else:
                if (i - cen_y)**2 + (j - cen_x)**2 >= r_in**2 and (i - cen_y)**2 + (j - cen_x)**2 <= r_out**2:
                    result[i, j] = 1
    return result


def addplanet(image, planet, orientat = None, starflux = 1, radius = None, angle = None, contrast=1, exptime=1, planetOnly = False, x_planet = None, y_planet = None, surroundingReplace = None):
    """
    Add a fake planet or image to ONE image of the star, "angle" is from north to east.
    Input:
        image: image of star, 
        orientat: ORIENTAT keyword, 
        starflux: total flux of the star,
        planet: image of fake planet, 
        radius: seperation between the star and the planet, 
        angle: angle of the planet on y-axis (deg E of N), 
        contrast: contrast,
        exptime: exposure time
        planetOnly: only output the planet?
        x_planet, y_planet: you can add the Cartesian coordinates directly -- this will overwrite the radius and angle info if you have already.
        surroundingReplace: number. If given, the surroundings of the planet (i.e., the star except the planet) will be replaced with this value."""
    #star
    centerx = (image.shape[1]-1)/2.0
    centery = (image.shape[0]-1)/2.0
    #planet
    nan_flag = 0 #flag of whether are there nan's in the planet array
    if np.sum(np.isnan(planet)) != 0:
        nan_flag = 1
        
        planet_nans = np.zeros(planet.shape) #array to store the nan valuse in the planet
        planet_nans[np.where(np.isnan(planet))] = 1 #Find the nan's in the planet, then use this mask to exclude the nan region in the last step.
        palnet_nans_added = addplanet(np.zeros(image.shape), planet = planet_nans)
        palnet_nans_added[palnet_nans_added <= 0.9] = 0
        palnet_nans_added[palnet_nans_added != 0] = 1
        
        planet[np.where(np.isnan(planet))] = 0 #now the nan's in the planet are replaced with 0's.
    
    if x_planet is None:
        if radius is not None:
            x_planet = round(radius*np.cos(np.deg2rad(-orientat+angle+90)) + centerx, 2)
        else:
            x_planet = (image.shape[1]-1)/2.0 #default is put in the center.
    if y_planet is None:
        if radius is not None:
            y_planet = round(radius*np.sin(np.deg2rad(-orientat+angle+90)) + centery, 2)
        else:
            y_planet = (image.shape[0]-1)/2.0
    x_range = np.arange(-(planet.shape[1]-1)/2.0+x_planet, 
                        -(planet.shape[1]-1)/2.0+x_planet + planet.shape[1] - 0.001,
                        1)
    y_range = np.arange(-(planet.shape[0]-1)/2.0++y_planet, 
                        -(planet.shape[0]-1)/2.0+y_planet + planet.shape[0] - 0.001,
                        1)
    planetfunc = interp2d(x_range, y_range, planet, kind='cubic') #Interpolation Part, (x_planet,y_planet) is maximum
    planetonly = np.zeros(image.shape) #This image contains only the planet
    x_range = np.arange( max(0, round(min(x_range)), 0), 
                        min(image.shape[1]-1, round(max(x_range), 0)) + 1
                        , 1)
    y_range = np.arange( max(0, round(min(y_range)), 0), 
                        min(image.shape[0]-1, round(max(y_range), 0)) + 1
                        , 1)
    if surroundingReplace is not None:
        planetonly[:,:] = surroundingReplace
    planetonly[int(min(y_range)):int(max(y_range))+1, int(min(x_range)):int(max(x_range))+1] = planetfunc(x_range, y_range)*starflux*contrast*exptime
    
    if nan_flag == 1:
        planetonly[np.where(palnet_nans_added == 1)] = np.nan
    
    if planetOnly or surroundingReplace is not None:
        return planetonly
    return planetonly+image
    
def cutImage(image, halfSize, x_cen = None, y_cen = None, halfSizeX = None, halfSizeY = None, mask = None, relative_shift = False, dx = None, dy = None):
    """Cut the given image"""
    
    image = np.copy(image)

    if x_cen is None:
        x_cen = (image.shape[1] - 1)/2.0
    if y_cen is None:
        y_cen = (image.shape[0] - 1)/2.0
    if halfSizeX is None:
        halfSizeX = halfSize
    if halfSizeY is None:
        halfSizeY = halfSize
        
    if relative_shift == True:
        if dx is not None:
            x_cen += dx
        if dy is not None:
            y_cen += dy            
    
    if mask is None:
        mask = np.ones(image.shape)
    mask[np.where(mask < 0.9)] = 0
    mask[np.where(mask != 0)] = 1    
    
    maskNaN = np.ones(image.shape) #In case there are NaN's in the image    
    maskNaN[np.isnan(image)] = 0
    
    mask *= maskNaN
    
    image[np.isnan(image)] = 0
    
    
    image_interp = interp2d(np.arange(image.shape[1]), np.arange(image.shape[0]), image)
    mask_interp = interp2d(np.arange(image.shape[1]), np.arange(image.shape[0]), mask)
    
    

    newImage = np.zeros((int(2*halfSizeY+1), int(2*halfSizeX+1)))
    x_range = np.round(np.arange(x_cen - halfSizeX, x_cen + halfSizeX + 0.1, 1), decimals = 2)
    y_range = np.round(np.arange(y_cen - halfSizeY, y_cen + halfSizeY + 0.1, 1), decimals = 2)
    newImage = image_interp(x_range, y_range)
    maskInterped = mask_interp(x_range, y_range) #Interpolate the image and mask
    
    maskInterped[np.where(maskInterped < 0.9)] = 0
    maskInterped[np.where(maskInterped == 0)] = np.nan
    
    return newImage*maskInterped
    
def bin_data(data, bin_size = 3, data_type = 'data', bin_method = 'average'):
    """Bin the data with a bin_size*bin_size box. If the data_type is `data`, then a simple addition is performed,
    if it is an `uncertainty` map, then the square root of the squared sums are returned.
    
    The bin_method can be assigned with 'average' or 'sum':
        if 'sum', the raw binned data will be returned;
        if 'average', the raw binned data will be divided by bin_size^2 then returned.
    """
    if data_type == 'uncertainty':
        data = data**2
    total_size = np.ceil(data.shape[0] / bin_size) * bin_size
    half_size = (total_size - 1)/2.0
    data_extended = cutImage(data, halfSize=half_size)
    data_extended[np.isnan(data_extended)] = 0
    bin_matrix = np.zeros((int(total_size), int(total_size//bin_size)))
    for i in range(int(total_size//bin_size)):
        bin_matrix[bin_size*i:bin_size*(i+1), i] = 1
    data_binned = np.dot(bin_matrix.T, np.dot(data_extended, bin_matrix))
    
    if bin_method == 'sum':
        pass
    elif bin_method == 'average':
        if data_type == 'data':
            data_binned /= bin_size**2
        elif data_type == 'uncertainty':
            data_binned /= bin_size**4 #the extra power of 2 is because the raw data is squared for the uncertainty map
        
    if data_type == 'uncertainty':
        return np.sqrt(data_binned)
    return data_binned
    
def rotateImage(cube, mask = None, angle = None, reshape = False, new_width = None, new_height = None, thresh = 0.9, maskedNaN = False, outputMask = True, instrument = None):
    """Rotate an image with 1 mask and 1 angle."""
    cube0 = np.copy(cube)
    cube0[np.where(np.isnan(cube0))] = 0
    
    #1. Prepare the Cube and Mask
    if reshape:
        if new_width is None and new_height is None:
            new_width = int(np.sqrt(np.sum(np.asarray(cube.shape)**2)))
            new_height = new_width
        cube = np.zeros((new_height, new_width))
        cube += addplanet(cube, planet = cube0, surroundingReplace = np.nan, planetOnly = True)
        #Replace the surroundings of extended cube with NaN's -- this is used to generate a mask if no mask is provided.

        if mask is not None:
            mask0 = np.copy(mask)
            mask = np.zeros(cube.shape)
            mask += addplanet(cube, planet = mask0, surroundingReplace = 0, planetOnly = True)
            mask[np.where(mask < thresh)] = 0
            mask[np.where(mask != 0)] = 1
        else:
            mask = np.ones(cube.shape)
            mask[np.where(np.isnan(cube))] = 0
        
        cube[np.where(np.isnan(cube))] = 0
    else:
        if mask is None:
            mask = np.ones(cube.shape)
            mask[np.where(np.isnan(cube))] = 0
            cube = cube0
        else:
            mask2 = np.ones(mask.shape)
            mask2[np.isnan(mask)] = 0
            mask2[np.where(mask == 0)] = 0
            mask = np.copy(mask2)
            cube = cube0
            
              
    #2. Rotate
    if angle is None:
        angle = 0
    if instrument == "GPI":
        angle -= 66.5 #IFS rotation
    result = rotate(cube, angle, reshape = False)
    rotatedMask = rotate(mask, angle, reshape = False)
    
    rotatedMask[np.where(rotatedMask < thresh)] = 0
    rotatedMask[np.where(rotatedMask != 0)] = 1
    
    result *= rotatedMask
        
    if maskedNaN:
        result[np.where(rotatedMask == 0)] = np.nan
    
    if instrument == "GPI":
        result = np.fliplr(result)
        rotatedMask = np.fliplr(rotatedMask)
    
    if outputMask:
        return result, rotatedMask
    else:
        return result

def rotateCube(cube, mask = None, angle = None, reshape = False, new_width = None, new_height = None, thresh = 0.9, maskedNaN = False, outputMask = True, instrument = None):
    """Rotation function for a cube.
    =======
    Input:
            cube (2- or 3-D array): either an image or an image cube
            mask (2- or 3-D array): either a mask or a mask cube
            angle (float number or 1-D array): either an angle or an angle array
            reshape (boolean): change the size? If yes,
                new_width (integer): new width of the output (can be larger or smaller than before)
                new_height (integer): new height of the output (can be larger or smaller then before)
            thresh (float, 0 to 1): if the mask is smaller than 0.9 then it will be regarded as 0
            maskedNaN (boolean): put the masked pixels as NaN value?
            outputMask (boolean): output the rotated mask(s)?
    Output:
            first one: results
            second one: rotatedMasks (only when outputMask == True)
    ========
    Example:
        results, masks = rotateCube(data, mask= mask, angle=-angles, maskedNaN= True, reshape=True)
    """
    # print("Rotating a cube...")
    cube0 = np.copy(cube)
    # cube0[np.where(np.isnan(cube0))] = 0
    mask0 = np.copy(mask)
    if mask is None:
        mask0 = None
    
    angle0 = np.copy(angle)
    if (angle is None) or (np.asarray(angle).shape == ()):
        angle = [angle]
    angle = np.asarray(angle)  
    
    if len(cube0.shape) == 2:
        # print("\tJust one input image, look easy.")
        #single image
        if len(angle) != 1:
            # print("\t\tBut with multiple angles, start working...")
            #multiple angles
            if (mask is None) or (len(mask.shape) == 2):
                # print("\t\t\t Just one input mask (or none), duplicating to make a mask cube.")
                #if single mask, then make multiple masks
                mask = np.asarray([mask0] * len(angle))
                
            #calculation
            if outputMask:
                #need rotated masks
                # print("\t\t\t\t Rotating...")
                for i in range(len(angle)):
                    results_temp, rotatedMask_temp = rotateImage(cube, mask = mask[i], angle = angle[i], reshape = reshape,
                                                             new_width = new_width, new_height = new_height, thresh = thresh,
                                                             maskedNaN = maskedNaN, outputMask = outputMask, instrument = instrument)
                    if i == 0:
                        results = np.zeros((mask.shape[0], ) + results_temp.shape)
                        rotatedMasks = np.zeros(results.shape)
                    results[i] = results_temp
                    rotatedMasks[i] = rotatedMask_temp
                # print("\t\t\t\t\t Done. Returning.")
                return results, rotatedMasks
            else:
                #don't need rotated masks
                # print("\t\t\t\t Rotating...")
                for i in range(len(angle)):
                    # print(i, cube.shape, mask, angle[i])
                    results_temp = rotateImage(cube, mask = mask[i], angle = angle[i], reshape = reshape,
                                             new_width = new_width, new_height = new_height, thresh = thresh,
                                             maskedNaN = maskedNaN, outputMask = outputMask, instrument = instrument)
                    if i == 0:
                        results = np.zeros((mask.shape[0], ) + results_temp.shape)
                    results[i] = results_temp
                # print("\t\t\t\t\t Done. Returning.")
                return results
        else:
            # print("\t\tAnd just one angle, looks easier..")
            if (mask is None) or (len(mask.shape) == 2):
                # print("\t\t\t Yeah and there is only one mask or no mask. Hooray!")
                if outputMask:
                    pass # print("\t\t\t\t Returning results and rotated masks.")
                else:
                    pass # print("\t\t\t\t Returning results.")
                return rotateImage(cube, mask = mask, angle = angle[0], reshape = reshape,
                                   new_width = new_width, new_height = new_height, thresh = thresh,
                                   maskedNaN = maskedNaN, outputMask = outputMask, instrument = instrument)
            else:
                # print("\t\t\t Hmmmmm, several masks, working on that...")
                if outputMask:
                    for i in range(mask.shape[0]):
                        results_temp, rotatedMask_temp = rotateImage(cube, mask = mask[i], angle = angle[0], reshape = reshape,
                                                                     new_width = new_width, new_height = new_height, thresh = thresh,
                                                                     maskedNaN = maskedNaN, outputMask = outputMask, instrument = instrument)
                        if i == 0:
                            results = np.zeros((mask.shape[0], ) + results_temp.shape)
                            rotatedMasks = np.zeros(results.shape)
                        results[i] = results_temp
                        rotatedMasks[i] = rotatedMask_temp
                    # print("\t\t\t\t Returning results and rotated masks.")
                    return results, rotatedMasks
                else:
                    for i in range(mask.shape[0]):
                        results_temp = rotateImage(cube, mask = mask[i], angle = angle[0], reshape = reshape,
                                                                     new_width = new_width, new_height = new_height, thresh = thresh,
                                                                     maskedNaN = maskedNaN, outputMask = outputMask, instrument = instrument)
                        if i == 0:
                            results = np.zeros((mask.shape[0], ) + results_temp.shape)
                        results[i] = results_temp
                    # print("\t\t\t\t Returning results.")
                    return results
    elif len(cube0.shape) == 3:
        # print("\tOh the input is really an image cube, working...")
        if (mask is None) or (len(mask.shape) == 2):
            # print("\t\t Just one input mask (or none), duplicating to make a mask cube.")
            #if single mask, then make multiple masks
            mask = np.asarray([mask0] * cube0.shape[0])
        if len(angle) == 1:
            # print("\t\t Just one input angle (or none), duplicating to make a mask cube.")
            angle = np.asarray([angle[0]] * cube0.shape[0])
        # print("\t\t\t Rotating...")
        
        if outputMask:
            for i in range(cube0.shape[0]):
                results_temp, rotatedMask_temp = rotateImage(cube0[i], mask = mask[i], angle = angle[i], reshape = reshape,
                                                             new_width = new_width, new_height = new_height, thresh = thresh,
                                                             maskedNaN = maskedNaN, outputMask = outputMask, instrument = instrument)
                if i == 0:
                    results = np.zeros((mask.shape[0], ) + results_temp.shape)
                    rotatedMasks = np.zeros(results.shape)
                results[i] = results_temp
                rotatedMasks[i] = rotatedMask_temp
            # print("\t\t\t\t Returning results and rotated masks.")
            return results, rotatedMasks
        else:
            for i in range(cube0.shape[0]):
                results_temp = rotateImage(cube0[i], mask = mask[i], angle = angle[i], reshape = reshape,
                                                             new_width = new_width, new_height = new_height, thresh = thresh,
                                                             maskedNaN = maskedNaN, outputMask = outputMask, instrument = instrument)
                if i == 0:
                    results = np.zeros((mask.shape[0], ) + results_temp.shape)
                results[i] = results_temp
            # print("\t\t\t\t Returning results.")
            return results     
