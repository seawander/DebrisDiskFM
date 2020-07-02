########################################################
########################################################
#### MAIN AUTHOR Max Millar Blanchaer
#### from a James Graham's original code
#### Initial model by Max Millar Blanchaer from a James Graham's original code, 
### If you are using this model, please cite 
### Millar-Blanchaer, M. A., Graham, J. R., Pueyo, L., et al. 2015, ApJ, 811, 18
#### exctracted from anadisk_e.py sent by Max in 2017
#### This file is downloaded from https://github.com/seawander/debrisdisk_mcmc_fit_and_plot/blob/master/disk_models.py on 2020 July 2.
########################################################
########################################################

import math as mt
import numpy as np

from scipy.integrate import quad

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def hg_1g(scatt_angles, g1, Norm):
    """
    take a set of scatt angles and a set of HG parameter and return a
    1g HG SPF

    Args:
        scatt_angles: a list of angles in degrees of dimensions N_angles.
                        The list must contains 90 degree values
        g1: first HG parameter
        Norm: Normalisation (value at 90 degree of the function)

    Returns:
        the 1g SPF, list of dimensions N_angles.

    """

    scattered_angles_rad = np.radians(scatt_angles)
    cos_phi = np.cos(scattered_angles_rad)

    g1_2 = g1 * g1  #First HG g squared
    #Constant for HG function
    k = 1. / (4 * np.pi)

    #Henyey Greenstein function
    hg1 = k  * (1. - g1_2) / (1. + g1_2 - (2 * g1 * cos_phi))**1.5
    hg = hg1
    hg_norm = hg / hg[np.where(scatt_angles == 90)] * Norm
    return hg_norm



def hg_2g(scatt_angles, g1, g2, alpha, Norm):
    """
    take a set of scatt angles and a set of HG parameter and return a
    2g HG SPF

    Args:
        scatt_angles: a list of angles in degrees of dimensions N_angles.
                        The list must contains 90 degree values
        g1: first HG parameter
        g2: second HG parameter
        alpha: relative weight
                 hg = alpha * hg1 * hg2 + (1 - alpha) * hg2
        Norm: Normalisation (value at 90 degree of the function)

    Returns:
        the 2g SPF, list of dimensions N_angles.

    """

    scattered_angles_rad = np.radians(scatt_angles)
    cos_phi = np.cos(scattered_angles_rad)

    g1_2 = g1 * g1  #First HG g squared
    g2_2 = g2 * g2  #Second HG g squared
    #Constant for HG function
    k = 1. / (4 * np.pi)

    #Henyey Greenstein function
    hg1 = k * alpha * (1. - g1_2) / (1. + g1_2 - (2 * g1 * cos_phi))**1.5
    hg2 = k * (1 - alpha) * (1. - g2_2) / (1. + g2_2 - (2 * g2 * cos_phi))**1.5
    hg = hg1 + hg2
    hg_norm = hg / hg[np.where(scatt_angles == 90)] * Norm
    return hg_norm


def hg_3g(scatt_angles, g1, g2, g3, alpha1, alpha2, Norm):
    """
    take a set of scatt angles and a set of HG parameter and return a
    3g HG SPF

    Args:
        scatt_angles: a list of angles in degrees of dimensions N_angles.
                        The list must contains 90 degree values
        g1: first HG parameter
        g2: second HG parameter
        g3: third HG parameter
        alpha1: first relative weight
        alpha2: second relative weight
                hg = alpha1 * hg1 + alpha2 * hg2 + (1 - alpha1 - alpha2) * hg3
        Norm: Normalisation (value at 90 degree of the function)

    Returns:
        the 3g SPF, list of dimensions N_angles.
    """

    scattered_angles_rad = np.radians(scatt_angles)
    cos_phi = np.cos(scattered_angles_rad)

    g1_2 = g1 * g1  #First HG g squared
    g2_2 = g2 * g2  #Second HG g squared
    g3_2 = g3 * g3  #Third HG g squared

    #Constant for HG function
    k = 1. / (4 * np.pi)

    #Henyey Greenstein function
    hg1 = k * (1. - g1_2) / (1. + g1_2 - (2 * g1 * cos_phi))**1.5
    hg2 = k * (1. - g2_2) / (1. + g2_2 - (2 * g2 * cos_phi))**1.5
    hg3 = k * (1. - g3_2) / (1. + g3_2 - (2 * g3 * cos_phi))**1.5
    hg = alpha1 * hg1 + alpha2 * hg2 + (1 - alpha1 - alpha2) * hg3

    hg_norm = hg / hg[np.where(scatt_angles == 90)] * Norm

    return hg_norm


def log_hg_2g(scatt_angles, g1, g2, alpha, Norm):
    """
    take a set of scatt angles and a set of HG parameter and return
    the log of a 2g HG SPF (usefull to fit from a set of points)

    Args:
        scatt_angles: a list of angles in degrees of dimensions N_angles.
                        The list must contains 90 degree values
        g1: first HG parameter
        g2: second HG parameter
        alpha: relative weight
                 hg = alpha * hg1 * hg2 + (1 - alpha) * hg2
        Norm: Normalisation (value at 90 degree of the function)

    Returns:
        the log of the 2g SPF, list of dimensions N_angles.

    """

    return np.log(hg_2g(scatt_angles, g1, g2, alpha, Norm))


def log_hg_3g(scatt_angles, g1, g2, g3, alpha1, alpha2, Norm):
    """
    take a set of scatt angles and a set of HG parameter and return the
    log of the 3g HG SPF (usefull to fit from a set of points)

    Args:
        scatt_angles: a list of angles in degrees of dimensions N_angles.
                        The list must contains 90 degree values
        g1: first HG parameter
        g2: second HG parameter
        g3: third HG parameter
        alpha1: first relative weight
        alpha2: second relative weight
                hg = alpha1 * hg1 + alpha2 * hg2 + (1 - alpha1 - alpha2) * hg3
        Norm: Normalisation (value at 90 degree of the function)

    Returns:
        the log of the 3g SPF, list of dimensions N_angles.
    """
    return np.log(hg_3g(scatt_angles, g1, g2, g3, alpha1, alpha2, Norm))

def integrand_dxdy_1g(xp, yp_dy2, yp2, zp, zp2, zpsi_dx, zpci, R1, R2, beta,
                      a_r, g1, g1_2, ci, si, maxe, dx, dy,
                      k):
    # author : Max Millar Blanchaer
    # compute the scattering integrand
    # see analytic-disk.nb

    xx = (xp * ci + zpsi_dx)

    d1 = mt.sqrt((yp_dy2 + xx * xx))

    if (d1 < R1 or d1 > R2):
        return 0.0

    d2 = xp * xp + yp2 + zp2

    #The line of sight scattering angle
    cos_phi = xp / mt.sqrt(d2)
    # phi=np.arccos(cos_phi)

    #Henyey Greenstein function
    hg = k * (1. - g1_2) / (1. + g1_2 - (2 * g1 * cos_phi))**1.5


    #Radial power low r propto -beta
    int1 = hg * (R1 / d1)**beta

    #The scale height function
    zz = (zpci - xp * si)
    hh = (a_r * d1)
    expo = zz * zz / (hh * hh)

    # if expo > 2*maxe:   # cut off exponential after 28 e-foldings (~ 1E-06)
    #     return 0.0

    int2 = np.exp(0.5 * expo)
    int3 = int2 * d2

    return int1 / int3


def gen_disk_dxdy_1g(dim,
                     param_disk,
                     mask=None,
                     sampling=1,
                     distance=72.8,
                     pixscale=0.01414):
    """ author : Max Millar Blanchaer
        modified by Johan Mazoyer
        create a 1g SPF disk model. The disk is normalized at Norm at 90degree
        (before star offset). also normalized by aspect_ratio. These
        normalization avoid weird correlation in the parameters


    Args:
        dim: dimension of the image in pixel assuming square image
        param_disk: a dict with keywords: 
                R1: inner radius of the disk
                R2: outer radius of the disk
                beta: radial power law of the disk between R1 and R2
                aspect_ratio=0.1 vertical width of the disk
                g1: %, 1st HG param
                inc: degree, inclination
                pa: degree, principal angle
                dx: au, + -> NW offset disk plane Minor Axis
                dy: au, + -> SW offset disk plane Major Axis
                offset: vertical residue image
        mask: a np.where result that give where the model should be
              measured (important to save a lot of time)
        sampling: increase this parameter to bin the model
                  and save time
        distance: distance of the star
        pixscale: pixel scale of the instrument

    Returns:
        a 2d model
    """

    R1 = param_disk['r1']
    R2 = param_disk['r2']
    beta = param_disk['beta']

    inc = param_disk['inc']
    pa = param_disk['PA']
    dx = param_disk['dx']
    dy = param_disk['dy']
    Norm = param_disk['Norm']

    g1 = param_disk['g1']

    aspect_ratio = param_disk['a_r']
    offset = param_disk['offset']

    max_fov = dim / 2. * pixscale  #maximum radial distance in AU from the center to the edge
    npts = int(np.floor(dim / sampling))
    xsize = max_fov * distance  #maximum radial distance in AU from the center to the edge

    #The coordinate system here [x,y,z] is defined :
    # +ve x is the line of sight
    # +ve y is going right from the center
    # +ve z is going up from the center

    # y = np.linspace(0,xsize,num=npts/2)
    y = np.linspace(-xsize, xsize, num=npts)
    z = np.linspace(-xsize, xsize, num=npts)

    #Only need to compute half the image
    # image =np.zeros((npts,npts/2+1))
    image = np.zeros((npts, npts))

    #Some things we can precompute ahead of time
    maxe = mt.log(np.finfo('f').max)  #The log of the machine precision

    #Inclination Calculations
    incl = np.radians(90 - inc)
    ci = mt.cos(incl)  #Cosine of inclination
    si = mt.sin(incl)  #Sine of inclination

    #Position angle calculations
    pa_rad = np.radians(90 - pa)  #The position angle in radians
    cos_pa = mt.cos(pa_rad)  #Calculate these ahead of time
    sin_pa = mt.sin(pa_rad)

    #HG g value squared
    g1_2 = g1 * g1  # HG g squared
    
    #Constant for HG function
    k = 1. / (4 * np.pi)

    #The aspect ratio
    a_r = aspect_ratio

    #Henyey Greenstein function at 90
    hg_90 = k * (1. - g1_2) / (1. + g1_2)**1.5


    #If there's no mask then calculate for the full image
    if len(np.shape(mask)) < 2:

        for i, yp in enumerate(y):
            for j, zp in enumerate(z):

                #This rotates the coordinates in the image frame
                yy = yp * cos_pa - zp * sin_pa  #Rotate the y coordinate by the PA
                zz = yp * sin_pa + zp * cos_pa  #Rotate the z coordinate by the PA

                #The distance from the center (in each coordinate) squared
                y2 = yy * yy
                z2 = zz * zz

                #This rotates the coordinates in and out of the sky
                zpci = zz * ci  #Rotate the z coordinate by the inclination.
                zpsi = zz * si
                #Subtract the offset
                zpsi_dx = zpsi - dx

                #The distance from the offset squared
                yy_dy = yy - dy
                yy_dy2 = yy_dy * yy_dy

                image[j, i] = quad(integrand_dxdy_1g,
                                   -R2,
                                   R2,
                                   epsrel=0.5e-3,
                                   limit=75,
                                   args=(yy_dy2, y2, zp, z2, zpsi_dx, zpci, R1,
                                         R2, beta, a_r, g1, g1_2, ci, si, maxe, dx, dy, k))[0]

    #If there is a mask then don't calculate disk there
    else:
        hmask = mask
        # hmask = mask[:,140:] #Use only half the mask

        for i, yp in enumerate(y):
            for j, zp in enumerate(z):

                # if hmask[j,npts/2+i]: #This assumes
                # that the input mask has is the same size as
                # the desired image (i.e. ~ size / sampling)
                if hmask[j, i]:

                    image[j, i] = 0.  #np.nan

                else:

                    #This rotates the coordinates in the image frame
                    yy = yp * cos_pa - zp * sin_pa  #Rotate the y coordinate by the PA
                    zz = yp * sin_pa + zp * cos_pa  #Rotate the z coordinate by the PA

                    #The distance from the center (in each coordinate) squared
                    y2 = yy * yy
                    z2 = zz * zz

                    #This rotates the coordinates in and out of the sky
                    zpci = zz * ci  #Rotate the z coordinate by the inclination.
                    zpsi = zz * si
                    #Subtract the offset
                    zpsi_dx = zpsi - dx

                    #The distance from the offset squared
                    yy_dy = yy - dy
                    yy_dy2 = yy_dy * yy_dy

                    image[j, i] = quad(integrand_dxdy_1g,
                                       -R2,
                                       R2,
                                       epsrel=0.5e-3,
                                       limit=75,
                                       args=(yy_dy2, y2, zp, z2, zpsi_dx, zpci,
                                             R1, R2, beta, a_r, g1, g1_2, ci, si, maxe, dx,
                                             dy, k))[0]

    # print("Running time: ", datetime.now()-starttime)

    # # normalize the HG function by the width
    image = image / a_r

    # normalize the HG function at the PA
    image = Norm * image / hg_90

    # add offset
    image = image + offset

    return image



def integrand_dxdy_2g(xp, yp_dy2, yp2, zp, zp2, zpsi_dx, zpci, R1, R2, beta,
                      a_r, g1, g1_2, g2, g2_2, alpha1, ci, si, maxe, dx, dy,
                      k):
    # author : Max Millar Blanchaer
    # compute the scattering integrand
    # see analytic-disk.nb

    xx = (xp * ci + zpsi_dx)

    d1 = mt.sqrt((yp_dy2 + xx * xx))

    if (d1 < R1 or d1 > R2):
        return 0.0

    d2 = xp * xp + yp2 + zp2

    #The line of sight scattering angle
    cos_phi = xp / mt.sqrt(d2)
    # phi=np.arccos(cos_phi)

    #Henyey Greenstein function
    hg1 = k * alpha1 * (1. - g1_2) / (1. + g1_2 - (2 * g1 * cos_phi))**1.5
    hg2 = k * (1 - alpha1) * (1. - g2_2) / (1. + g2_2 -
                                            (2 * g2 * cos_phi))**1.5

    hg = hg1 + hg2

    #Radial power low r propto -beta
    int1 = hg * (R1 / d1)**beta

    #The scale height function
    zz = (zpci - xp * si)
    hh = (a_r * d1)
    expo = zz * zz / (hh * hh)

    # if expo > 2*maxe:   # cut off exponential after 28 e-foldings (~ 1E-06)
    #     return 0.0

    int2 = np.exp(0.5 * expo)
    int3 = int2 * d2

    return int1 / int3


def gen_disk_dxdy_2g(dim,
                     param_disk,
                     mask=None,
                     sampling=1,
                     distance=72.8,
                     pixscale=0.01414):
    """ author : Max Millar Blanchaer
        modified by Johan Mazoyer
        create a 2g SPF disk model. The disk is normalized at 1 at 90degree
        (before star offset). also normalized by aspect_ratio. These
        normalization avoid weird correlation in the parameters


    Args:
        dim: dimension of the image in pixel assuming square image
        param_disk: a dict with keywords: 
                R1: inner radius of the disk
                R2: outer radius of the disk
                beta: radial power law of the disk between R1 and R2
                aspect_ratio=0.1 vertical width of the disk
                g1: %, 1st HG param
                g2: %, 2nd HG param
                Aplha: %, relative HG weight
                inc: degree, inclination
                pa: degree, principal angle
                dx: au, + -> NW offset disk plane Minor Axis
                dy: au, + -> SW offset disk plane Major Axis
                offset: vertical residue image
        mask: a np.where result that give where the model should be
              measured (important to save a lot of time)
        sampling: increase this parameter to bin the model
                  and save time
        distance: distance of the star
        pixscale: pixel scale of the instrument

    Returns:
        a 2d model
    """

    R1 = param_disk['r1']
    R2 = param_disk['r2']
    beta = param_disk['beta']

    inc = param_disk['inc']
    pa = param_disk['PA']
    dx = param_disk['dx']
    dy = param_disk['dy']
    Norm = param_disk['Norm']

    g1 = param_disk['g1']
    g2 = param_disk['g2']
    alpha1 = param_disk['alpha1']

    aspect_ratio = param_disk['a_r']
    offset = param_disk['offset']

    max_fov = dim / 2. * pixscale  #maximum radial distance in AU from the center to the edge
    npts = int(np.floor(dim / sampling))
    xsize = max_fov * distance  #maximum radial distance in AU from the center to the edge

    #The coordinate system here [x,y,z] is defined :
    # +ve x is the line of sight
    # +ve y is going right from the center
    # +ve z is going up from the center

    # y = np.linspace(0,xsize,num=npts/2)
    y = np.linspace(-xsize, xsize, num=npts)
    z = np.linspace(-xsize, xsize, num=npts)

    #Only need to compute half the image
    # image =np.zeros((npts,npts/2+1))
    image = np.zeros((npts, npts))

    #Some things we can precompute ahead of time
    maxe = mt.log(np.finfo('f').max)  #The log of the machine precision

    #Inclination Calculations
    incl = np.radians(90 - inc)
    ci = mt.cos(incl)  #Cosine of inclination
    si = mt.sin(incl)  #Sine of inclination

    #Position angle calculations
    pa_rad = np.radians(90 - pa)  #The position angle in radians
    cos_pa = mt.cos(pa_rad)  #Calculate these ahead of time
    sin_pa = mt.sin(pa_rad)

    #HG g value squared
    g1_2 = g1 * g1  #First HG g squared
    g2_2 = g2 * g2  #Second HG g squared
    #Constant for HG function
    k = 1. / (4 * np.pi)

    #The aspect ratio
    a_r = aspect_ratio

    #Henyey Greenstein function at 90
    hg1_90 = k * alpha1 * (1. - g1_2) / (1. + g1_2)**1.5
    hg2_90 = k * (1 - alpha1) * (1. - g2_2) / (1. + g2_2)**1.5

    hg_90 = hg1_90 + hg2_90

    #If there's no mask then calculate for the full image
    if len(np.shape(mask)) < 2:

        for i, yp in enumerate(y):
            for j, zp in enumerate(z):

                #This rotates the coordinates in the image frame
                yy = yp * cos_pa - zp * sin_pa  #Rotate the y coordinate by the PA
                zz = yp * sin_pa + zp * cos_pa  #Rotate the z coordinate by the PA

                #The distance from the center (in each coordinate) squared
                y2 = yy * yy
                z2 = zz * zz

                #This rotates the coordinates in and out of the sky
                zpci = zz * ci  #Rotate the z coordinate by the inclination.
                zpsi = zz * si
                #Subtract the offset
                zpsi_dx = zpsi - dx

                #The distance from the offset squared
                yy_dy = yy - dy
                yy_dy2 = yy_dy * yy_dy

                image[j, i] = quad(integrand_dxdy_2g,
                                   -R2,
                                   R2,
                                   epsrel=0.5e-3,
                                   limit=75,
                                   args=(yy_dy2, y2, zp, z2, zpsi_dx, zpci, R1,
                                         R2, beta, a_r, g1, g1_2, g2, g2_2,
                                         alpha1, ci, si, maxe, dx, dy, k))[0]

    #If there is a mask then don't calculate disk there
    else:
        hmask = mask
        # hmask = mask[:,140:] #Use only half the mask

        for i, yp in enumerate(y):
            for j, zp in enumerate(z):

                # if hmask[j,npts/2+i]: #This assumes
                # that the input mask has is the same size as
                # the desired image (i.e. ~ size / sampling)
                if hmask[j, i]:

                    image[j, i] = 0.  #np.nan

                else:

                    #This rotates the coordinates in the image frame
                    yy = yp * cos_pa - zp * sin_pa  #Rotate the y coordinate by the PA
                    zz = yp * sin_pa + zp * cos_pa  #Rotate the z coordinate by the PA

                    #The distance from the center (in each coordinate) squared
                    y2 = yy * yy
                    z2 = zz * zz

                    #This rotates the coordinates in and out of the sky
                    zpci = zz * ci  #Rotate the z coordinate by the inclination.
                    zpsi = zz * si
                    #Subtract the offset
                    zpsi_dx = zpsi - dx

                    #The distance from the offset squared
                    yy_dy = yy - dy
                    yy_dy2 = yy_dy * yy_dy

                    image[j, i] = quad(integrand_dxdy_2g,
                                       -R2,
                                       R2,
                                       epsrel=0.5e-3,
                                       limit=75,
                                       args=(yy_dy2, y2, zp, z2, zpsi_dx, zpci,
                                             R1, R2, beta, a_r, g1, g1_2, g2,
                                             g2_2, alpha1, ci, si, maxe, dx,
                                             dy, k))[0]

    # print("Running time: ", datetime.now()-starttime)

    # # normalize the HG function by the width
    image = image / a_r

    # normalize the HG function at the PA
    image = Norm * image / hg_90

    # add offset
    image = image + offset

    return image


def integrand_dxdy_3g(xp, yp_dy2, yp2, zp, zp2, zpsi_dx, zpci, R1, R2, beta,
                      a_r, g1, g1_2, g2, g2_2, g3, g3_2, alpha1, alpha2, ci,
                      si, maxe, dx, dy, k):

    # compute the scattering integrand
    # see analytic-disk.nb

    xx = (xp * ci + zpsi_dx)

    d1 = mt.sqrt((yp_dy2 + xx * xx))

    if (d1 < R1 or d1 > R2):
        return 0.0

    d2 = xp * xp + yp2 + zp2

    #The line of sight scattering angle
    cos_phi = xp / mt.sqrt(d2)
    # phi=np.arccos(cos_phi)

    #Henyey Greenstein function
    # hg1=k*alpha1*     (1. - g1_2)/(1. + g1_2 - (2*g1*cos_phi))**1.5
    # hg2=k*(1-alpha1)* (1. - g2_2)/(1. + g2_2 - (2*g2*cos_phi))**1.5

    #Henyey Greenstein function
    hg1 = k * (1. - g1_2) / (1. + g1_2 - (2 * g1 * cos_phi))**1.5
    hg2 = k * (1. - g2_2) / (1. + g2_2 - (2 * g2 * cos_phi))**1.5
    hg3 = k * (1. - g3_2) / (1. + g3_2 - (2 * g3 * cos_phi))**1.5

    hg = alpha1 * hg1 + alpha2 * hg2 + (1 - alpha1 - alpha2) * hg3
    #Radial power low r propto -beta
    int1 = hg * (R1 / d1)**beta

    #The scale height function
    zz = (zpci - xp * si)
    hh = (a_r * d1)
    expo = zz * zz / (hh * hh)

    # if expo > 2*maxe:   # cut off exponential after 28 e-foldings (~ 1E-06)
    #     return 0.0

    int2 = np.exp(0.5 * expo)
    int3 = int2 * d2

    return int1 / int3



def gen_disk_dxdy_3g(dim,
                     param_disk,
                     mask=None,
                     sampling=1,
                     distance=72.8,
                     pixscale=0.01414):
    """ adapted from  Max Millar Blanchaer by Johan Mazoyer
        create a 3g SPF disk model. The disk is normalized at 1 at 90degree
        (before star offset). also normalized by aspect_ratio


    Args:
        dim: dimension of the image in pixel assuming square image
        param_disk: a dict with keywords: 
            R1: inner radius of the disk
            R2: outer radius of the disk
            beta: radial power law of the disk between R1 and R2
            aspect_ratio vertical width of the disk
            g1: %, 1st HG param
            g2: %, 2nd HG param
            g3: %, 3rd HG param
            Aplha1: %, first relative HG weight
            Aplha2: %, second relative HG weight
            inc: degree, inclination
            pa: degree, principal angle
            dx: au, + -> NW offset disk plane Minor Axis
            dy: au, + -> SW offset disk plane Major Axis
            offset: vertical residue image
        mask: a np.where result that give where the model should be
            measured (important to save a lot of time)
        sampling: increase this parameter to bin the model
                and save time
        distance: distance of the star
        pixscale: pixel scale of the instrument

    Returns:
        a 2d model
    """
    R1 = param_disk['r1']
    R2 = param_disk['r2']
    beta = param_disk['beta']

    inc = param_disk['inc']
    pa = param_disk['PA']
    dx = param_disk['dx']
    dy = param_disk['dy']
    Norm = param_disk['Norm']

    g1 = param_disk['g1']
    g2 = param_disk['g2']
    alpha1 = param_disk['alpha1']

    g3 = param_disk['g3']
    alpha2 = param_disk['alpha2']
    
    aspect_ratio = param_disk['a_r']
    offset = param_disk['offset']

    # starttime=datetime.now()

    max_fov = dim / 2. * pixscale  #maximum radial distance in AU from the center to the edge
    npts = int(np.floor(dim / sampling))
    xsize = max_fov * distance  #maximum radial distance in AU from the center to the edge

    #The coordinate system here [x,y,z] is defined :
    # +ve x is the line of sight
    # +ve y is going right from the center
    # +ve z is going up from the center

    # y = np.linspace(0,xsize,num=npts/2)
    y = np.linspace(-xsize, xsize, num=npts)
    z = np.linspace(-xsize, xsize, num=npts)

    #Only need to compute half the image
    # image =np.zeros((npts,npts/2+1))
    image = np.zeros((npts, npts))

    #Some things we can precompute ahead of time
    maxe = mt.log(np.finfo('f').max)  #The log of the machine precision

    #Inclination Calculations
    incl = np.radians(90 - inc)
    ci = mt.cos(incl)  #Cosine of inclination
    si = mt.sin(incl)  #Sine of inclination

    #Position angle calculations
    pa_rad = np.radians(90 - pa)  #The position angle in radians
    cos_pa = mt.cos(pa_rad)  #Calculate these ahead of time
    sin_pa = mt.sin(pa_rad)

    #HG g value squared
    g1_2 = g1 * g1  #First HG g squared
    g2_2 = g2 * g2  #Second HG g squared
    g3_2 = g3 * g3  #Second HG g squared

    #Constant for HG function
    k = 1. / (4 * np.pi) * 100
    ### we add a 100 multiplicateur to k avoid hg values to be too small, it makes the integral fail on certains points
    ### Since we normalize by hg90 at the end, this has no impact on the actual model

    #The aspect ratio
    a_r = aspect_ratio

    #Henyey Greenstein function at 90

    hg1_90 = k * (1. - g1_2) / (1. + g1_2)**1.5
    hg2_90 = k * (1. - g2_2) / (1. + g2_2)**1.5
    hg3_90 = k * (1. - g3_2) / (1. + g3_2)**1.5

    hg_90 = alpha1 * hg1_90 + alpha2 * hg2_90 + (1 - alpha1 - alpha2) * hg3_90

    #If there's no mask then calculate for the full image
    if len(np.shape(mask)) < 2:

        for i, yp in enumerate(y):
            for j, zp in enumerate(z):

                #This rotates the coordinates in the image frame
                yy = yp * cos_pa - zp * sin_pa  #Rotate the y coordinate by the PA
                zz = yp * sin_pa + zp * cos_pa  #Rotate the z coordinate by the PA

                #The distance from the center (in each coordinate) squared
                y2 = yy * yy
                z2 = zz * zz

                #This rotates the coordinates in and out of the sky
                zpci = zz * ci  #Rotate the z coordinate by the inclination.
                zpsi = zz * si
                #Subtract the offset
                zpsi_dx = zpsi - dx

                #The distance from the offset squared
                yy_dy = yy - dy
                yy_dy2 = yy_dy * yy_dy

                # image[j,i]=  quad(integrand_dxdy_2g, -R2, R2, epsrel=0.5e-3,limit=75,args=(yy_dy2,y2,zp,z2,zpsi_dx,zpci,R1,R2,beta,a_r,g1,g1_2,g2,g2_2, alpha1,ci,si,maxe,dx,dy,k))[0]
                image[j, i] = quad(integrand_dxdy_3g,
                                   -R2,
                                   R2,
                                   epsrel=0.5e-12,
                                   limit=75,
                                   args=(yy_dy2, y2, zp, z2, zpsi_dx, zpci, R1,
                                         R2, beta, a_r, g1, g1_2, g2, g2_2, g3,
                                         g3_2, alpha1, alpha2, ci, si, maxe,
                                         dx, dy, k))[0]

    #If there is a mask then don't calculate disk there
    else:
        hmask = mask
        # hmask = mask[:,140:] #Use only half the mask

        for i, yp in enumerate(y):
            for j, zp in enumerate(z):

                # if hmask[j,npts/2+i]: #This assumes that the input mask has is the same size as the desired image (i.e. ~ size / sampling)
                if hmask[j, i]:

                    image[j, i] = 0.  #np.nan

                else:

                    #This rotates the coordinates in the image frame
                    yy = yp * cos_pa - zp * sin_pa  #Rotate the y coordinate by the PA
                    zz = yp * sin_pa + zp * cos_pa  #Rotate the z coordinate by the PA

                    #The distance from the center (in each coordinate) squared
                    y2 = yy * yy
                    z2 = zz * zz

                    #This rotates the coordinates in and out of the sky
                    zpci = zz * ci  #Rotate the z coordinate by the inclination.
                    zpsi = zz * si
                    #Subtract the offset
                    zpsi_dx = zpsi - dx

                    #The distance from the offset squared
                    yy_dy = yy - dy
                    yy_dy2 = yy_dy * yy_dy

                    # image[j,i]=  quad(integrand_dxdy_2g, -R2, R2, epsrel=0.5e-3,limit=75,args=(yy_dy2,y2,zp,z2,zpsi_dx,zpci,R1,R2,beta,a_r,g1,g1_2,g2,g2_2, alpha1,ci,si,maxe,dx,dy,k))[0]
                    image[j, i] = quad(integrand_dxdy_3g,
                                       -R2,
                                       R2,
                                       epsrel=0.5e-12,
                                       limit=75,
                                       args=(yy_dy2, y2, zp, z2, zpsi_dx, zpci,
                                             R1, R2, beta, a_r, g1, g1_2, g2,
                                             g2_2, g3, g3_2, alpha1, alpha2,
                                             ci, si, maxe, dx, dy, k))[0]

    # # normalize the HG function by the width
    image = image / a_r

    # normalize the HG function at the PA
    image = Norm * image / hg_90

    # add offset
    image = image + offset
    # print("Running time 3g: {0}".format(datetime.now()-starttime))

    return image


def integrand_dxdy_flat(xp, yp_dy2, yp2, zp, zp2, zpsi_dx, zpci, R1, R2, beta,
                        a_r, ci, si, maxe, dx, dy):

    # compute the scattering integrand
    # see analytic-disk.nb

    xx = (xp * ci + zpsi_dx)

    d1 = mt.sqrt((yp_dy2 + xx * xx))

    if (d1 < R1 or d1 > R2):
        return 0.0

    d2 = xp * xp + yp2 + zp2

    #The line of sight scattering angle
    # cos_phi = xp / mt.sqrt(d2)
    # phi=np.arccos(cos_phi)

    #Radial power low r propto -beta
    int1 = (R1 / d1)**beta

    #The scale height function
    zz = (zpci - xp * si)
    hh = (a_r * d1)
    expo = zz * zz / (hh * hh)

    # if expo > 2*maxe:   # cut off exponential after 28 e-foldings (~ 1E-06)
    #     return 0.0

    int2 = np.exp(0.5 * expo)
    int3 = int2 * d2

    return int1 / int3


def gen_disk_dxdy_flat(dim,
                       R1=74.42,
                       R2=82.45,
                       beta=1.0,
                       aspect_ratio=0.1,
                       inc=76.49,
                       pa=30,
                       dx=0,
                       dy=0.,
                       mask=None,
                       sampling=1,
                       distance=72.8,
                       pixscale=0.01414):
    """ adapted from  Max Millar Blanchaer by Johan Mazoyer
        create a 3g SPF disk model. The disk is normalized at 1 at 90degree
        (before star offset). also normalized by aspect_ratio


    Args:
        dim: dimension of the image in pixel assuming square image
        R1: inner radius of the disk
        R2: outer radius of the disk
        beta: radial power law of the disk between R1 and R2
        aspect_ratio=0.1 vertical width of the disk
        g1: %, 1st HG param
        g2: %, 2nd HG param
        g3: %, 3rd HG param
        Aplha1: %, first relative HG weight
        Aplha2: %, second relative HG weight
        inc: degree, inclination
        pa: degree, principal angle
        dx: au, + -> NW offset disk plane Minor Axis
        dy: au, + -> SW offset disk plane Major Axis
        mask: a np.where result that give where the model should be
              measured (important to save a lot of time)
        sampling: increase this parameter to bin the model
                  and save time
        distance: distance of the star
        pixscale: pixel scale of the instrument

    Returns:
        a 2d model
    """
    # starttime=datetime.now()

    max_fov = dim / 2. * pixscale  #maximum radial distance in AU from the center to the edge
    npts = int(np.floor(dim / sampling))
    xsize = max_fov * distance  #maximum radial distance in AU from the center to the edge

    #The coordinate system here [x,y,z] is defined :
    # +ve x is the line of sight
    # +ve y is going right from the center
    # +ve z is going up from the center

    # y = np.linspace(0,xsize,num=npts/2)
    y = np.linspace(-xsize, xsize, num=npts)
    z = np.linspace(-xsize, xsize, num=npts)

    #Only need to compute half the image
    # image =np.zeros((npts,npts/2+1))
    image = np.zeros((npts, npts))

    #Some things we can precompute ahead of time
    maxe = mt.log(np.finfo('f').max)  #The log of the machine precision

    #Inclination Calculations
    incl = np.radians(90 - inc)
    ci = mt.cos(incl)  #Cosine of inclination
    si = mt.sin(incl)  #Sine of inclination

    #Position angle calculations
    pa_rad = np.radians(90 - pa)  #The position angle in radians
    cos_pa = mt.cos(pa_rad)  #Calculate these ahead of time
    sin_pa = mt.sin(pa_rad)

    #The aspect ratio
    a_r = aspect_ratio

    #If there's no mask then calculate for the full image
    if len(np.shape(mask)) < 2:

        for i, yp in enumerate(y):
            for j, zp in enumerate(z):

                #This rotates the coordinates in the image frame
                yy = yp * cos_pa - zp * sin_pa  #Rotate the y coordinate by the PA
                zz = yp * sin_pa + zp * cos_pa  #Rotate the z coordinate by the PA

                #The distance from the center (in each coordinate) squared
                y2 = yy * yy
                z2 = zz * zz

                #This rotates the coordinates in and out of the sky
                zpci = zz * ci  #Rotate the z coordinate by the inclination.
                zpsi = zz * si
                #Subtract the offset
                zpsi_dx = zpsi - dx

                #The distance from the offset squared
                yy_dy = yy - dy
                yy_dy2 = yy_dy * yy_dy

                # image[j,i]=  quad(integrand_dxdy_2g, -R2, R2, epsrel=0.5e-3,limit=75,args=(yy_dy2,y2,zp,z2,zpsi_dx,zpci,R1,R2,beta,a_r,g1,g1_2,g2,g2_2, alpha1,ci,si,maxe,dx,dy,k))[0]
                image[j,
                      i] = quad(integrand_dxdy_flat,
                                -R2,
                                R2,
                                epsrel=0.5e-12,
                                limit=75,
                                args=(yy_dy2, y2, zp, z2, zpsi_dx, zpci, R1,
                                      R2, beta, a_r, ci, si, maxe, dx, dy))[0]

    #If there is a mask then don't calculate disk there
    else:
        hmask = mask
        # hmask = mask[:,140:] #Use only half the mask

        for i, yp in enumerate(y):
            for j, zp in enumerate(z):

                # if hmask[j,npts/2+i]: #This assumes that the input mask has is the same size as the desired image (i.e. ~ size / sampling)
                if hmask[j, i]:

                    image[j, i] = 0.  #np.nan

                else:

                    #This rotates the coordinates in the image frame
                    yy = yp * cos_pa - zp * sin_pa  #Rotate the y coordinate by the PA
                    zz = yp * sin_pa + zp * cos_pa  #Rotate the z coordinate by the PA

                    #The distance from the center (in each coordinate) squared
                    y2 = yy * yy
                    z2 = zz * zz

                    #This rotates the coordinates in and out of the sky
                    zpci = zz * ci  #Rotate the z coordinate by the inclination.
                    zpsi = zz * si
                    #Subtract the offset
                    zpsi_dx = zpsi - dx

                    #The distance from the offset squared
                    yy_dy = yy - dy
                    yy_dy2 = yy_dy * yy_dy

                    # image[j,i]=  quad(integrand_dxdy_2g, -R2, R2, epsrel=0.5e-3,limit=75,args=(yy_dy2,y2,zp,z2,zpsi_dx,zpci,R1,R2,beta,a_r,g1,g1_2,g2,g2_2, alpha1,ci,si,maxe,dx,dy,k))[0]
                    image[j, i] = quad(integrand_dxdy_flat,
                                       -R2,
                                       R2,
                                       epsrel=0.5e-12,
                                       limit=75,
                                       args=(yy_dy2, y2, zp, z2, zpsi_dx, zpci,
                                             R1, R2, beta, a_r, ci, si, maxe,
                                             dx, dy))[0]

    # # normalize the HG function by the width
    image = image / a_r

    # print("Running time 3g: {0}".format(datetime.now()-starttime))

    return image
