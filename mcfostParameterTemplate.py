####################### ####################### ####################### 
####################### READ ME PLEASE !!!!!!!! #######################
####################### ####################### ####################### 
# This code works only in Python 3 
# (2.x does not work because of the print function).
# This code works only for MCFOST 3.0 
# (the 2.0 version has the "#cavity" block which is dropped in 3.0.)

# This code create a template data structure for the MCFOST parameter file 
#                                         (type: collections.OrderedDict())  
# which is called then by display_file() to either 
# display it properly on screen, i.e., in the MCFOST parameter file format,
# or save to a file.

# Generate the data structure with the following command:
#
#     >>>        sample_para_dict = generateMcfostTemplate(n_zone = n_zone, n_species = n_species, n_star = n_star)  <<<
# 
#         where (1) n_zone is the value for the "#Number of zones" block in a MCFOST parameter file, e.g., n_zone = 2
#               (2) n_spieces the value for the "Number of species" in the "#Grain properties" block,
#                          Depending on the n_zone number, it should be 
#                             a list with len(n_spieces) = n_zone, e.g., n_species = [1, 2]
#               (3) n_star is the value for the "Number of stars" in the "#Star properties" block
#     Use the following command for the Docstring of the function for an example:
#     >>>        ?generateMcfostTemplate

#######################################################################
################ ADVANTAGES of this creator: ##########################
#######################################################################
#             The keywords in the MCFOST parameter file               #
#                        can be used directly.                        #
#######################################################################
#         It supports (1) multiple zones                              #
#                     (2) for each zone, multiple species,            #
#                and  (3) multiple stars.                             #                            
#######################################################################
# To access the values of the three above supported features, the     #
# only added parameters are:                                          #
#                 'zoneX', 'speciesY', and 'starZ'                    #
# where X, Y, Z are the numbers (starting from 0, ending by number-1) #
#######################################################################


######## Data Structure #######
# The data that generateMcfostTemplate() returns, i.e., sample_para_dict, is a collections.OrderedDict() type structure.
# sample_para_dict is an ordered dict, and the order is as follows:
# 1. First index: the 13 blocks in the MCFOST parameter file.
#    The 13 blocks are also of type OrderedDict(), except for a few with only one value. Call them with:
#     >>>        sample_para_dict['mcfost version']   <<< for the version of mcfost (a value: NOT an OrderedDict)
#     >>>        sample_para_dict['#Number of photon packages']   <<< for that block
#     >>>        sample_para_dict['#Wavelength']   <<< for that block
#     >>>        sample_para_dict['#Number of photon packages']   <<< for that block
#     >>>        sample_para_dict['#Grid geometry and size']   <<< for that block
#     >>>        sample_para_dict['#Maps']   <<< for that block
#     >>>        sample_para_dict['#Scattering method']   <<< for that block
#     >>>        sample_para_dict['#Symmetries']   <<< for that block
#     >>>        sample_para_dict['#Disk physics']   <<< for that block
#     >>>        sample_para_dict['#Number of zones]   <<< for the number of zones (an integer: NOT an OrderedDict)
#     >>>        sample_para_dict['#Density structure']   <<< for that block
#     >>>        sample_para_dict['#Grain properties']   <<< for that block
#     >>>        sample_para_dict['#Molecular RT settings]   <<< for that block
#     >>>        sample_para_dict['#Star properties] for that block
# 2. Second+ index: for the block which are of type OrderedDict, call them with
#  Scenario 1: SIMPLE blocks, i.e., the ones EXCEPT the ones in the next line,
#                        (i.e., NOT: '#Density structure', '#Grain properties', or '#Star properties')
#                   for a given block name (e.g., '#Number of photon packages', '#Wavelength', '#Grid geometry and size', etc.)
#                        call them with:
#     >>>        sample_para_dict[block_name]['row0']    <<< 
#                                  to 
#     >>>        sample_para_dict[block_name]['row5']    <<< 
#                        if there are 6 rows (0, 1, 2, 3, 4, 5)
#              The above commands in Scenario 1 returns OrderedDict structures again,
#              Now, you can access the values or edit them with the following command example
#     >>>        sample_para_dict['#Maps']['row0']['nx']  <<< for acess the 'nx' parameter in the 0-th row of the '#Maps' block
#     >>>        sample_para_dict['#Maps']['row0']['nx'] = 500 <<< to edit that value
#  Scenario 2: COMPLEX blocks, i.e., the ones IN the ones in the next line,
#                        (i.e., '#Density structure', '#Grain properties', or '#Star properties')
#                        call them with:
#     >>>        sample_para_dict[block_name][DETAILS]['row0']    <<< 
#                                  to 
#     >>>        sample_para_dict[block_name][DETAILS]['row5']    <<< 
#                        if there are 6 rows (0, 1, 2, 3, 4, 5)
#              Now, to access the values, replace the [DETAILS] in the above two command lines in the following way
#              (1) For sample_para_dict['#Density structure'], use
#     >>>       sample_para_dict['#Density structure']['zone0']['row0']  <<<
#                   then we are back to Scenario 1. 
#                                 ('zone0' can be changed to 'zoneX', where X = 0 to n_zone-1)
#              (2) For sample_para_dict['#Grain properties'], use
#     >>>       sample_para_dict['#Grain properties']['zone0']['species0']['row0']  <<<
#                   then we are back to Scenario 1.
#                                 ('zone0' can be changed to 'zoneX', where X = 0 to n_zone-1)
#                                 ('species0' can be changed to 'speciesY', where Y = 0 to n_species[X]-1)
#              (3) For sample_para_dict['#Grain properties'], use
#     >>>       sample_para_dict['#Star properties']['star0']['row0']  <<<
#                   then we are back to Scenario 1.
#                                 ('star0' can be changed to 'starZ', where Z = 0 to n_star-1)

import collections
import copy
paramfile_dict = collections.OrderedDict()


# # Create the (1) density structure (2) grain property (3) star property templates
# # for the creation of OrderdDict


def generateDensityStructure(n_zone = 0):
    # a template for density structure:
    density_structure_template = collections.OrderedDict(
        [('row0', collections.OrderedDict([
                   ('zone type', 4)])),
         ('row1', collections.OrderedDict([
                   ('dust mass', 1.256e-07),
                   ('gas-to-dust mass ratio', 100.0)])),
         ('row2', collections.OrderedDict([
                   ('scale height', 4.450),
                   ('reference radius', 60.0),
                   ('vertical profile exponent', 0.725)])),
         ('row3', collections.OrderedDict([
                   ('Rin', 56.58),
                   ('edge', 2.0),
                   ('Rout', 80.0),
                   ('Rc', 60.0)])),
         ('row4', collections.OrderedDict([
                   ('flaring exponent', 1.000)])),
         ('row5', collections.OrderedDict([
                   ('surface density exponent/alpha_in', 1.625),
                   ('-gamma_exp/alpha_out', 1.493)]))
        ])
    if n_zone == 0:
        return density_structure_template
    else:
        density_structure = collections.OrderedDict()
        for i in range(n_zone):
            density_structure['zone' + str(i)] = copy.deepcopy(generateDensityStructure())
        return density_structure


def generateGrainProperties(number_of_species = 0):
    # a template for grain property:
    grain_properties_template = collections.OrderedDict(
        [('row0', collections.OrderedDict([
                   ('Grain type', 'Mie'),
                   ('N_components', 1),
                   ('mixing rule', 1),
                   ('porosity', 0.010),
                   ('mass fraction', 0.182),
                   ('Vmax', 0.9)])),
         ('row1', collections.OrderedDict([
                   ('Optical indices file', 'dlsi_opct.dat'),
                   ('volume fraction', 1.0)])),
         ('row2', collections.OrderedDict([
                   ('Heating method', 2)])),
         ('row3', collections.OrderedDict([
                   ('amin', 3.946),
                   ('amax', 1000.000),
                   ('aexp', 3.155),
                   ('nbr_grains', 100)]))
        ])
    
    if number_of_species == 0:
        return grain_properties_template
    else:
        grain_properties = collections.OrderedDict()
        for i in range(number_of_species):
            grain_properties['species' + str(i)] = copy.deepcopy(generateGrainProperties())
        return grain_properties


def generateStarProperties(n_star = 1):
    # a template for star structure:
    star_property_template = collections.OrderedDict([
         ('row0', collections.OrderedDict([
                   ('Temp', 6460.0),
                   ('radius', 1.23),
                   ('M', 1.30),
                   ('x', 0.0),
                   ('y', 0.0),
                   ('z', 0.0),
                   ('is a blackbody?', 'F')])),
         ('row1', 'Kurucz6500-4.0.fits.gz'),
         ('row2', collections.OrderedDict([
                   ('fUV', 0.),
                   ('slope_fUV', 2.2)]))])


    if n_star == 1:
        return star_property_template
    else:
        return 0


# # Create the OrderdDict

def generateMcfostTemplate(n_zone = 1, n_species = [1], n_star = 1):
    '''Generate a template dictionary (collections.OrderedDict) for n_zone zones (integer), and n_spieces (list).
    Input: (1) n_zone -- "#Number of zones" (integer), e.g., 2;
           (2) n_species -- "Number of species" in each zone (list), should have length equal to n_zone, e.g., [3, 4];
               The example means there are 2 zones, and the 1st zone has 3 grain spieces, and the 2nd has 4.
           (3) n_star -- "Number of stars" (integer), e.g., 1.
    Output: an ordered dictionary (collections.OrderdDict).
    Example:
>>>         n_zone = 1 #2
>>>         n_species = [1] #[3, 2]
>>>         n_star = 1 #2
>>>         sample_para_dict = generateMcfostTemplate(n_zone = n_zone, n_species = n_species, n_star = n_star)
        The above commands returns an OrderedDict named 'sample_para_dict',
        and it has 1 disk zone, and 1 grain species, and 1 star.
        The commented has 2 disk zones, and 3 grains / 2 grains in the two zones, respectively, with 2 stars in the system
            '''
    if n_zone > 1 and len(n_species) != n_zone:
        raise ValueError("n_species should be a list, and len(n_spieces)=n_zone! Returning Error")
    
    ###################################################################
    ############# Now the MCFOST input parameter template #############
    ###################################################################

    paramfile_dict['mcfost version'] = 3.0

    paramfile_dict['#Number of photon packages'] = collections.OrderedDict([
         ('row0', collections.OrderedDict([
                   ('nbr_photons_eq_th', 1.28e+05)])), 
         ('row1', collections.OrderedDict([
                   ('nbr_photons_lambda', 1000)])), 
         ('row2', collections.OrderedDict([
                   ('nbr_photons_image', 1.28e+05)]))])

    paramfile_dict['#Wavelength'] = collections.OrderedDict([
         ('row0', collections.OrderedDict([
                   ('n_lambda', 300),
                   ('lambda_min', 0.10),
                   ('lambda_max', 3000)])),
         ('row1', collections.OrderedDict([
                   ('compute temperature?', 'T'),
                   ('compute sed?', 'T'),
                   ('use default wavelength grid?', 'T')])),
         ('row2', collections.OrderedDict([
                   ('wavelength file', 'IMLup.lambda')])),
         ('row3', collections.OrderedDict([
                   ('separation of different contributions?', 'F'),
                   ('stokes parameters?', 'T')]))])

    paramfile_dict['#Grid geometry and size'] = collections.OrderedDict([
         ('row0', 1),
         ('row1', collections.OrderedDict([
                   ('n_rad', 15),
                   ('nz', 10),
                   ('n_az', 1),
                   ('n_rad_in', 1)]))])

    paramfile_dict['#Maps'] = collections.OrderedDict([
         ('row0', collections.OrderedDict([
                   ('nx', 281),
                   ('ny', 281),
                   ('size', 409.6)])),
         ('row1', collections.OrderedDict([
                   ('imin', 83.96),
                   ('imax', 83.96),
                   ('n_incl', 1),
                   ('centered ?', 'T')])),
         ('row2', collections.OrderedDict([
                   ('az_min', 0),
                   ('az_max', 0),
                   ('n_az', 1)])),
         ('row3', collections.OrderedDict([
                   ('distance', 102.90)])),
         ('row4', collections.OrderedDict([
                   ('disk PA', -75.00)]))])

    paramfile_dict['#Scattering method'] = collections.OrderedDict([
         ('row0', 0),
         ('row1', 1)])

    paramfile_dict['#Symmetries'] = collections.OrderedDict([
         ('row0', collections.OrderedDict([('image symmetry', 'T')])),
         ('row1', collections.OrderedDict([('central symmetry', 'T')])),
         ('row2', collections.OrderedDict([('axial symmetry', 'T')]))])

    paramfile_dict['#Disk physics'] = collections.OrderedDict([
          ('row0', collections.OrderedDict([
                   ('dust_settling', 0),
                   ('exp_strat', 0.5),
                   ('a_strat', 1.00)])),
          ('row1', collections.OrderedDict([
                   ('dust radial migration', 'F')])),
          ('row2', collections.OrderedDict([
                   ('sublimate dust', 'F')])),
          ('row3', collections.OrderedDict([
                   ('hydrostatic equilibrium', 'F')])),
          ('row4', collections.OrderedDict([
                   ('viscous heating', 'F'),
                   ('alpha_viscosity', 1e-05)]))])

    paramfile_dict['#Number of zones'] = n_zone

    paramfile_dict['#Density structure'] = generateDensityStructure(n_zone)

    grain_props = collections.OrderedDict()
    for i in range(n_zone):
        grain = collections.OrderedDict()
        grain_props['zone'+str(i)] = copy.deepcopy(grain)
        grain_props['zone'+str(i)]['Number of species'] = n_species[i]
        for j in range(n_species[i]):
            grain_props['zone'+str(i)]['species' + str(j)] = copy.deepcopy(generateGrainProperties())
        
    paramfile_dict['#Grain properties'] = grain_props
            

    paramfile_dict['#Molecular RT settings'] = collections.OrderedDict([
         ('row0', collections.OrderedDict([
                   ('lpop', 'F'),
                   ('laccurate_pop', 'F'),
                   ('LTE', 'F'),
                   ('profile width', 15.0)])),
         ('row1', collections.OrderedDict([
                   ('v_turb (delta)', 0.2)])),
         ('row2', collections.OrderedDict([
                   ('nmol', 1)])),
         ('row3', collections.OrderedDict([
                   ('molecular data filename', 'co@xpol.dat'),
                   ('level_max', 6)])),
         ('row4', collections.OrderedDict([
                   ('vmax', 1.0),
                   ('n_speed', 50)])),
         ('row5', collections.OrderedDict([
                   ('cst molecule abundance ?', 'T'),
                   ('abundance', 1.0e-6),
                   ('abundance file', 'abundance.fits.gz')])),
         ('row6', collections.OrderedDict([
                   ('ray tracing ?', 'T'),
                   ('number of lines in ray-tracing', 3)])),
         ('row7', collections.OrderedDict([
                   ('transition numbers', '1 2 3')]))])
    
    star_props = collections.OrderedDict()
    star_props['Number of stars'] = n_star
    for i in range(n_star):
        star = copy.deepcopy(generateStarProperties())
        star_props['star'+str(i)] = star
    paramfile_dict['#Star properties'] = star_props

    return paramfile_dict

def print0_mcfost_version(sample_para_dict):
    print(sample_para_dict['mcfost version'], '\t', 'mcfost version')
    print()
    
def print1_Number_of_photon_packages(sample_para_dict):
    print('#Number of photon packages')
    for row_name in sample_para_dict['#Number of photon packages']:
        for item in sample_para_dict['#Number of photon packages'][row_name]:
            print(sample_para_dict['#Number of photon packages'][row_name][item], '\t', item)
    print()
    
def print2_Wavelength(sample_para_dict):
    block_name = '#Wavelength'
    print(block_name)
    for row_name in sample_para_dict[block_name]:
        for item in sample_para_dict[block_name][row_name]:
            print(sample_para_dict[block_name][row_name][item], end = ' ')
        print('\t', end = '')
        for item in sample_para_dict[block_name][row_name]:
            print(item, end = ', ')
        print()
    print()
        
def print3_Grid_geometry_and_size(sample_para_dict):
    block_name = '#Grid geometry and size'
    print(block_name)
    for row_name in sample_para_dict[block_name]:
        if type(sample_para_dict[block_name][row_name]) != type(collections.OrderedDict()):
            print(sample_para_dict[block_name][row_name], '\t', '1 = cylindrical, 2 = spherical, 3 = Voronoi tesselation (this is in beta, please ask Christophe)')
        else:
            for item in sample_para_dict[block_name][row_name]:
                print(sample_para_dict[block_name][row_name][item], end = ' ')
            print('\t', end = '')
            for item in sample_para_dict[block_name][row_name]:
                print(item, end = ', ')
            print('')
    print()

def print4_Maps(sample_para_dict):
    block_name = '#Maps'
    print(block_name)
    for row_name in sample_para_dict[block_name]:
        if type(sample_para_dict[block_name][row_name]) != type(collections.OrderedDict()):
            print(sample_para_dict[block_name][row_name])#, '\t', '1 = cylindrical, 2 = spherical, 3 = Voronoi tesselation (this is in beta, please ask Christophe)')
        else:
            for item in sample_para_dict[block_name][row_name]:
                print(sample_para_dict[block_name][row_name][item], end = ' ')
            print('\t', end = '')
            for item in sample_para_dict[block_name][row_name]:
                print(item, end = ', ')
            print('')
    print()   
    
def print5_Scattering_method(sample_para_dict):
    block_name = '#Scattering method'
    print(block_name)
    for row_name in sample_para_dict[block_name]:
        if type(sample_para_dict[block_name][row_name]) != type(collections.OrderedDict()):
            print(sample_para_dict[block_name][row_name])#, '\t', '1 = cylindrical, 2 = spherical, 3 = Voronoi tesselation (this is in beta, please ask Christophe)')
        else:
            for item in sample_para_dict[block_name][row_name]:
                print(sample_para_dict[block_name][row_name][item], end = ' ')
            print('\t', end = '')
            for item in sample_para_dict[block_name][row_name]:
                print(item, end = ', ')
            print('')
    print() 
    
def print6_Symmetries(sample_para_dict):
    block_name = '#Symmetries'
    print(block_name)
    for row_name in sample_para_dict[block_name]:
        if type(sample_para_dict[block_name][row_name]) != type(collections.OrderedDict()):
            print(sample_para_dict[block_name][row_name])#, '\t', '1 = cylindrical, 2 = spherical, 3 = Voronoi tesselation (this is in beta, please ask Christophe)')
        else:
            for item in sample_para_dict[block_name][row_name]:
                print(sample_para_dict[block_name][row_name][item], end = ' ')
            print('\t', end = '')
            for item in sample_para_dict[block_name][row_name]:
                print(item, end = ', ')
            print('')
    print()   
    
def print7_Disk_physics(sample_para_dict):
    block_name = '#Disk physics'
    print(block_name)
    for row_name in sample_para_dict[block_name]:
        if type(sample_para_dict[block_name][row_name]) != type(collections.OrderedDict()):
            print(sample_para_dict[block_name][row_name])#, '\t', '1 = cylindrical, 2 = spherical, 3 = Voronoi tesselation (this is in beta, please ask Christophe)')
        else:
            for item in sample_para_dict[block_name][row_name]:
                print(sample_para_dict[block_name][row_name][item], end = ' ')
            print('\t', end = '')
            for item in sample_para_dict[block_name][row_name]:
                print(item, end = ', ')
            print('')
    print()  
    
def print8_Number_of_zones(sample_para_dict):
    block_name = '#Number of zones'
    print(block_name + ': 1 zone = 1 density structure + corresponding grain properties')
    print(sample_para_dict[block_name])
    print('')
    
def print9_Density_structure(sample_para_dict):
    block_name = '#Density structure'
    print(block_name)
    for zone_id in range(sample_para_dict['#Number of zones']):
        block_zone = sample_para_dict[block_name]['zone' + str(zone_id)]
        for i, row_name in enumerate(block_zone):
            if type(block_zone[row_name]) != type(collections.OrderedDict()):
                print(block_zone[row_name])
            else:
                for item in block_zone[row_name]:
                    print(block_zone[row_name][item], end = ' ')
                print('\t', end = '')
                for j, item in enumerate(block_zone[row_name]):
                    if i == 0 and j == 0:
                        print(item, end = ' : 1 = disk, 2 = tapered-edge disk, 3 = envelope, 4 = debris disk, 5 = wall')
                    else:
                        print(item, end = ', ')
                print('')
        print('')
        
def print10_Grain_properties(sample_para_dict):
    block_name = '#Grain properties'
    print(block_name)
    for zone_id in range(sample_para_dict['#Number of zones']):
        block_zone = sample_para_dict[block_name]['zone' + str(zone_id)]
        for category in block_zone:
            if type(block_zone[category]) != type(collections.OrderedDict()):
                print(block_zone[category], '\t Number of species')
            else:
                for row_name in block_zone[category]:
                    if type(block_zone[category][row_name]) != type(collections.OrderedDict()):
                        print(block_zone[category][row_name])
                    else:
                        for item in block_zone[category][row_name]:
                            print(block_zone[category][row_name][item], end = ' ')
                        print('\t', end = '')
                        for item in block_zone[category][row_name]:
                            print(item, end = ', ')
                        print('')
                print('')
                
def print11_Molecular_RT_settings(sample_para_dict):
    block_name = '#Molecular RT settings'
    print(block_name)
    for row_name in sample_para_dict[block_name]:
        if type(sample_para_dict[block_name][row_name]) != type(collections.OrderedDict()):
            print(sample_para_dict[block_name][row_name])#, '\t', '1 = cylindrical, 2 = spherical, 3 = Voronoi tesselation (this is in beta, please ask Christophe)')
        else:
            for item in sample_para_dict[block_name][row_name]:
                print(sample_para_dict[block_name][row_name][item], end = ' ')
            print('\t', end = '')
            for item in sample_para_dict[block_name][row_name]:
                print(item, end = ', ')
            print('')
    print()  
    
    
def print12_Star_properties(sample_para_dict):
    block_name = '#Star properties'
    print(block_name)
    print(sample_para_dict[block_name]['Number of stars'], '\tNumber of stars')
    for star_id in range(sample_para_dict[block_name]['Number of stars']):
        block_star = sample_para_dict[block_name]['star' + str(star_id)]
        for row_name in block_star:
            if type(block_star[row_name]) != type(collections.OrderedDict()):
                print(block_star[row_name])
            else:
                for item in block_star[row_name]:
                    print(block_star[row_name][item], end = ' ')
                print('\t', end = '')
                for item in block_star[row_name]:
                    print(item, end = ', ')
                print('')
        print('')

def display_file(para_dict, save_path = None):
    """Display the parameter file.
        Input:
            para_dict: a collections.OrderedDict type data structure, please generate it with 'generateMcfostTemplate()' function;
            save_path: a string, if None (default), then display only on the screen,
                                 if it is indeed a string, then the file will be saved to that location.
        Output:
            Either a display on the screen (when save_path is None)
            Or save to a file (when save_path is an address, e.g., './template_mcfost_para.para')
    """
    if save_path is not None:
        import sys
        sys.stdout = open(save_path, "w+")
    print0_mcfost_version(para_dict)
    print1_Number_of_photon_packages(para_dict)
    print2_Wavelength(para_dict)
    print3_Grid_geometry_and_size(para_dict)
    print4_Maps(para_dict)
    print5_Scattering_method(para_dict)
    print6_Symmetries(para_dict)
    print7_Disk_physics(para_dict)
    print8_Number_of_zones(para_dict)
    print9_Density_structure(para_dict)
    print10_Grain_properties(para_dict)
    print11_Molecular_RT_settings(para_dict)
    print12_Star_properties(para_dict)

# n_zone = 2
# n_species = [1, 1]#[3, 2]
# n_star = 1#2
# sample_para_dict = generateMcfostTemplate(n_zone = n_zone, n_species = n_species, n_star = n_star)
# display_file(sample_para_dict, save_path='./test.para')
