# DebrisDiskFM
Forward modeling for circumstellar debris disks in scattered light.

Method: Monte Carlo Markov Chain (MCMC) with the MCFOST disk modeling software.

## 0. Installation
```pip install --user -e git+https://github.com/seawander/DebrisDiskFM.git#egg=Package```

The above command does not require administrator access, and can be run both on one's personal desktop and on a computer cluster.

## 1. Utilization
### 1.1 Parameter File Setup
#### 1.1.1 MCFOST Parameter File Template
For a given sytem, generate a disk template with the following command
```python
sampleTemplate = mcfostParameterTemplate.generateMcfostTemplate(n_zone = n_zone, n_species = n_species, n_star = n_star)
```
The above command generate a template in the ```collections.OrderedDict()``` structure, for this template, there are ```n_zone``` zones; ```n_species``` species (note: ```n_species``` is a list. For example, when ```n_zone = 2```, we can let ```n_species = [3, 2]```, then in the 0th zone, i.e., ```zone0```, there are 3 species of grains, and in the 1st zone, i.e., ```zone1```, there are 2 species; and ```n_star``` stars shining the whole system.
#### 1.1.2 MCFOST Parameter File Sample for a Specific Target
From the previous paragraph, we have a sample template, then we should modify the parameters in the template for our specific target.

The structure of the template is in this mind-map-structured PDF file: [MCFOST Parameter OrderedDict.pdf](https://github.com/seawander/DebrisDiskFM/blob/master/MCFOST%20Parameter%20OrderedDict.pdf). In this PDF file, the **quoted** parameters are what you can modify, and all of the parameters have the ***same names*** as the MCFOST parameter file. The *only* added ones are the row numbers in each block (named as ```'rowW'``` where W = 0 to the number of rows - 1 in that block), and ```'zoneX'``` where X = 0 to (n_zone - 1), ```'speciesY'``` where Y = 0 to (n_species[X] - 1), and ```'starZ'``` where Z = 0 to (n_star - 1).

For example, if you want to turn on the Stokes maps, use 
```python
sampleTemplate['#Wavelength']['row3']['stokes parameters?'] = 'T'
```

or if you want to change the input file for optical indices to 'ice_opct.dat' for the 2nd species in zone0, use

```python
sampleTemplate['#Grain properties']['zone0']['species2']['row1']['Optical indices file'] = 'ice_opct.dat'
```

for the detailed parameters that can be changed, refer to the PDF file.

#### 1.1.3 Save parameter file
Just call 
```python
save_path = None
mcfostParameterTemplate.display_file(sampleTemplate, save_path)
```
and it will save the parameter structure in a proper MCFOST parameter file format to ```save_path```, if ```None``` then it will display to the screen only; or save to the address if it is not ```None```.
