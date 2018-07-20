import mcfostParameterTemplate

n_zone = 1
n_species = [3]
n_star = 1
sampleTemplate = mcfostParameterTemplate.generateMcfostTemplate(n_zone = n_zone, n_species = n_species, n_star = n_star)

sampleTemplate['#Density structure']['zone0']['row0']['zone type'] = 3

mcfostParameterTemplate.display_file(sampleTemplate, save_path='/Users/echoquet/Desktop/test.para')

print("yeah!")
