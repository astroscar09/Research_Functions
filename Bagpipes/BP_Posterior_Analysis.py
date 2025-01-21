import bagpipes as pipes
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from Bagpipes_Fitting_w_Spectra import *
import sys
import matplotlib.pyplot as plt
import time



def min_chi_sq_index(fit): 

    chisq = fit.posterior.samples['chisq_phot']
    
    idx_min_chisq = np.argmin(chisq)

    amin_chi2 = np.amin(chisq)

    return (amin_chi2, idx_min_chisq)

def get_advanced_quantities(fit):

    fit.posterior.get_advanced_quantities()

def get_posterior_samples(fit):
    
    return fit.posterior.samples

def get_ages(fit):
    
    ages = fit.posterior.sfh.ages
    
    return ages

def get_sfhs(fit):
    sfh = fit.posterior.samples['sfh']
    
    return sfh

def subselect_age_range(ages, start, end):
    
    idxs = np.where((start <= ages) & (ages <= end))[0]
    
    return idxs

def grab_avg_sfh_per_time(fit, start, end, avg_type = 'mean'):
    
    ages = get_ages(fit)
    sfh = get_sfhs(fit)
    idxs = subselect_age_range(ages, start, end)
    
    in_range = sfh[:, idxs]
    
    if avg_type == 'mean':
        
        avg_sfr = np.mean(in_range, axis = 1)
        l16, med, u84 = np.percentile(in_range, axis = 1, q = (16, 50, 84))
    
    elif avg_type == 'median':
        
        avg_sfr = np.mean(in_range, axis = 1)
        l16, med, u84 = np.percentile(in_range, axis = 1, q = (16, 50, 84))
        
    return avg_sfr, l16, med, u84


def get_burstiness_metric(fit):
    
    avg_sfr_10myr, l16_10myr, med_10myr, u84_10myr = grab_avg_sfh_per_time(fit = fit, start = 0, end = 10e6)
    avg_sfr_100myr, l16_100myr, med_100myr, u84_100myr  = grab_avg_sfh_per_time(fit = fit, start = 0, end = 100e6, 
                                                                             avg_type = 'median')

    burstiness = np.log10(avg_sfr_10myr/avg_sfr_100myr)
    
    return burstiness


def get_posterior_samples(fit):
    
    return fit.posterior.samples
    

def get_post_samples(fit):

    '''
    This function takes a fit objects and returns the 2d array of posterior
    values explored with each column representing a free parameter in the BP 
    model.
    
    Inputs:
    fit: BP fit object
    
    Returns:
    samples2d: A 2d array of all the posterior values explored with each column being
               a BP parameter model
    
    
    '''
    
    #this returns all the values in the chain with columns 1-3 being stuff for the spectra correction
    #and we do not use those in the model generation so gonna remove them
    samples2d = fit.results['samples2d']

    return samples2d



def get_redshift_fit(fit):
    '''
    This function gets the redshift for a given bagpipes fit object
    
    Inputs:
    fit: bagpipes fit object
    
    Returns:
    z: the redshift found by the median if it is a free redshift model or 
       inputted as part of the fit instructions for fixed-z models
    
    '''
    try:
        z = np.median(fit.posterior.samples['redshift'])

        return z

    except:

        z = fit.fit_instructions['redshift']

        return z

def min_chi_sq_index(fit): 

    #fit.posterior.get_advanced_quantities()

    chisq = fit.posterior.samples['chisq_phot']
    l16_chi2, med_chi2, u84_chi2 = np.percentile(chisq, q = (16, 50, 84))

    idx_min_chisq = np.argmin(chisq)

    min_chisqr = chisq[idx_min_chisq]

    return min_chisqr, idx_min_chisq, l16_chi2, med_chi2, u84_chi2 

def get_ID_fit(fit):

    ID = fit.galaxy.ID

    return ID

def mass_estimate_distribution(fit, end_time=10e6, start_time = 0):
    
    '''
    Computes the Total Stellar Mass Between a start and end time where start of 0 is where the source is
    
    
    '''

    ages = fit.posterior.sfh.ages
    sfh = fit.posterior.samples['sfh']

    mass_vs_time_arr =  fit.posterior.sfh.age_widths * sfh

    mass_within_time = sfh[:, (start_time < ages) & (ages < end_time)]
    total_mass_within_time = np.sum(mass_within_time, axis = 1)

    fractional_mass_within_time = total_mass_within_time/np.sum(mass_vs_time_arr, axis = 1)

    return total_mass_within_time, fractional_mass_within_time

def compute_percentiles(data, q = (16, 50, 84)):
    
    return np.percentile(data, q = q)

def get_fit_instructions(fit):
    
    fit_instruction = fit.fit_instructions
    
    return fit_instruction

def params_in_fit_instructions(fit):
    
    fit_instruction  = get_fit_instructions(fit)
    
    params = list(fit_instruction.keys())
    
    return params

def make_galaxy_property_df(fit):
    
    galaxy_prop_dict = {}
    
    posterior_dict = fit.posterior.samples
    
    for key, val in posterior_dict.items():
        if len(val.shape) == 1:
            galaxy_prop_dict[key] = val
            
    DF = pd.DataFrame(galaxy_prop_dict)
    
    return DF

'''
['burst', 'ID', 'delayed:age', 'delayed:massformed', 'delayed:metallicity',
       'delayed:tau', 'dust:Av', 'nebular:logU', 'redshift', 'stellar_mass',
       'formed_mass', 'sfr', 'ssfr', 'nsfr', 'mass_weighted_age', 'tform',
       'tquench', 'mass_weighted_zmet', 'chisq_phot']
'''

def make_median_galaxy_property_df(fit, ID):
    
    galaxy_prop_list = [ID]
    
    posterior_dict = fit.posterior.samples
    
    for key, val in posterior_dict.items():
        if len(val.shape) == 1:
            galaxy_prop_list.append(np.median(val))

    return galaxy_prop_list


def save_median_to_file(fit, ID):
    
    medians = make_median_galaxy_property_df(fit, ID)
    burst = get_burstiness_metric(fit)
    
    med_burst = np.median(burst)
    
    string = f'{med_burst:.2f} '
    
    for vals in medians:
        string += f'{vals:.5f} '
    
    string += '\n'
    
    file = open(f'Median_Vals/{ID}_Median_Params.txt', 'w+')
    file.write(string)
    file.close()
       
def make_starting_dictionary(fit):
    
    params = params_in_fit_instructions(fit)
    
    if 'continuity' in params:
        
        model_comp = {}
        
        dust = {}
        dust['type'] = 'Calzetti'
        
        continuity = {}
        
        nebular = {}
        
        model_comp['continuity'] = continuity
        model_comp['continuity']['bin_edges'] = fit.fit_instructions['continuity']['bin_edges']
        model_comp['dust'] = dust
        model_comp['nebular'] = nebular
        
        return model_comp
        
    elif 'delayed' in params:
        
        model_comp = {}
        
        dust = {}
        dust['type'] = 'Calzetti'
        
        delayed = {}
        
        nebular = {}
        
        model_comp['delayed'] = delayed
        model_comp['dust'] = dust
        model_comp['nebular'] = nebular
        
        return model_comp
    
    else:
        print('No Valid Model Given. Returning None')
        
        
def make_BP_DF_for_posteriors(fit):
    
    params = params_in_fit_instructions(fit)
    start_dict = make_starting_dictionary(fit)
    
    df_dict = {}
    
    for key,val in fit.posterior.samples.items():

        check = key.split(':')[0]
    
        if check in params:

            check2 = key.split(':')[-1]
            df_dict[key] = val
        
    DF = pd.DataFrame(df_dict)

    return DF

def free_z_continuum_estimates_samples(fit, diagnose = False):
    
    '''
    This function calculates a series of continuum estimate near lyman alpha from the 
    posterior fit objects
    to be used for EW calculations. This uses the bagpipes update feature as the redshift 
    is fixed and so only the 
    galaxy property values will need to be updated to generate a new spectra.
    
    -------------------------
    Input: 
    fit: This is a Bagpipes fit object
    
    -------------------------
    
    Output: 
    Continuum (list): This is a list of continuum estimates near the lyman alpha line
                      with each entry being a different set of param values explored
            
    
    '''
    
    #gets the redshift and fixes it to z
    z = get_redshift_fit(fit)

    #gets the 2d array of parameter values BP explored this should be 1000 x num_of_params
    params_dict = get_posterior_samples(fit)


    #this is the parameter values which match up with the columns of the params2d 2D-array
    #     params = ['delayed:age', 'delayed:massformed',
    #               'delayed:metallicity', 'delayed:tau', 'dust:Av', 
    #               'nebular:logU']

    start_dict = make_starting_dictionary(fit)
    DF = make_BP_DF_for_posteriors(fit)
        #z = get_redshift_fit(fit)
    
    start_dict['redshift'] = z
    
    if diagnose:
        
        print(f'The redshift of this source is: {z:.2f}')
        print('The dictionary we are using to start the model is: ')
        print(start_dict)
        print()
        print('The DataFrame that has the model values is: ')
        print(DF)
        
    
     #list to hold the continuum values for each of the BP realizations
    Continuum = []
    
    for i in range(DF.shape[0]):

        if i == 0:

            for key in DF.columns:
                if key == 'redshift':
                    val = DF.loc[i, key]
                    start_dict[key] = val
                else:
                    val = DF.loc[i, key]
                    check = key.split(':')[0]
                    check2 = key.split(':')[-1]
                    start_dict[check][check2] = val
                

            #specifying offset from Lyman-Alpha in angstroms, this is in Observed Frame
            offset = 30 * (1+z)
            lyman_next = 1230 * (1+z)

            #this spectum is just to the red of lyman alpha to get an EW estimate and so this is 
            #making a spectrum between 1230 - 1260 angstrom, #note that lya is at 1216 
            #may need to tweak the lower bound if I see lyman alpha emission in some of them
            spec_wave_min = lyman_next 
            spec_wave_max = spec_wave_min + offset

            #making a wavelength array near the lyman alpha line
            wavelength = np.linspace(spec_wave_min, spec_wave_max + 1, 100)
            
            if diagnose:
                
                print('In the dictionary population we have the following Bagpipes Model')
                print(start_dict)
                print('We are looking at wavelengths')
                print(wavelength)

            #Makes a bagpipes model with the model components above 
            #and has the wavelengths next to lyman alpha
            galaxy_model = pipes.model_galaxy(start_dict, filters, spec_wavs=wavelength)

            #the wavelength values are stored in the first column 
            wave = galaxy_model.spectrum[:, 0]

            #flux values are stored in the second column hence the 0 and 1 above and below
            flux = galaxy_model.spectrum[:, 1]
            
            if diagnose:
                
                fig, ax = plt.subplots(figsize = (6, 3), constrained_layout = True)
                ax.step(wavelength, flux, where = 'mid', color = 'black')
                ax.set_xlabel('Wavelength')
                ax.set_ylabel('Flux')
                ax.set_title('BP Model Generated Spectrum')
                fig.savefig(f'BP_Spectra_i_{i}.png', dpi = 150)
                plt.close(fig)

            #this is what will be in the numerator of the integrand
            num_integrand = wave * flux

            #integrating here using the trapezoid method
            num = trapezoid(num_integrand, x = wave, dx = .1)
            denom = trapezoid(wavelength,x = wave, dx = .1)

            #getting the continuum values
            avg_cont_flux = num/denom

            #storing the continuum value
            Continuum.append(avg_cont_flux)

        else:

            for key in DF.columns:

                if key == 'redshift':
                    val = DF.loc[i, key]
                    start_dict[key] = val
                else:
                    val = DF.loc[i, key]
                    check = key.split(':')[0]
                    check2 = key.split(':')[-1]
                    start_dict[check][check2] = val
                
            if diagnose & (i%100 == 0):
                
                print('Updated BP Model dictionary is:')
                print(start_dict)
                print()

            galaxy_model.update(start_dict)
            
            if diagnose & (i%100 == 0):
                
                fig, ax = plt.subplots(figsize = (6, 3), constrained_layout = True)
                ax.step(wavelength, flux, where = 'mid', color = 'black')
                ax.set_xlabel('Wavelength')
                ax.set_ylabel('Flux')
                ax.set_title('BP Model Generated Spectrum')
                fig.savefig(f'BP_Spectra_i_{i}.png', dpi = 150)
                plt.close(fig)

            #gets the new spectrum
            wave = galaxy_model.spectrum[:, 0]
            flux = galaxy_model.spectrum[:, 1]

            #gets the integrand for the Continuum estimate
            num_integrand = wave * flux

            num = trapezoid(num_integrand, x = wave, dx = .1)
            denom = trapezoid(wavelength, x = wave, dx = .1)

            avg_cont_flux = num/denom
            Continuum.append(avg_cont_flux)

    return Continuum

def fixed_z_continuum_estimates(fit):
    
    '''
    This function calculates a series of continuum estimate near lyman alpha from the 
    posterior fit objects
    to be used for EW calculations. This uses the bagpipes update feature as the redshift 
    is fixed and so only the 
    galaxy property values will need to be updated to generate a new spectra.
    
    -------------------------
    Input: 
    fit: This is a Bagpipes fit object
    
    -------------------------
    
    Output: 
    Continuum (list): This is a list of continuum estimates near the lyman alpha line
                      with each entry being a different set of param values explored
            
    
    '''
    
    #gets the redshift and fixes it to z
    z = get_redshift(fit)
    
    #gets the 2d array of parameter values BP explored this should be 1000 x num_of_params
    params2d = get_post_samples(fit)
    
    
    #this is the parameter values which match up with the columns of the params2d 2D-array
    params = ['delayed:age', 'delayed:massformed',
              'delayed:metallicity', 'delayed:tau', 'dust:Av', 
              'nebular:logU']
    
    #making the model dictionaries
    model_comp = {}
    dust = {}
    dust['type'] = 'Calzetti'
    delayed = {}
    nebular = {}

    #a counter variable to only generate the BP model once when counter equals 0
    counter = 0
    
     #list to hold the continuum values for each of the BP realizations
    Continuum = []
        
    #looping over all the rows in the params2d array, which may be larger than 1000 from posteriors
    for row in params2d:

        if counter == 0:
            
            #this takes the row and param names and pairs them with 
            #key = param_name and val = parameter value
            for key, val in zip(params, row):
                
                #splits up the name if possible
                check = key.split(':')
                
                #if first entry is delayed we take the name and value and 
                #store that into the delayed dictionary
                
                if check[0] == 'delayed':
                    delayed[check[-1]] = val
                    
                #if first entry is dust we take the name and value and 
                #store that into the dust dictionary
                
                elif check[0] == 'dust':
                    dust[check[-1]] = val
                
                #if first entry is nebular we take the name and value and 
                #store that into the nebular dictionary
                elif check[0] == 'nebular':
                    
                    if np.isnan(logU):
                    
                        nebular[check[-1]] = val
                        
                    else:
                        nebular[check[-1]] = logU
                
                #all other entries are part of model component and so we put them there
                else:
                    model_comp[key] = val

            model_comp['delayed'] = delayed
            model_comp['dust'] = dust
            model_comp['nebular'] = nebular
            model_comp['redshift'] = z
            
            #specifying offset from lyman alpha in angstroms, this is in Observed Frame
            offset = 30 * (1+z)
            lyman_next = 1225 * (1+z)

            #this spectum is just to the red of lyman alpha to get an EW estimate and so this is 
            #making a spectrum between 1225 - 1255 angstrom, #note that lya is at 1216 
            #may need to tweak the lower bound if I see lyman alpha emission in some of them
            spec_wave_min = lyman_next 
            spec_wave_max = spec_wave_min + offset

            #making a wavelength array near the lyman alpha line
            wavelength = np.linspace(spec_wave_min, spec_wave_max + 1, 100)

            #print('Model Components')
            #print(model_comp)
            
            #print()
            print(model_comp)
            #Makes a bagpipes model with the model components above 
            #and has the wavelengths next to lyman alpha
            galaxy_model = pipes.model_galaxy(model_comp, filters, spec_wavs=wavelength)
            
            #updates counter so that the next loop it does not generate a new model but updates it
            counter += 1

            #need to get continuum by integrating lambda*F_lambda*T_lambda but 
            #T_Lambda is 1 in this range
            
            #the wavelength values are stored in the first column 
            wavelength = galaxy_model.spectrum[:, 0]
            
            #flux values are stored in the second column hence the 0 and 1 above and below
            flux = galaxy_model.spectrum[:, 1]

            #this is what will be in the numerator of the integrand
            num_integrand = wavelength * flux
#             print('Flux')
#             print(flux)
#             print()
#             print('Wavelengths')
#             print(wavelength)
#             print()
#             print('----------------------------')
#             print()
            
            #print(num_integrand)
            
            #Code for debugging purposes
            #uncomment only when debugging
            #plt.plot(wavelength, fnu_)
            #plt.show()

            #integrating here using the trapezoid method
            num = trapezoid(num_integrand, x = wavelength, dx = .1)
            denom = trapezoid(wavelength,x = wavelength, dx = .1)

            #getting the continuum values
            avg_cont_flux = num/denom
            
            #storing the continuum value
            Continuum.append(avg_cont_flux)
        
        else:
            
            #updates the model parameters by using the same key but changing the value
            for key, val in zip(params, row):
                
                check = key.split(':')

                #print(check)
                if check[0] == 'delayed':
                    delayed[check[-1]] = val
                    
                elif check[0] == 'dust':
                    dust[check[-1]] = val
                    
                elif check[0] == 'nebular':
                    
                    if np.isnan(logU):
                    
                        nebular[check[-1]] = val
                        
                    else:
                        nebular[check[-1]] = logU
                    #nebular[check[-1]] = val
                    
                else:
                    model_comp[key] = val
            #note we do not update redshift as that is a fixed parameter
            model_comp['delayed'] = delayed
            model_comp['dust'] = dust
            model_comp['nebular'] = nebular
            #print(model_comp)
            #BP function that updates the model with the new values
            galaxy_model.update(model_comp)

            #gets the new spectrum
            wavelength = galaxy_model.spectrum[:, 0]
            flux = galaxy_model.spectrum[:, 1]
            
            #gets the integrand for the Continuum estimate
            num_integrand = wavelength * flux
          
            num = trapezoid(num_integrand, x = wavelength, dx = .1)
            denom = trapezoid(wavelength, x = wavelength, dx = .1)

            avg_cont_flux = num/denom
            Continuum.append(avg_cont_flux)
            
    return Continuum

def Continuum_Estimates(fit):
    
    '''
    This function takes in a fit object and gets the continuum for both fixed-z or 
    free-z 
    '''
    
    keys = fit.posterior.samples.keys()
    
    if 'redshift' in keys:
        
        Continuum = free_z_continuum_estimates_samples(fit)
    
        return np.array(Continuum)
    
    else:
        
        Continuum = fixed_z_continuum_estimates(fit)
    
        return np.array(Continuum)
    
    
def save_median_to_file(fit, ID):
    
    medians = make_median_galaxy_property_df(fit, ID)
    burst = get_burstiness_metric(fit)
    
    med_burst = np.median(burst)
    
    string = f'{med_burst:.2f} '
    
    for vals in medians:
        string += f'{vals:.5f} '
        
    cont = Continuum_Estimates(fit)
    
    med_cont = np.median(cont)
    
    string += f'{med_cont:.4e} '
    
    
    string += '\n'
    
    file = open(f'Median_Vals/{ID}_Median_Params.txt', 'w+')
    file.write(string)
    file.close()
    
if __name__ == "__main__":
    
    start = time.time()
    
    print('Grabbing Bagpipes Run Name')
    run = get_run_name()
    print(f'Run-Name: {run} Acquired')

    print('Attempting to read in Bagpipes Photometric Catalog')
    Bagpipes_Phot_DF = read_input_phot_cat()
    print('Read in Bagpipes Catalog')

    ID = get_ID()
    
    z_tesla = get_redshift(Bagpipes_Phot_DF, ID)
    
    try:
        
        fit = fit_BP(ID, filters, load_phot, z_tesla, run, only_fit = False, model = 'delayed_tau')
        get_advanced_quantities(fit)
        #print(f'Successfully fitted ID: {ID}')
        #print()
    
    except:
        print('ERROR IN FITTING!!!')
        print(f'Check on ID: {ID}')
        print()
        
        
    #burst = get_burstiness_metric(fit)
    save_median_to_file(fit, ID)
        
    end = time.time()
    
    print(f'Total time elapsed is: {end - start:.3f} s')
    #print('Saving Continuum values')
    #np.savetxt(f'{FOLDER}/{ID}.txt', galaxy_continuum)
    #print('Continuum Values Saved')