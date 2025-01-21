from astropy.table import Table
import numpy as np
from astropy import constants as const
from astropy import units as u
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import bagpipes as pipes
import pandas as pd
import seaborn as sb
import sys
from BP_PreLoad import GetIndex
from Bagpipes_Fitting_w_Spectra import *

plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['font.family'] = 'serif'
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13


xlabels = {'delayed:age': 'Age [Gyr]', 
           'delayed:massformed': r'log(M/M$_{\odot}$)', 
           'delayed:metallicity': r'Z/Z$_{\odot}$', 
           'delayed:t_bc': 'Time of Birth Cloud', 
           'delayed:tau': r'$\tau$ [Gyr]', 
           'dust:Av': r'A$_V$ [Mag]', 
           'nebular:logU': 'log$_{10}$(U)', 
           'stellar_mass': r'log$_{10}$(M$_*$/M$_{\odot}$)', 
           'formed_mass': r'log$_{10}$(M/M$_{\odot}$)', 
           'sfr': r'SFR $M_{\odot}$yr$^{-1}$', 
           'ssfr': r'log(sSFR) [yr$^{-1}$]', 
           'nsfr': 'Normalized SFR', 
           'mass_weighted_age': 'Mass Weighted Age [Gyr]', 
           'tform': 'Time of Formation [Gyr]', 
           'tquench': 'Quench Time [Gyr]', 
           'sfh': 'SFH [$M_{\odot}$yr$^{-1}$]', 
           'chisq_phot': r'$\xi^2$'}

def get_advanced_quantities(fit):

    fit.posterior.get_advanced_quantities()

def galaxy_ID(fit):

    return fit.galaxy.ID

def posterior_specs(fit):

    fit.posterior.get_advanced_quantities()
    spec_2d = fit.posterior.samples['spectrum_full']

    return spec_2d

def getting_spectrum_bounds(fit):

    spec_2d = posterior_specs(fit)

    one_sig = 16
    median = 50
    one_sig_high = 84

    low_sig_spec, median_spec, high_sig_spec = np.percentile(spec_2d, 
                                                             q = (one_sig, median, one_sig_high),
                                                             axis = 0)

    return (low_sig_spec, median_spec, high_sig_spec)


def plot_SFH_post(fit, ax = None, logy = False, logx = False, xlim = (-1, 2000), min_chisqr = False):
    
    SFH = fit.posterior.samples['sfh']
    
    l16_sfh, median_sfh, u84_sfh = np.percentile( SFH, 
                                                  axis = 0, 
                                                  q = (16, 50, 84))
    ages = fit.posterior.sfh.ages

    
    if ax == None:
    
        fig = plt.figure(figsize = (10, 5), constrained_layout = True)
        Myr = 1e6
        #plotting the input spectrum
        plt.fill_between(ages/Myr, 
                         y1 = u84_sfh , 
                         y2 = l16_sfh, 
                         color = 'dodgerblue')
        
        plt.plot(ages/Myr, median_sfh, color = 'black', label = 'Median SFH')
        
        if min_chisqr:
            
            chisqr = fit.posterior.samples['chisq_phot']
        
            min_chiqsr_idx = np.argmin(chisqr)
        
            plt.plot(ages/Myr, SFH[min_chiqsr_idx], color = 'red', label = r'Minimum $\chi^2$ SFH')

        plt.ylabel(r'SFR [M$_{\odot}$yr$^{-1}$]', fontsize = 15)
        plt.xlabel(r'Lookback Time [Myr]', fontsize = 15)
        plt.minorticks_on()
        
        if logy:
            plt.yscale('log')
            
        if logx:
            plt.xscale('log')
            
        plt.xlim(xlim)
        plt.legend()

        return fig
    
    else:
        Myr = 1e6
        #plotting the SFH posterior
        ax.fill_between(ages/Myr , 
                         y1 = u84_sfh , 
                         y2 = l16_sfh, 
                         color = 'dodgerblue')
        
        ax.plot(ages/Myr, median_sfh, color = 'black', label = 'Median SFH')
        if min_chisqr:
            
            chisqr = fit.posterior.samples['chisq_phot']
        
            min_chiqsr_idx = np.argmin(chisqr)
        
            ax.plot(ages/Myr, SFH[min_chiqsr_idx], color = 'red', label = r'Minimum $\chi^2$ SFH')

        ax.set_ylabel(r'SFR [M$_{\odot}$yr$^{-1}$]', fontsize = 15)
        ax.set_xlabel(r'Lookback Time [Myr]', fontsize = 15)
        ax.set_xlim(xlim)
        
        if logy:
            ax.set_yscale('log')
            
        if logx:
            ax.set_xscale('log')
        
        ax.minorticks_on()
        ax.legend()

        return ax

def getting_photometry(fit):

    phot_2d = fit.posterior.samples['photometry']

    return phot_2d

def get_phot_bounds(fit):

    phot_2d = getting_photometry(fit)

    one_sig = 16
    median = 50
    one_sig_high = 84

    low_sig_phot, median_phot, high_sig_phot = np.percentile(phot_2d, 
                                                             q = (one_sig, median, one_sig_high), 
                                                             axis = 0)

    return (low_sig_phot, median_phot, high_sig_phot)

def min_chi_sq_index(fit): 

    chisq = fit.posterior.samples['chisq_phot']
    
    idx_min_chisq = np.argmin(chisq)

    amin_chi2 = np.amin(chisq)

    return (amin_chi2, idx_min_chisq)

def phot_wavelength(fit):

    eff_wave = fit.posterior.galaxy.filter_set.eff_wavs

    return eff_wave

def spec_wavelength(fit):

    spec_wave = fit.posterior.model_galaxy.wavelengths

    return spec_wave

def get_redshift_fit(fit):

    try:
        z = np.median(fit.posterior.samples['redshift'])
        return z
    
    except:
        z = fit.fit_instructions['redshift']
        return z

def F_nu_Conversion(model_spectrum, wavelength):

    c_light = const.c.to(u.AA/u.second).value

    conversion_f_nu = wavelength**2/c_light

    f_nu_spectrum = model_spectrum * conversion_f_nu

    return f_nu_spectrum

def Spectra_Plot_Setup(fit):
    
    #gets the eff wavelengths for the photometric filters
    phot_waves = phot_wavelength(fit)
    
    #gets the wavelengths for the spectra
    spec_wave = spec_wavelength(fit)
    
    #gets the best fit redshift or input redshift for fixed-z model
    z = get_redshift_fit(fit)
    
    #gets the spectra and returns back the 16th 50th and 84th percentile
    low_spec, median_spec, high_spec = getting_spectrum_bounds(fit)
    
    #redshifts the rest frame wavelength to the observed frame
    redshifted_wavelengths = spec_wave * (1+z)
    
    #converts the spectra from F_lambda to F_nu
    fnu_upper_spec = F_nu_Conversion(high_spec, redshifted_wavelengths)
    med_fnu_spec = F_nu_Conversion(median_spec, redshifted_wavelengths)
    fnu_lower_spec = F_nu_Conversion(low_spec, redshifted_wavelengths)
    
    #wavelength to the left of the minimum photometric wavelength available
    left_wave = 500
    min_wave = np.amin(phot_waves) - left_wave
    
    #wavelength rightward of the maximium photmoetric wavelength available
    right_wave = 500
    max_wave = np.amax(phot_waves) + right_wave

    #applying a selection cut to only include wavelengths in min_wave-max_wave
    filt = (redshifted_wavelengths > min_wave) & (redshifted_wavelengths < max_wave)
    
    #applying the filter to spectra and wavelength arrays
    
    filtered_lower_spec = fnu_lower_spec[filt]
    filtered_median_spec = med_fnu_spec[filt]
    filtered_upper_spec = fnu_upper_spec[filt]
    
    filtered_wavelengths = redshifted_wavelengths[filt]
    
    return (filtered_wavelengths, filtered_lower_spec, filtered_median_spec, filtered_upper_spec)
    
def Phot_Plot_Setup(fit):    
    
    phot_waves = phot_wavelength(fit)
    low_phot, median_phot, high_phot = get_phot_bounds(fit)
    
    fnu_phot_up = F_nu_Conversion(high_phot, phot_waves)
    med_fnu_phot = F_nu_Conversion(median_phot, phot_waves)
    fnu_phot_lower = F_nu_Conversion(low_phot, phot_waves)
    
    return (phot_waves, fnu_phot_lower, med_fnu_phot, fnu_phot_up)

def Best_Fit_Model_Spec(fit):
    
    #gets the photometric wavelengths
    phot_waves = phot_wavelength(fit)
    
    
    #gets the 2d array of posterior spec, should be 1000 x num of entries in wavelength array
    specs_2d = posterior_specs(fit)
    
    
    #gets the wavelengths for the spectra
    spec_wave = spec_wavelength(fit)
    
    #gets the best fit redshift or input redshift for fixed-z model
    z = get_redshift_fit(fit)
    
    #redshifts the rest frame wavelength to the observed frame
    redshifted_wavelengths = spec_wave * (1+z)
    
    #gets the best fit model using minimum chi2
    min_chisqr, idx_min_chisq = min_chi_sq_index(fit)
    
    #trimming the wavelengths to include just near the end of my photometric data
    left_wave = 500
    min_wave = np.amin(phot_waves) - left_wave
    
    #wavelength rightward of the maximium photmoetric wavelength available
    right_wave = 500
    max_wave = np.amax(phot_waves) + right_wave

    #applying a selection cut to only include wavelengths in min_wave-max_wave
    #filt = (redshifted_wavelengths > min_wave) & (redshifted_wavelengths < max_wave)
    
    #gets the best fit sppectra from the 2d spectra
    best_fit_spec = specs_2d[idx_min_chisq, :]
    
    #converts it to F_nu units
    fnu_best_spec = F_nu_Conversion(best_fit_spec, redshifted_wavelengths)
    
    #applies filtered wavelenths to the spec and wavelength
    filtered_wavelengths = redshifted_wavelengths#[filt]
    filtered_spec = fnu_best_spec#[filt]
    
    return filtered_wavelengths, filtered_spec

def Spec_Phot_Plots(fit, ax, colors, labels = 'Lya', min_chisqr = False):
    
    if min_chisqr:
        
        #just making sure that we get the advanced quantities of the fit 
        #need this to get the chisqr info
        get_advanced_quantities(fit)

        #we then pass the fit object into the following function to get the min chi2
        min_chisqr = np.amin(fit.posterior.samples['chisq_phot'])
        
              
        #getting the best fit model spec and wavelengths   
        filtered_wavelength, min_chisqr_spec = Best_Fit_Model_Spec(fit)

        #getting the BP photometry
        phot_waves, fnu_phot_lower, med_fnu_phot, fnu_phot_up = Phot_Plot_Setup(fit)

        #rounding the chi2 value
        min_chsq = round(min_chisqr, 2)

        #computing upper and lower photometric errors for the errorbars
        phot_err_upper = fnu_phot_up - med_fnu_phot
        phot_err_lower = med_fnu_phot - fnu_phot_lower
        
        spec_props = {'color':colors, 'label': fr'{labels} $\chi^2 = $ {min_chsq}' }
        
        ax.plot(filtered_wavelength, min_chisqr_spec, **spec_props)

        ax.plot(phot_waves, med_fnu_phot, '.', color = colors)
        
        errorbar_props = {'yerr':[phot_err_upper, phot_err_lower], 'elinewidth':20, 
                         'fmt': 'none', 'color': colors, 'alpha': .5}
        
        ax.errorbar(phot_waves, med_fnu_phot, **errorbar_props)

        ax.set_xscale('log')
        ax.set_yscale('log')
        
        ID = galaxy_ID(fit)
        ax.set_title(f'Object: {ID}')
        
        phot_waves = phot_wavelength(fit)
        #wavelength to the left of the minimum photometric wavelength available
        left_wave = 1000
        min_wave = np.amin(phot_waves) - left_wave

        #wavelength rightward of the maximium photmoetric wavelength available
        right_wave = 1000
        max_wave = np.amax(phot_waves) + right_wave
        
        ax.set_xlim(min_wave, max_wave)
        
   
        #getting xticks
        adding_factor = len(filtered_wavelength)//10
        
        indices = np.arange(0, 10) * adding_factor
        
        xticks = filtered_wavelength[indices]
        
        ax.set_xticks(xticks)
        ax.set_xticklabels(np.round(xticks))
        #ax.ticklabel_format(scilimits = (-4, 4))
        
        ax.set_ylabel(r'$F_{\nu}$', size = 20)
        ax.set_xlabel(r'Wavelength [$\AA$]', size = 20)
        ax.legend()

        return ax
    
    else:
        #gets the advanced quantities for the chi2 info needed
        get_advanced_quantities(fit)

        min_chisqr = np.amin(fit.posterior.samples['chisq_phot'])

        filtered_wavelengths, filtered_lower_spec, filtered_median_spec, filtered_upper_spec = Spectra_Plot_Setup(fit)

        phot_waves, fnu_phot_lower, med_fnu_phot, fnu_phot_up = Phot_Plot_Setup(fit)

        min_chsq = round(min_chisqr, 2)

        phot_err_upper = fnu_phot_up - med_fnu_phot
        phot_err_lower = med_fnu_phot - fnu_phot_lower

        ax.fill_between(filtered_wavelengths, filtered_upper_spec, filtered_lower_spec, 
                        color = colors, alpha= .5)
        ax.plot(phot_waves, med_fnu_phot, '.', color = colors)
        ax.plot([] , [], color = colors, label = fr'{labels} $\chi^2$ = {min_chsq}')
                             
        error_props = {'yerr': [phot_err_upper, phot_err_lower], 'elinewidth': 20,
                       'fmt': 'none', 'color': colors, 'alpha':.5}
        
        ax.errorbar(phot_waves, med_fnu_phot, **error_props)

        ax.set_xscale('log')
        ax.set_yscale('log')
        
        phot_waves = phot_wavelength(fit)
        #wavelength to the left of the minimum photometric wavelength available
        left_wave = 1000
        min_wave = np.amin(phot_waves) - left_wave

        #wavelength rightward of the maximium photmoetric wavelength available
        right_wave = 1000
        max_wave = np.amax(phot_waves) + right_wave
        
        ax.set_xlim(min_wave, max_wave)
        
        adding_factor = len(filtered_wavelengths)//10
        
        indices = np.arange(0, 10) * adding_factor
        
        xticks = filtered_wavelengths[indices]
        
        ax.set_xticks(xticks)
        ax.set_xticklabels(np.round(xticks))
        #ax.ticklabel_format(scilimits = (-4, 4))
        
        ax.set_ylabel(r'$F_{\nu}$', size = 20)
        ax.set_xlabel(r'Wavelength [$\AA$]', size = 20)
        ID = galaxy_ID(fit)
        ax.set_title(f'Object: {ID}')
        
        return ax
    
########################
# Photometry inputs
#   
########################
def convert_to_microjansky(flux):
    
    #MagAB = 25 - 2.5*np.log10(flux)
    
    # 25 - 2.5log10(flux) = 8.9 - 2.5log(f_Jy)
    # 25 - 8.9 = -2.5(log(f_Jy) - log(flux))
    # 25 - 8.9/-2.5 = log(f_Jy/flux)
    # flux * 10**(25 - 8.9)/-2.5
    
    
    Jy = flux * 10**((25 - 8.9)/(-2.5)) #1e6muJy = 1Jy
    
    muJy = Jy * 1e6 #muJy/Jy
    
    return muJy

# def load_phot(ID):
    
#     Bagpipes_Phot_DF = pd.read_csv('../../data/Final_Matches/NEP/Original_Photom_For_Bagpipes.txt', 
#                                    sep = ' ', index_col = 0)
    
#     ID = int(ID)
    
#     #defining the columns we will use in the photometry
#     flux_cols = ['CFHT_u_FLUX',
#                      'HSC_g_FLUX',
#                      'HSC_r_FLUX',
#                      'HSC_i_FLUX',
#                      'HSC_z_FLUX',
#                      'HSC_y_FLUX',
#                      'IRAC_CH1_FLUX',
#                      'IRAC_CH2_FLUX']

#     flux_err_cols = ['CFHT_u_FLUXERR',
#                          'HSC_g_FLUXERR',
#                          'HSC_r_FLUXERR',
#                          'HSC_i_FLUXERR',
#                          'HSC_z_FLUXERR',
#                          'HSC_y_FLUXERR',
#                          'IRAC_CH1_FLUXERR',
#                          'IRAC_CH2_FLUXERR']

#     non_irac_fluxes = ['CFHT_u_FLUX',
#                          'HSC_g_FLUX',
#                          'HSC_r_FLUX',
#                          'HSC_i_FLUX',
#                          'HSC_z_FLUX',
#                          'HSC_y_FLUX']

#     non_irac_err = ['CFHT_u_FLUXERR',
#                          'HSC_g_FLUXERR',
#                          'HSC_r_FLUXERR',
#                          'HSC_i_FLUXERR',
#                          'HSC_z_FLUXERR',
#                          'HSC_y_FLUXERR']
    
#     #getting the full flux and flux error info
#     photom_flux = Bagpipes_Phot_DF.loc[ID, flux_cols]
#     photom_flux_err = Bagpipes_Phot_DF.loc[ID, flux_err_cols]

#     #we are artificially inflating the flux errors for non_irac filters by 5%
#     photom_flux_err[non_irac_err] = np.sqrt(((photom_flux_err[non_irac_err].values.astype(float))**2 + 
#                                          (.05 * photom_flux[non_irac_fluxes].values.astype(float))**2))



#     #############
#     # Making error be 20% larger for IRAC 
#     #############
    
#     flux_irac = ['IRAC_CH1_FLUX', 'IRAC_CH2_FLUX']
#     error_flux_irac = ['IRAC_CH1_FLUXERR', 'IRAC_CH2_FLUXERR']

#     photom_flux_err[error_flux_irac] = np.sqrt(((photom_flux_err[error_flux_irac].values.astype(float))**2 + 
#                                                (.2 *  photom_flux[flux_irac].values.astype(float))**2))

#     #getting the snr of sources
#     snr = photom_flux/photom_flux_err
    
#     #if the snr is below -5 then we know it is bad we make the flux 0 and error really big
#     bad_flux_idx = snr < -5

#     #setting bad flux to a really small value and error to be really big
#     photom_flux[bad_flux_idx] = 0
#     photom_flux_err[bad_flux_idx] = 1e99

#     TESLA_phot = np.c_[photom_flux.astype(float), photom_flux_err.astype(float)]

#     return TESLA_phot

def cat_phot_data(fit):
    
    get_advanced_quantities(fit)
    #global_DF_function()
    
    data = load_phot(fit.galaxy.ID)

    conv_data = np.array(data) * 1e-29 #converts the microjansky to F_nu units
    #I get the columns for flux and flux_err
    phot_flux = conv_data[:, 0]
    phot_err = conv_data[:, 1]
    
    return phot_flux, phot_err

def Plotting_Catalog_Photometry(fit, ax):
    
    #gets the advanced quantities
    get_advanced_quantities(fit)
    
    #gets the eff wavelength form the fit object
    phot_waves = phot_wavelength(fit)
    
    #gets the flux and flux_err form the photometry
    cat_flux, cat_err = cat_phot_data(fit)
    
    #making a properties dictionary for the phot plot
    phot_props = {'marker': 'o', 'color': 'navy', 'markersize': 15, 'fillstyle':'none'}
    
    ax.plot(phot_waves, cat_flux, **phot_props)
    
    #same thing for the error plot
    err_props = {'yerr': cat_err, 'capsize': 10, 'fmt':'none', 'color': 'navy'}
    ax.errorbar(phot_waves, cat_flux)
    
    return ax

def plotting_Spec_with_photometry(fit, labels = 'Lya', color = 'orchid', ax = None, 
                                  min_chisqr = True, save = False, 
                                  plot_output = '/work/07446/astroboi/ls6/TESLA/data/Plots/Bagpipes_Plots'):
    
    #this is for the ylim take min and max photometry value and multiply it 
    #by the correpsonding number
    upper_limit = 3
    lower_limit = .3
    
    if ax == None:
        
        if min_chisqr:
            #we need to get the minimum chisqr spectrum to plot
            
            #we get the min chisqr value
            min_chisqr = np.amin(fit.posterior.samples['chisq_phot'])
            
            #gets the catalog flux and flux error
            cat_flux, cat_err = cat_phot_data(fit)

            #Since no ax was given we make a plot for the spectrum and photometry
            
            fig = plt.figure(figsize = (12, 10), constrained_layout=True)
            
            spec2 = gridspec.GridSpec(ncols=1, nrows=3, figure=fig)
            
            f2_ax1 = fig.add_subplot(spec2[0, :])
            
            #gets the eff wavelength
            phot_waves = phot_wavelength(fit)
            
            #gets the best fit spectra wavelengths and spectrum
            best_fit_wavelengths, best_fit_spec = Best_Fit_Model_Spec(fit)
            
            #we get the BP photometric setup
            phot_waves, fnu_phot_lower, med_fnu_phot, fnu_phot_up = Phot_Plot_Setup(fit)

            #rounding the chisqr value
            min_chsq = round(min_chisqr, 2)

            #we then are selecting the min and max photometric value
            catalog_flux = np.unique(cat_flux)
            
            ymin = -99
            ymax = -99
            
            #we see if there is a zero in the photometry flux
            if 0 in catalog_flux:
                
                non_zero_mask = catalog_flux > 0
                non_zero_phot = catalog_flux[non_zero_mask]
                
                ymin = np.amin(non_zero_phot) * lower_limit
                ymax = np.amax(non_zero_phot) * upper_limit
            
            else:
            
                ymax = np.amax(catalog_flux) * upper_limit
                ymin = np.amin(catalog_flux) * lower_limit


            #plotting up the BP photometry
            #f2_ax1.fill_between(phot_waves, fnu_phot_up, fnu_phot_lower, 
            #                    color = color, alpha= .5)
            phot_err_upper = fnu_phot_up - med_fnu_phot
            phot_err_lower = med_fnu_phot - fnu_phot_lower
            
            errorbar_props = {'yerr':[phot_err_upper, phot_err_lower], 'elinewidth':20, 
                         'fmt': 'none', 'color': color, 'alpha': .5}
        
            f2_ax1.errorbar(phot_waves, med_fnu_phot, **errorbar_props)
            f2_ax1.plot(phot_waves, med_fnu_phot, '.', color = color)
            
            #plotting up the best fit spectrum
            f2_ax1.plot(best_fit_wavelengths, best_fit_spec, 
                        color = color, label = fr'{labels} $\chi^2$ = {min_chsq}')
            
            #setting scales to log
            f2_ax1.set_xscale('log')
            f2_ax1.set_yscale('log')
            
            f2_ax1.set_ylabel(r'$F_{\nu}$', size = 20)
            f2_ax1.set_xlabel(r'Wavelength [$\AA$]', size = 20)
            
            #plotting up the catalog photometric info
            f2_ax1.plot(phot_waves, cat_flux, 'o', color = 'navy', markersize = 15, fillstyle='none')
            f2_ax1.errorbar(phot_waves, cat_flux, yerr = cat_err, 
                            capsize = 3, fmt='none', color = 'navy')

            f2_ax1.set_ylim(ymin, ymax)
            
            ID = galaxy_ID(fit)
            f2_ax1.set_title(f'Object: {ID}', fontsize = 15)

            #getting xticks
            adding_factor = len(best_fit_wavelengths)//10

            indices = np.arange(0, 10) * adding_factor

            xticks = best_fit_wavelengths[indices]

            f2_ax1.set_xticks(xticks)
            f2_ax1.set_xticklabels((xticks.astype(int)))
            
            window = 1000
            min_x = np.amin(phot_waves) - window
            max_x = np.amax(phot_waves) + window + 5000
            
            f2_ax1.set_xlim(min_x, max_x)
            
            f2_ax1.legend(fontsize = 15)
            
            if save == True:
                #print('Saving the output and placing the output here:')
                
                unique_id = ID
                #print(f'{plot_output}/{unique_id}.png')
                fig.savefig(f'{plot_output}/{unique_id}_SED.png', dpi = 150)
            
            plt.show()
            
        else:
            
            #We are NOT plotting the min chisqr spectrum
            
            # we get the catalog flux and flux errors
            cat_flux, cat_err = cat_phot_data(fit)

            #still no ax was given so we make the figure for the user
            fig = plt.figure(figsize = (12, 10), constrained_layout=True)
            spec2 = gridspec.GridSpec(ncols=1, nrows=3, figure=fig)

            f2_ax1 = fig.add_subplot(spec2[0, :])

            #we get the eff wavelengths for the photometry
            phot_waves = phot_wavelength(fit)
            
            #getting the bagpipes spectrum info
            filtered_wavelengths, filtered_lower_spec, filtered_median_spec, filtered_upper_spec = Spectra_Plot_Setup(fit)

            #get the BP photometry
            phot_waves, fnu_phot_lower, med_fnu_phot, fnu_phot_up = Phot_Plot_Setup(fit)
            
            #get the min_chi2 spectra to overplot over the 68% CL
            best_fit_wavelengths, best_fit_spec = Best_Fit_Model_Spec(fit)

            #we get the min chisqr to present in the plot
            min_chisqr = np.amin(fit.posterior.samples['chisq_phot'])
            min_chsq = round(min_chisqr, 2)

            #compute error for the BP photometry
            phot_err_upper = fnu_phot_up - med_fnu_phot
            phot_err_lower = med_fnu_phot - fnu_phot_lower

            #we see what the min and max value are from the photometric catalog
            catalog_flux = np.unique(cat_flux)
            
            ymin = -99
            ymax = -99
            
            if 0 in catalog_flux:
                
                non_zero_mask = catalog_flux > 0
                non_zero_phot = catalog_flux[non_zero_mask]
                
                ymin = np.amin(non_zero_phot) * lower_limit
                ymax = np.amax(non_zero_phot) * upper_limit
            else:
                
                ymax = np.amax(catalog_flux) * upper_limit
                ymin = np.amin(catalog_flux) * lower_limit
                
            #this is plotting up the Bagpipes 1 sigma spectra interval
            f2_ax1.fill_between(filtered_wavelengths, filtered_upper_spec, filtered_lower_spec, 
                                color = color, alpha= .5)
            f2_ax1.plot([] , [], color = 'black', ls = '-', label = fr'{labels} $\chi^2$ = {min_chsq}')
            
            f2_ax1.plot(best_fit_wavelengths, best_fit_spec, 
                        color = 'black')
            
            #this plots up the photometry
            f2_ax1.plot(phot_waves, med_fnu_phot, '.', color = color)
            f2_ax1.errorbar(phot_waves, med_fnu_phot, yerr = [phot_err_upper, phot_err_lower],
                            elinewidth=20, fmt = 'none', color = color, alpha =1)


            f2_ax1.set_xscale('log')
            f2_ax1.set_yscale('log')
            
            f2_ax1.set_ylabel(r'$F_{\nu}$', size = 20)
            f2_ax1.set_xlabel(r'Wavelength [$\AA$]', size = 20)

           #plotting up the catalog photometric info
            f2_ax1.plot(phot_waves, cat_flux, 'o', color = 'navy', markersize = 15, fillstyle='none')
            f2_ax1.errorbar(phot_waves, cat_flux, yerr = cat_err, capsize = 3, 
                            fmt='none', color = 'navy')
            
            f2_ax1.set_ylim(ymin, ymax)
            
            window = 1000
            min_x = np.amin(phot_waves) - window
            max_x = np.amax(phot_waves) + window + 5000
            
            f2_ax1.set_xlim(min_x, max_x)
            
            ID = galaxy_ID(fit)
            f2_ax1.set_title(f'Object: {ID}', fontsize = 15)
            
            f2_ax1.legend(fontsize = 15)
            
            if save == True:
                unique_id = galaxy_ID(fit)
                fig.savefig(f'{plot_output}/{unique_id}_SED.png')

            plt.show()

    else:
        #this is the part if an ax was given to us
        
        if min_chisqr:
            #we want the minimum chisqr spectrum plotted
            
            #we get the eff wavelength and catalog flux and flux_err
            phot_waves = phot_wavelength(fit)
            cat_flux, cat_err = cat_phot_data(fit)
            
            #print('Catalog Flux')
            #print(cat_flux)
            #print('Catalog Error')
            #print(cat_err)

            #We call another function called Spec_Phot_Plots to plot up the best fit spectrum
            top_ax = Spec_Phot_Plots(fit, ax, colors = color, labels = labels, min_chisqr = min_chisqr)
            
            #we simply add on the catalog photometry

            top_ax.plot(phot_waves, cat_flux, 'o', markersize = 15, fillstyle='none', color = 'navy')
            top_ax.errorbar(phot_waves, cat_flux, yerr = cat_err, 
                            capsize = 5, fmt='none', color = 'navy')
            
            catalog_flux = np.unique(cat_flux)
            
            ymin = 0
            ymax = 0
            
            if 0 in catalog_flux:
                
                non_zero_mask = catalog_flux > 0
                non_zero_phot = catalog_flux[non_zero_mask]
                
                ymin = np.amin(non_zero_phot) * lower_limit
                ymax = np.amax(non_zero_phot) * upper_limit
                
            else:
                
                ymax = np.amax(catalog_flux) * upper_limit
                ymin = np.amin(catalog_flux) * lower_limit
            
            top_ax.set_ylim(ymin, ymax)
            
            window = 1000
            
            xmin= np.amin(phot_waves) - window
            xmax = np.amax(phot_waves) + (window +250) + 5000
            
            top_ax.set_xlim(xmin, xmax)
            
            
            return top_ax
            
        else:
            
            phot_waves = phot_wavelength(fit)
            cat_flux, cat_err = cat_phot_data(fit)

            top_ax = Spec_Phot_Plots(fit, ax, labels = labels, colors = color)

            top_ax.plot(phot_waves, cat_flux, 'o', markersize = 15, fillstyle='none', color = 'navy')
            top_ax.errorbar(phot_waves, cat_flux, yerr = cat_err, capsize = 3, 
                            fmt='none', color = 'darkslategrey')
            
            catalog_flux = np.unique(cat_flux)
            
            ymin = -99
            ymax = -99
            
            if 0 in catalog_flux:
                
                non_zero_mask = catalog_flux > 0
                non_zero_phot = catalog_flux[non_zero_mask]
                
                ymin = np.amin(non_zero_phot) * lower_limit
                ymax = np.amax(non_zero_phot) * upper_limit
                
            else:
                
            
                ymax = np.amax(catalog_flux) * upper_limit
                ymin = np.amin(catalog_flux) * lower_limit
            
            #top_ax.set_ylim(ymin, ymax)
           
            window = 1000
            
            xmin= np.amin(phot_waves) - window
            xmax = np.amax(phot_waves) + window
            
            top_ax.set_xlim(xmin, xmax)
            top_ax.set_ylim(ymin, ymax)

            return top_ax
        
def posterior_plot(fit, param, color = None, stat = 'density', element = 'step',  
                   fill = True, kde = True, bins = 20, alpha = 1):
    
    fig, ax2 = plt.subplots(figsize = (7, 5), constrained_layout = True, facecolor = 'white')
    
    hist_props = {'ax': ax2, 'stat': stat, 'element': element, 'fill': fill, 'kde': kde, 'bins':bins,
                  'alpha': alpha}
    
    if color:
        hist_props.update({'color': color})
    
    dat = fit.posterior.samples[param]
    
    ax2 = sb.histplot(data = dat, **hist_props)

    ax2.axvline(np.median(dat), color = 'black', linestyle = '--', 
                label = f'Median: {np.median(dat):.2f}')
    ax2.axvline(np.percentile(dat, q = 16), color = 'black', linestyle = '-')
    ax2.axvline(np.percentile(dat, q = 84), color = 'black', linestyle = '-')
    
    plot_props = {'title': 'Posterior Distribution'}
    
    try:
        #plot_props.update({'xlabel': xlabels[param]})
        ax2.set_xlabel(f'{xlabels[param]}', fontsize = 15)
    except:
        #plot_props.update({'xlabel': param})
        ax2.set_xlabel(param, fontsize = 15)
        
    ax2.set(**plot_props)
    
    ax2.legend()
    
    return fig, ax2
    
        
def posterior_plot_w_ax(fit, param, ax1, color = None, stat = 'density', element = 'step',  
                   fill = True, kde = True, bins = 20, alpha = 1):
    
    hist_props = {'ax': ax1, 'stat': stat, 'element': element, 'fill': fill, 'kde': kde, 'bins':bins,
                  'alpha': alpha}
    
    if color:
        hist_props.update({'color': color})
    

    dat = fit.posterior.samples[param]
    
    ax2 = sb.histplot(data = dat, **hist_props)

    ax2.axvline(np.median(dat), color = 'black', linestyle = '--', 
                label = f'Median: {np.median(dat):.2f}')
    ax2.axvline(np.percentile(dat, q = 16), color = 'black', linestyle = '-')
    ax2.axvline(np.percentile(dat, q = 84), color = 'black', linestyle = '-')
    
    plot_props = {'title': 'Posterior Distribution'}
    
    try:
        #plot_props.update({'xlabel': xlabels[param]})
        ax2.set_xlabel(f'{xlabels[param]}', fontsize = 15)
    except:
        #plot_props.update({'xlabel': param})
        ax2.set_xlabel(param, fontsize = 15)
        
    ax2.set(**plot_props)
    
    ax2.legend()
    
    return ax2

def spectrum_with_posteriors(fit, spec_color = 'orchid', dist1_color = 'red', 
                             dist2_color = 'dodgerblue', dist3_color = 'salmon',
                             save = None, min_chisqr = True):
    
    if save:
        
        fig = plt.figure(figsize = (12, 7), constrained_layout=True)
        spec2 = gridspec.GridSpec(ncols=3, nrows=2, figure=fig)

        f1_ax1 = fig.add_subplot(spec2[0, :])
        f1_ax2 = fig.add_subplot(spec2[1, 0])
        f1_ax3 = fig.add_subplot(spec2[1, 1])
        f1_ax4 = fig.add_subplot(spec2[1, 2])

        post_fit = plotting_Spec_with_photometry(fit, labels = '', ax = f1_ax1, 
                                                 min_chisqr = min_chisqr , color = spec_color)
        #post_fit.set_ylim(1e-30, 1e-28)
        post_fit.legend()

        post_ax1 = posterior_plot_w_ax(fit, 'stellar_mass', ax1 = f1_ax2, color = dist1_color)

        post_ax2 = posterior_plot_w_ax(fit, 'ssfr', ax1 = f1_ax3, color = dist2_color)

        post_ax3 = posterior_plot_w_ax(fit, 'dust:Av', ax1 = f1_ax4, color = dist3_color)
        
        plot_output = '/work/07446/astroboi/ls6/TESLA/data/Plots/Bagpipes_Plots'
        ID = galaxy_ID(fit)
        
        fig.savefig(f'{plot_output}/{ID}_Spectrum_and_Posterior', dpi = 150)
        
        #plt.show()
        
    else:

        fig = plt.figure(figsize = (12, 7), constrained_layout=True)
        spec2 = gridspec.GridSpec(ncols=3, nrows=2, figure=fig)

        f1_ax1 = fig.add_subplot(spec2[0, :])
        f1_ax2 = fig.add_subplot(spec2[1, 0])
        f1_ax3 = fig.add_subplot(spec2[1, 1])
        f1_ax4 = fig.add_subplot(spec2[1, 2])

        post_fit = plotting_Spec_with_photometry(fit, labels = '', ax = f1_ax1, 
                                                 min_chisqr = min_chisqr, color = spec_color)
        #post_fit.set_ylim(1e-30, 1e-28)
        post_fit.legend()

        post_ax1 = posterior_plot(fit, 'stellar_mass', ax1 = f1_ax2, color = dist1_color)
        #post_ax1.set_xlabel(xlabels['stellar_mass'])

        post_ax2 = posterior_plot(fit, 'ssfr', ax1 = f1_ax3, color = dist2_color)
        #post_ax2.set_xlabel(xlabels['sfr'])

        post_ax3 = posterior_plot(fit, 'dust:Av', ax1 = f1_ax4, color = dist3_color)
        #post_ax3.set_xlabel(xlabels['dust:Av'])

        #plt.show()
        
        return fig, post_fit, post_ax1, post_ax2, post_ax3
        
if __name__ == "__main__":
    
    args = list(map(str.lower,sys.argv))
    
    print('Getting the Index')
    index = GetIndex()
    
    fit = fit_BP(index, Bagpipes_Phot_DF, load_spec_and_phot, fit_BP = False)
    fit.posterior.get_advanced_quantities()

    if '--save' in args:

        indx = args.index("--save")

        save = bool(sys.argv[indx + 1])

    else:
        print('No Input File Name Present')
        sys.exit(-1)

    if '--plot_output' in args:

        indx = args.index("--plot_output")

        plot_output = sys.argv[indx + 1]

    else:
        print('No Output Plot File Name Present')
        sys.exit(-1)    
        
    try:        
        plotting_Spec_with_photometry(fit, labels = r'Ly$\alpha$', color = 'orchid', 
                                      ax = None, min_chisqr = False, save = save)

    except:
        print()
        #unique_id = Bagpipes_Phot_DF.loc[index, 'Unique_IDs']
        #print(f'{unique_id} had an error in grabbing the spectrum!')
        #print()