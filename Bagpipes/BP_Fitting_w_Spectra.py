from BP_PreLoad import *
import bagpipes as pipes
import pandas as pd
from astropy.io import fits
import numpy as np
import time
from astropy.cosmology import LambdaCDM
from astropy.table import Table

print('Grabbing the necessary filter curves')
filters = make_filters('UDS')

def get_redshift(DF, ID):
    
    z_tesla = DF.loc[ID, 'Redshift']

    return z_tesla

def Galaxy_Model_Builder(ID, load_func, filters):
    '''
    This function will make a bagpipes galaxy model for one ID
    
    '''
    #need to take the extra step of converting ID to string for bagpipes galaxy function to work properly
    galaxy_bp = pipes.galaxy(str(ID), load_func, filt_list = filters, spectrum_exists=True, spec_units = 'mujy')
    
    return galaxy_bp

def load_spec(ID):

    spectra = Table.read('CAPERS_UDS_P2_s000136645_x1d.fits')

    micron_wave = np.array(spectra['wave']).astype(float)
    fnu = np.array(spectra['flux']).astype(float)
    fnu_err = np.array(spectra['err']).astype(float)
    
    nan_mask_flux= np.isfinite(fnu)
    nan_mask_err = np.isfinite(fnu_err)
    
    mask = nan_mask_flux & nan_mask_err

    good_wave = micron_wave[mask]
    good_fnu = fnu[mask]
    good_fnu_err = fnu_err[mask]

    spectrum = np.c_[good_wave*1e4, #converting the microns to Angstroms
                     good_fnu, 
                     good_fnu_err]

    return spectrum

def load_phot_UDS(ID):
    
    ##########
    #WILL NEED TO CHANGE THIS BASED OFF OF THE INPUT CATALOG
    #WE NEED TO DO THIS 
    ##########
    Bagpipes_Phot_DF = pd.read_csv('/work/07446/astroboi/ls6/TESLA/PRIMER_UDS/CAPERS_UDS_136645_UNICORN_PHOTOM.txt', 
                                   sep = ' ', index_col = 0)
    
    ID = int(ID)
    
    #defining the columns we will use in the photometry
    flux_cols = [
                 # 'FLUX_F435W',
                 # 'FLUX_F606W',
                 # 'FLUX_F814W',
                 # 'FLUX_F125W',
                 # 'FLUX_F140W',
                 # 'FLUX_F160W',
                 'FLUX_F090W',
                 'FLUX_F115W',
                 'FLUX_F150W',
                 'FLUX_F200W',
                 'FLUX_F277W',
                 'FLUX_F356W',
                 'FLUX_F410M',
                 'FLUX_F444W']

    flux_err_cols = [
                     # 'FLUXERR_F435W',
                     # 'FLUXERR_F606W',
                     # 'FLUXERR_F814W',
                     # 'FLUXERR_F125W',
                     # 'FLUXERR_F140W',
                     # 'FLUXERR_F160W',
                     'FLUXERR_F090W',
                     'FLUXERR_F115W',
                     'FLUXERR_F150W',
                     'FLUXERR_F200W',
                     'FLUXERR_F277W',
                     'FLUXERR_F356W',
                     'FLUXERR_F410M',
                     'FLUXERR_F444W']
    
    #getting the full flux and flux error info
    photom_flux = Bagpipes_Phot_DF.loc[ID, flux_cols]
    photom_flux_err = Bagpipes_Phot_DF.loc[ID, flux_err_cols]

    #we are artificially inflating the flux errors for non_irac filters by 5%
    photom_flux_err[flux_err_cols] = np.sqrt(((photom_flux_err[flux_err_cols].values.astype(float))**2 + 
                                         (.05 * photom_flux[flux_cols].values.astype(float))**2))

    snr = photom_flux/photom_flux_err
    
    bad_flux_idx = snr < -5

    photom_flux[bad_flux_idx] = 0
    photom_flux_err[bad_flux_idx] = 1e12

    UDS_phot = np.c_[photom_flux.astype(float)/1000,     #converting nJy to muJy
                       photom_flux_err.astype(float)/1000] #converting nJy to muJy

    return UDS_phot

def load_both(ID):
    
    spectrum = load_spec(ID)
    phot = load_phot_UDS(ID)

    return spectrum, phot

def fit_instruction_nebular_fixedz(z, model = 'delayed_tau'):
    
    '''
    Function that creates the bagpipes fit model, this is what bagpipes will attempt to fit with.
    
    Input:
    z (float): redshift of the source, but feel free to change this if you do not have redshift
    model (str): 
    returns:
    fit_instruction (dictionary): the intructions BP will use to fit the galaxy
    '''
    
    if model == 'delayed_tau':
        print("Making Fit Instructions for Delayed-Tau SFH Model")
        
        #Model Building 
        model = {}
        model['age'] = (.01, 13)                  # Age of the galaxy in Gyr

        model['tau'] = (.02, 14)                  # Delayed decayed time
        model["metallicity"] = (0., 2)          # Metallicity in terms of Z_sol
        model["massformed"] = (4., 13.)           # log10 of mass formed
        
        dust = {}                                 # Dust component
        dust["type"] = "Calzetti"                 # Define the shape of the attenuation curve
        dust["Av"] = (0., 3.)                     # Vary Av between 0 and 3 magnitudes

        #############
        #NOTE: Before next run need to talk to steve about fixing this at 1.44 or letting it be a param
        #      BP will fit
        #############
        dust["eta"] = 1

        #will need to include this, this includes SF regions and their emission to the spectrum
        nebular = {}

        #changed upper limits from -4 to -2 as it seems I keep getting an error with -1
        nebular["logU"] = (-4, -1)

        fit_instructions = {}
        fit_instructions['delayed'] = model
        
        fit_instructions['redshift'] = z
        
        fit_instructions['dust'] = dust
        fit_instructions['nebular'] = nebular
    
    
        return fit_instructions
    
    elif model == 'nonparam':
        #values taken from the PLANCK CMB 2018 paper
        Om0 = .315
        Ode0 = 1 - Om0
        cosmo = LambdaCDM(H0 = 67.4, 
                          Om0 = .315, 
                          Ode0 = Ode0)
        
        age_Gyr = cosmo.age(z).value
        age_Myr = age_Gyr * 1e3
        
        starting_bin = np.array([0])
        bin_end = np.log10(age_Myr) - .05
        
        bins = np.logspace(np.log10(5), bin_end, 9)
        
        age_bins = np.append(starting_bin, bins)
        
        
        print("Making Fit Instructions for Non-Parametric SFH Model")
        dust = {}
        dust["type"] = "Calzetti"
        dust["eta"] = 1.
        dust["Av"] = (0., 3.)

        nebular = {}
        nebular["logU"] = (-4, -1)

        fit_instructions = {}
        fit_instructions["dust"] = dust
        fit_instructions["nebular"] = nebular
        fit_instructions["redshift"] = z

        print(age_bins)
        
        continuity = {}
        continuity["massformed"] = (0.0001, 13.)
        continuity["metallicity"] = (0.01, 3.)
        continuity["metallicity_prior"] = "log_10"
        continuity["bin_edges"] = list(age_bins)

        for i in range(1, len(continuity["bin_edges"])-1):
            continuity["dsfr" + str(i)] = (-10., 10.)
            continuity["dsfr" + str(i) + "_prior"] = "student_t"

        fit_instructions["continuity"] = continuity
        
        return fit_instructions
        
    elif model == "bursty":

        rtab = Table.read('jwst_nirspec_prism_disp.fits')

        wave_angstroms = np.array(rtab['WAVELENGTH']).astype(float) * 1e4 #converting microns to angstroms
        R = np.array(rtab['R']).astype(float)

        BP_R_curve = np.c_[wave_angstroms, R]
        
        #values taken from the PLANCK CMB 2018 paper
        Om0 = .315
        Ode0 = 1 - Om0
        cosmo = LambdaCDM(H0 = 67.4, 
                          Om0 = .315, 
                          Ode0 = Ode0)
        
        age_Gyr = cosmo.age(z).value
        age_Myr = age_Gyr * 1e3
        
        starting_bin = np.array([0])
        bin_end = np.log10(age_Myr) - .01
        
        bins = np.logspace(np.log10(5), bin_end, 9)
        
        age_bins = np.append(starting_bin, bins)
        
        print("Making Fit Instructions for Bursty Non-Parametric SFH Model")
        dust = {}
        dust["type"] = "Calzetti"
        dust["eta"] = 1.
        dust["Av"] = (0., 3.)

        nebular = {}
        nebular["logU"] = (-4, -1)

        fit_instructions = {}
        fit_instructions["dust"] = dust
        fit_instructions["nebular"] = nebular
        fit_instructions["redshift"] = z


        continuity = {}
        continuity["massformed"] = (0.0001, 13.)
        continuity["metallicity"] = (0.01, 5.)
        continuity["metallicity_prior"] = "log_10"
        continuity["bin_edges"] = list(age_bins)

        for i in range(1, len(continuity["bin_edges"])-1):
            continuity["dsfr" + str(i)] = (-10., 10.)
            continuity["dsfr" + str(i) + "_prior"] = "student_t"
            
            #adding this prior scale to make it bursty
            continuity["dsfr" + str(i) + "_prior_scale"] =2.0
            
        fit_instructions["continuity"] = continuity

        calib = {}
        calib["type"] = "polynomial_bayesian"
        
        calib["0"] = (0.5, 1.5) # Zero order is centred on 1, at which point there is no change to the spectrum.
        calib["0_prior"] = "Gaussian"
        calib["0_prior_mu"] = 1.0
        calib["0_prior_sigma"] = 0.25
        
        calib["1"] = (-0.5, 0.5) # Subsequent orders are centred on zero.
        calib["1_prior"] = "Gaussian"
        calib["1_prior_mu"] = 0.
        calib["1_prior_sigma"] = 0.25
        
        calib["2"] = (-0.5, 0.5)
        calib["2_prior"] = "Gaussian"
        calib["2_prior_mu"] = 0.
        calib["2_prior_sigma"] = 0.25
        
        fit_instructions["calib"] = calib

        noise = {}
        noise["type"] = "white_scaled"
        noise["scaling"] = (1., 10.)
        noise["scaling_prior"] = "log_10"
        fit_instructions["noise"] = noise
        
        fit_instructions['R_curve'] = BP_R_curve
        
        print(fit_instructions)
        
        return fit_instructions

def fit_BP(index, filters, load_func, z, run, only_fit = True, model = 'delayed_tau'):

    print('Making the BP Galaxy Model')
    BP_Galaxy = Galaxy_Model_Builder(index, load_func, filters)
    
    print('Getting the BP Fit Instructions')
    print(f'Redshift is: {z: .5f}')
    fit_instructions = fit_instruction_nebular_fixedz(z, model = model)
    
    if only_fit:
        
        start = time.time()
        fit = pipes.fit(BP_Galaxy, fit_instructions, run = run)
    
        fit.fit(verbose=True)
        end = time.time()
        
        duration = end - start
        print(f'Full Time of the Fit is: {duration:.2f} seconds, {duration/60:.2f} Minutes')
        
    else:
        
        fit = pipes.fit(BP_Galaxy, fit_instructions, run = run)
    
        fit.fit(verbose=True)
    
        return fit
    
def fit_serial_bp(DF, IDs, run,
                  load_func, 
                  filters = filters,
                  only_fit = True, 
                  test = False, model = 'nonparam'):
    
    if test:
        print('Testing the Code on the First 10 Sources')
        
        for idx in IDs[:10]:
            print(f'Fitting Galaxy ID: {idx}')
            z_tesla = get_redshift(DF, idx)
            fit_BP(idx, filters, load_func, only_fit = only_fit, model = model, z = z_tesla, run = run)
        
    else:
        
        print(f'Running on the Full Sample of: {DF.shape[0]} Sources')
        
        for idx in IDs:
            
            z_tesla = get_redshift(DF, idx)
            
            fit_BP(idx, filters, load_func, only_fit = only_fit, z = z_tesla)
        
def get_ID():
    
    if '--id' in args:
        
        indx = args.index("--id")
        
        ID = int(sys.argv[indx + 1])
    
    else:
        print('No ID detected.')
        sys.exit(-1) 
    
    return ID


if __name__ == '__main__':
    
    print('Grabbing Bagpipes Run Name')
    run = get_run_name()
    print(f'Run-Name: {run} Acquired')

    print('Attempting to read in Bagpipes Photometric Catalog')
    Bagpipes_Phot_DF = read_input_phot_cat()
    print('Read in Bagpipes Catalog')
    
    if '--test' in args:
        
        indx = args.index("--test")
        
        test = sys.argv[indx + 1]
        
        test = bool(test)
    
    else:
        print('No test detected. Defaulting to no Test')
        test = False 
    
    ID = get_ID()
    
    z_tesla = get_redshift(Bagpipes_Phot_DF, ID)
    
    #try:
        
    fit_BP(ID, filters, load_both, z_tesla, run, only_fit = True, model = 'bursty')
    print(f'Successfully fitted ID: {ID}')
    print()
    #except:
    #    print('ERROR IN FITTING!!!')
    #    print(f'Check on ID: {ID}')
    #    print()
        
    