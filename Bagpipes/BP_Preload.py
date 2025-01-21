import sys
args = list(map(str.lower,sys.argv))
import pandas as pd

def GetIndex():
    
    if '--index' in args:
        
        indx = args.index("--index")
        
        gal_index = sys.argv[indx + 1]
        
        return gal_index
    
    else:
        print('No Galaxy Index detected. Use --index <index>')
        sys.exit(-1) 
        
def read_input_phot_cat():
    
    if '--bp_input_cat' in args:
        
        indx = args.index("--bp_input_cat")
        
        bp_phot_path = sys.argv[indx + 1]
        
        bp_phot_cat = pd.read_csv(bp_phot_path, 
                                  sep = ' ', 
                                  index_col = 0)
        
        return bp_phot_cat
    
    else:
        print('No Bagpipes input catalog. Use --bp_input_cat <path to BP phot Cat>')
        sys.exit(-1) 
        
def get_run_name():
    
    if '--run_name' in args:
        
        indx = args.index("--run_name")
        
        run_name = sys.argv[indx + 1]
        
        return run_name
    
    else:
        print('No Bagpipes input catalog. Use --bp_input_cat <path to BP phot Cat>')
        sys.exit(-1) 

def make_filters(filter_set = 'TESLA'):
    
    if filter_set == 'TESLA':
        
        print('TESLA Filter Set to be used \n')
        
        base_TESLA = '/work/07446/astroboi/ls6/bagpipes_install_test/filters/'

        TESLA_Filters = ['CFHT/H20_CFHT_Megaprime.u.dat', 
                         'HSC_Filters/HSC_g_filt.txt',
                         'HSC_Filters/HSC_r_filt.txt',
                         'HSC_Filters/HSC_i_filt.txt',
                         'HSC_Filters/HSC_z_filt.txt',
                         'HSC_Filters/HSC_y_filt.txt',
                         'HSC_Filters/IRAC_36_filt.txt',
                         'HSC_Filters/IRAC_45_filt.txt']    

        TESLA_filts = [base_TESLA+x for x in TESLA_Filters]
        
        print('READING IN TESLA FILTERS: ')
        for x in TESLA_filts:
            print(x)
        print()
        return TESLA_filts
    
    elif filter_set == 'SHELA':
        
        print('SHELA Filter Set to be used \n')
        
        base_SHELA = '/work/07446/astroboi/ls6/bagpipes_install_test/filters/'
        
#         TESLA_Filters = ['CFHT/H20_CFHT_Megaprime.u.dat', 
#                          'HSC_Filters/HSC_g_filt.txt',
#                          'HSC_Filters/HSC_r_filt.txt',
#                          'HSC_Filters/HSC_i_filt.txt',
#                          'HSC_Filters/HSC_z_filt.txt',
#                          'HSC_Filters/HSC_y_filt.txt',
#                          'HSC_Filters/IRAC_36_filt.txt',
#                          'HSC_Filters/IRAC_45_filt.txt']    

#         TESLA_filts = [base_TESLA+x for x in TESLA_Filters]

#         return TESLA_filts
        pass

    elif filter_set == 'UDS':
        # ['FLUXERR_F435W',
        #  'FLUXERR_F606W',
        #  'FLUXERR_F814W',
        #  'FLUXERR_F125W',
        #  'FLUXERR_F140W',
        #  'FLUXERR_F160W',
        #  'FLUXERR_F090W',
        #  'FLUXERR_F115W',
        #  'FLUXERR_F150W',
        #  'FLUXERR_F200W',
        #  'FLUXERR_F277W',
        #  'FLUXERR_F356W',
        #  'FLUXERR_F410M',
        #  'FLUXERR_F444W']
        base_uds = '/work/07446/astroboi/ls6/bagpipes_install_test/filters/'
        filter_files =  [
                        # 'HST/ACS/ACS_F435W.txt',
                        # 'HST/ACS/ACS_F606W.txt', 
                        # 'HST/ACS/ACS_F814W.txt',
                        # 'HST/WFC3/WFC3_F125W.txt',
                        # 'HST/WFC3/WFC3_F140W.txt',
                        # 'HST/WFC3/WFC3_F160W.txt',
                        'JWST/F090W.txt',
                        'JWST/F115W.txt',
                        'JWST/F150W.txt',
                        'JWST/F200W.txt',
                        'JWST/F277W.txt',
                        'JWST/F356W.txt',
                        'JWST/F410M.txt',
                        'JWST/F444W.txt']
        
        UDS_filts = [base_uds+x for x in filter_files]

        print('Reading in the UDS Filters: ')
        for x in UDS_filts:
            print(x)
        print('-------------------------------------------------------------------------------------')
        print()

        return UDS_filts        
    else:
        print('No Valid Filter Set Given')
        print('Only have TESLA and SHELA')
        sys.exit(-1)