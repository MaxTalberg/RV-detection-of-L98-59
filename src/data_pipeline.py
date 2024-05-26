import pandas as pd

def clean_data(espresso_path: str, harps_path: str):
    """
    Clean the data from the ESPRESSO and HARPS instruments.
    """


    ## HARPS
    # Column titles
    column_titles = [
    'Time', 'RV', 'e_RV', 'Halpha', 'e_Halpha', 'Hbeta', 'e_Hbeta', 'Hgamma', 'e_Hgamma', 'NaD', 'e_NaD',
    'Sindex', 'e_Sindex', 'FWHM', 'BIS'
    ]

    # Load the data
    harps_df = pd.read_csv(harps_path,
                           delim_whitespace=True,
                           names=column_titles
                           )
    
    # Adjust the time column to BJD by adding 2457000
    harps_df['Time'] += 2457000

    # Clean the data
    excluded_bjds = [2458503.795048, 2458509.552019, 2458511.568314, 2458512.581045] 
    harps_df = harps_df[~harps_df['Time'].isin(excluded_bjds)].copy()

    # Interpolate missing values
    harps_df['FWHM'] = harps_df['FWHM'].replace('---', 0).astype(float).interpolate(method='linear')
    harps_df['BIS'] = harps_df['BIS'].replace('---', 0).astype(float).interpolate(method='linear')


    ## ESPRESSO
    # Column titles
    espresso_column_titles = [
            'Time', 'RV', 'e_RV', 'FWHM', 'e_FWHM', 'BIS', 'e_BIS', 'Contrast', 'e_Contrast', 'Sindex', 'e_Sindex', 
    'Halpha', 'e_Halpha', 'NaD', 'e_NaD', 'BERV', 'Inst'
    ]

    # Load the data
    espresso_df = pd.read_csv(espresso_path,
                              delim_whitespace=True, 
                              names=espresso_column_titles
                              )
    
    # Adjust the time column to BJD by adding 2400000
    espresso_df['Time'] += 2400000
    
    # Clean the data
    excluded_bjds = [2458645.496, 2458924.639, 2458924.645]
    tolerance = 1e-3
    espresso_df = espresso_df[~espresso_df['Time'].apply(lambda x: any(abs(x - bjd) < tolerance for bjd in excluded_bjds))]

    # Split the data into pre and post fiber change
    pre_df = espresso_df[espresso_df['Inst'] == 'Pre']
    post_df = espresso_df[espresso_df['Inst'] == 'Post']


    ## Merge the data
    combined_df = pd.concat([espresso_df, harps_df], axis=0, ignore_index=True)

    return combined_df, harps_df, espresso_df, pre_df, post_df
