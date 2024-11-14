#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 13:04:20 2024

This code models transport-dependent abrasion of a sediment source of uniform
strengths and grain size distribution. This is then plotted as Volumetric fraction remaining for each uniform strength vs distance.

@author: Brian Pinke **Modified From Allison Pfeiffer

# Plot Figure 17: Modeled abrasion of pulses with uniform strength, using the MRWR GSD.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Read in the source material SHRS data  and fix up data to work in script

path=os.path.join( os.getcwd(), 'Data/fieldbook_data_2023_USE.xlsx')
# loads in SHRS data text files: (1) metadata (2) lithology (3) medians
field_data = pd.read_excel (
        path, 
        sheet_name='1 - Measurements', 
        header = 0)

path=os.path.join( os.getcwd(), 'Data/SuiattleFieldData_Combined20182019.xlsx')
# loads in Suiattle data text files: (1) metadata (2) lithology (3) medians
suiattleGSD_data = pd.read_excel (
        path, 
        sheet_name='3 - Grain size df deposit', 
        header = 0)

path=os.path.join( os.getcwd(), 'Data/SuiattleFieldData_Combined20182019.xlsx')
suiattleSHRS_data = pd.read_excel (
        path, 
        sheet_name='4-SHRSdfDeposit', 
        header = 0)

# Define Colors
color_dict = {
  (0.9, 0.9, 0.9): '10', # lightest gray
  (0.8, 0.8, 0.8): '20',
  (0.7, 0.7, 0.7): '30',
  (0.6, 0.6, 0.6): '40',
  (0.5, 0.5, 0.5): '50',
  (0.4, 0.4, 0.4): '60',
  (0.3, 0.3, 0.3): '70',
  (0.2, 0.2, 0.2): '80', # darkest gray
  (0.1, 0.1, 0.1):'90'
}

# change font size
plt.rcParams["font.family"] = "Arial"
plt.rcParams['font.size'] = 10

# set random seed to 5
np.random.seed(5) 

os.chdir(os.getcwd()+'/Figs')

#%%
# Denote what each deposit is called
MMLR = "Mt. Meager (Lillooet River)"
GPSR = "Glacier Peak (Suiattle River)"
MRWR = "Mt. Rainier (White River)"
MRKC = "Mt. Rainier (Kautz Creek)"
MASC = "Mt. Adams (Salt Creek)"

deposit_list = [MMLR, GPSR, MRWR, MRKC, MASC]

#%% Create Functions

def add_combine_exposures(data,exposures,deposit_name):
    """
    This function filters specific rows from the input DataFrame `data` based on a list of `exposures`,
    changes the 'Stop ID' of the filtered rows to a new `deposit_name`, and appends the modified rows 
    back to the original DataFrame. The result is a combined entry for the selected exposures.
    
    Parameters:
    data: The original DataFrame containing at least a 'Stop ID' column.
    exposures: A list of 'Stop ID' values to filter the data.
    deposit_name: A string of the new 'Stop ID' value that will be assigned to the filtered rows.
    
    Returns:
    DataFrame: The modified DataFrame with the combined exposures.
    """
    deposit = data.loc[data['Stop ID'].isin(exposures)]
    deposit['Stop ID'] = deposit_name
    data= pd.concat([data,deposit],axis=0)
    
    return data 
#%%
# Define the unique lithology categories and repeat each 1000 times

lithology_category = [10, 20, 30, 40, 50, 60, 70, 80, 90]
lithology_values = lithology_category * 1000

# Create the Mean_Median SHRS values, same as Lithology Category
mean_median_shrs_values = lithology_values

# Create a DataFrame
PerSHRS = pd.DataFrame({
    'Lithology Category': lithology_values,
    'Mean_Median SHRS': mean_median_shrs_values
})

# Convert 'Lithology Category' column to strings
PerSHRS['Lithology Category'] = PerSHRS['Lithology Category'].astype(str)

#%%
#Prep GSD
# prep suiattle gsd df for merge with 2023 data
suiattleGSD_data=suiattleGSD_data.drop(columns=('Row Number'))
suiattleGSD_data=suiattleGSD_data.rename(columns={"Stop Number": "Stop ID"})
suiattleGSD_data['Stop ID'] = suiattleGSD_data['Stop ID'].astype("string")

# set up for suiattle %fines
suiattleGSD_data.loc[(suiattleGSD_data['Stop ID'].isin( ['7.0', '7.5','9.0','10.0','11.0'])) & (suiattleGSD_data['Lithology Category'].isin(['F', 'S', 'sand'])), 'Size (cm)'] = 'F'


# merge 2023 and suiattle data
gsd_data= pd.concat([field_data,suiattleGSD_data],axis=0)

#Set all extra shrs measurements to nan
gsd_data.drop(gsd_data[gsd_data['Extra SHRS'] == 'y'].index, inplace = True)

# combine exposures as needed for plotting
gsd_data= add_combine_exposures(gsd_data,exposures=['KC1','KC2','KC3','KC4','KC5','KC6','KC7'],
                                deposit_name= MRKC)

gsd_data= add_combine_exposures(gsd_data,exposures=['T5A','T5B'],
                                deposit_name="T5")

gsd_data= add_combine_exposures(gsd_data,exposures=['T4','T5','T6','T7','T8'],
                                deposit_name= MRWR)

gsd_data= add_combine_exposures(gsd_data,exposures=['MM1','MM2','MM3','MM4','MM5','MM6','MM7','MM8','MM9', 'MM10', 'MM11'],
                                deposit_name= MMLR)

gsd_data= add_combine_exposures(gsd_data,exposures=['MA1','MA1B','MA2','MA2B','MA3','MA3B'],#removed 4A/4B
                                deposit_name= MASC)

gsd_data= add_combine_exposures(gsd_data,exposures=['7.0', '7.5','9.0','10.0','11.0'],
                                deposit_name= GPSR)

# set up for %fines
gsd_data.loc[(gsd_data['Stop ID'] == 'Suiattle River') & (gsd_data['Lithology Category'].isin(['F', 'S', 'sand'])), 'Size (cm)'] = 'F'

# Drop fines and NA vals
gsd_data = gsd_data.drop(gsd_data[gsd_data['Size (cm)'] == 'F'].index)
gsd_data = gsd_data.drop(gsd_data[gsd_data['Size (cm)'] == '0'].index) #2809
gsd_data = gsd_data[gsd_data['Size (cm)'].notna()]

# Change grain size to meters
gsd_data[["Size (cm)"]] = gsd_data[["Size (cm)"]].apply(pd.to_numeric)/100
gsd_data=gsd_data.rename(columns={"Size (cm)": "Size (m)"})


#%%


# Run abrasion code for given uniform SHRS values and MRWR GSD 
deposit_list = ['10', '20', '30', '40', '50', '60', '70', '80', '90']
GSD = MRWR #can choose any deposit GSD: MMLR, GPSR, MRWR, MRKC, MASC
test_data = []

# Iterate through each deposit_name
for idx, deposit_name in enumerate(deposit_list):
    # set random seed to 5
    np.random.seed(5) 
    lith_list = [deposit_name]

    gsd_df = gsd_data.loc[gsd_data['Stop ID'] == GSD].copy()
    
    # pull the GSD data
    D_source = np.array(gsd_df['Size (m)'].values)
    #D_source = [.1] #very interesting to watch what it does with a single grain size value

    SHRS = PerSHRS.loc[PerSHRS['Lithology Category'] == deposit_name, 'Mean_Median SHRS'].values
    lith = PerSHRS.loc[PerSHRS['Lithology Category'] == deposit_name, 'Lithology Category'].values
    
    k = 15  #UNKNOWN coefficient in transport dependent abrasion equation. mimics Attal and Lave conceptual diagram
    alphamultiplier = 2 # UNKNOWN tumbler correction value, literature suggests something in this range.
    
    # PART 1: Simple Sternberg abrasion calculation, based on our data. # updated with all 2024 combined data
    measured_alphas = alphamultiplier * np.exp(-1.1522292535105734) * np.exp(-0.08769532043420453 * SHRS) # original suiattle: alphamultiplier * 3.0715 * np.exp(-0.136 * SHRS)
    density = 608.6545032394779 * np.log(SHRS) + (77.75350574024881) # original suiattle: 923.63 * np.log(SHRS) + (- 1288.2)

    dist_km = np.arange(0, 101) # should be 101 # array to represent distance downstream from source
    Sternb_mass_abradingsourcematerial = np.exp(-np.expand_dims(measured_alphas, axis=0) * (np.expand_dims(dist_km, axis=1))) # classic abrasion calculation 
    
    Sternb_volume_abradingsourcematerial = (Sternb_mass_abradingsourcematerial / density)
    Sternb_NormVolume_downstream = np.sum(Sternb_volume_abradingsourcematerial, axis=1) / np.sum(Sternb_volume_abradingsourcematerial[0, :])
    Sternb_NormMass_downstream = np.sum(Sternb_mass_abradingsourcematerial, axis=1) / np.sum(Sternb_mass_abradingsourcematerial[0, :])

    d_alpha = 0.01 # np.median(measured_alphas)/3
    D_50 = 0.14 * np.exp(-d_alpha * dist_km)
    
    # THE NEXT STEP: low density grains higher transport rate--> higher abrasion rate
    
    n = 1 # number of repeated distributions, SET TO 1 FOR WEIGHTED VERSION BECAUSE INPUT IS 1000 SHRS AND LITH ALREADY
    density_n = 608.6545032394779 * np.log(np.tile(SHRS, n)) + (77.75350574024881)  # original: 923.63 * np.log(np.tile(SHRS, n)) + (-1288.2) # NOTE: this regression ignores 2 VV outlier points. Submerged density (to match mass measurements)
    SHRS_n = np.tile(SHRS, n)
    lith_n = np.tile(lith, n)
    D_matrix = np.full([np.size(dist_km), n * np.size(SHRS)], 0.) 
    Mass_matrix = np.full([np.size(dist_km), n * np.size(SHRS)], 0.)
    Vol_matrix = np.full([np.size(dist_km), n * np.size(SHRS)], 0.) 
    
    D_matrix[0, :] = np.random.choice(D_source, np.shape(SHRS_n)) # randomly sample from the grain size data..
    Vol_matrix[0, :] = 1.0 # all 1.0 m3 "parcels"
    Mass_matrix[0, :] = Vol_matrix[0, :] * density_n
    
    # ASSUMPTION: we're going to abrade these without considering selective transport. Represents a single upstream source (not terrible here), with no channel storage/aggradation/lag

    tauS_50 = 0.045 # a representative Shields stress for transport of D50..
    rho = 1000
    g = 9.81
    tau = np.full(np.shape(dist_km), fill_value=np.nan)
    mean_SHRS = np.full(np.shape(dist_km), fill_value=np.nan)
    
    # Calculate abrsion at each downstream distance
    
    for i in range(np.size(dist_km[1:])):
        # let's assume that a representative tau is twice that required to transport D50 of mean density
        tau[i] = 2.0 * tauS_50 * ((np.mean(density_n) - rho) * g * D_50[i])
        tauS = tau[i] / ((density_n - rho) * g * D_matrix[i, :])
        tauS_tauSc = tauS / tauS_50

        alpha = alphamultiplier * np.exp(-1.1522292535105734) * np.exp(-0.08769532043420453 * SHRS_n) # the constant alpha # original: alphamultiplier * 3.0715 * np.exp(-0.136 * SHRS_n)
        
        alpha_HighTransport = alpha * (k * (density_n - rho) / rho * D_matrix[i, :] * (tauS_tauSc - 3.3) + 1)
        
        alpha[tauS_tauSc > 3.3] = alpha_HighTransport[tauS_tauSc > 3.3]
        alpha[D_matrix[i, :] < 0.002] = 0 # don't abrade sand
        
        # Now, 
        Mass_matrix[i + 1, :] = Mass_matrix[i, :] * np.exp(-(dist_km[i + 1] - dist_km[i]) * alpha)
        D_matrix[i + 1, :] = D_matrix[i, :] * ((Mass_matrix[i + 1, :] / density_n) / (Mass_matrix[i, :] / density_n)) ** (1 / 3)
        
        # Volumetrically weighted mean strength at each distance ds
        vol_fraction = (Mass_matrix[i, :] / density_n) / np.sum(Mass_matrix[i, :] / density_n)
        mean_SHRS[i] = np.sum(SHRS_n * vol_fraction)

    Vol_matrix = Mass_matrix / density_n
    Vol_matrix[D_matrix <= 0.002] = 0 #When we measure bed material, we don't measure sand. 
    mass_norm = np.sum(Mass_matrix, axis=1) / np.sum(Mass_matrix[0, :])
    vol_norm = np.sum(Vol_matrix, axis=1) / np.sum(Vol_matrix[0, :])
    
    # How does lith by volume change?
    # Dictionary to store results
    lith_vol = {}
    vol_tot = np.sum(Vol_matrix, axis=1) #total volume of grains at each distance
    # Loop over the lithologies and calculate sums
    for lith_type in lith_list:
        lith_vol[f'vol_{lith_type}'] = np.sum(Vol_matrix[:, np.tile(lith, n) == lith_type], axis=1)
    
    # Plot
    vol_initial = np.sum(Vol_matrix[0, :])
    vol_lith_products = 0
    for lith_type in lith_list:
        vol_lith_products += lith_vol[f'vol_{lith_type}']
    vol_products = vol_initial - (vol_lith_products)
    
    # Create a list of values where each value in lith_vol is divided by vol_initial
    y_data = [lith_vol[f'vol_{lith_type}'] / vol_initial for lith_type in lith_list] + [vol_products / vol_initial]
    labels = lith_list
    
    test_data += ([lith_vol[f'vol_{lith_type}'] / vol_initial for lith_type in lith_list])

    print(1-(vol_products[100]/1000))
    
    colors = []
    for color, lithologies in color_dict.items():
        if deposit_name in lithologies:
            colors.append(color)
            break
    new_colors = colors.copy()
    new_colors.append('white')
    
#%%
# Plot Figure 17: Modeled abrasion of pulses with uniform strength, using the MRWR GSD.
color_dict = {
    '10': (0.9, 0.9, 0.9),
    '20': (0.8, 0.8, 0.8),
    '30': (0.7, 0.7, 0.7),
    '40': (0.6, 0.6, 0.6),
    '50': (0.5, 0.5, 0.5),
    '60': (0.4, 0.4, 0.4),
    '70': (0.3, 0.3, 0.3),
    '80': (0.2, 0.2, 0.2),
    '90': (0.1, 0.1, 0.1)
}

# X data
x_data = np.arange(0, 101)

# Create plot
plt.figure(figsize=(4.3, 3))

for i, data in enumerate(test_data):
    color = color_dict[deposit_list[i]]
    plt.plot(x_data, data, label=deposit_list[i],color = color )
    
# Adding labels and title
plt.xlabel('Distance from source (km)')
plt.ylabel('Volumetric fraction remaining')
plt.ylim(0, 1)
plt.xlim(0, 100)
#plt.title('Line Plot of Arrays in Test Data')
#plt.legend(loc='upper right')

# Show plot
plt.tight_layout()
plt.savefig('abrasion_PerSHRS', dpi=400, bbox_inches="tight")
plt.show()
