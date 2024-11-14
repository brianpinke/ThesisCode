#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 13:04:20 2024

This code models transport-dependent abrasion of a sediment source of given
strength and grain size distribution.

@author: Brian Pinke **Modified From Allison Pfeiffer

# Plot Figure 12: Modeled abrasion of the five study deposits sediment over 100 km of downstream transport. 
# The products of abrasion are shown in white. Starting and ending mean SHRS of sediment 
# denoted in white on left and right corners. Modified from Pfeiffer et al. (2022).
# OR Plot Figure 14: Modeled Abrasion for (A) all small GSD (MA-SC) and (B) all large GSD (GP-SR).

Define the active GSD used as the "Standard" for each deposit, or all "Large" (GPSR), or all "Small" (MASC)
Figure 12 = "Standard, Figure 14A = "Small", Figure 14B = "Large"

activeGSD = "Standard" #Standard, Large, or Small

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


# define colors
LightGray = (0.8274509803921568, 0.8274509803921568, 0.8274509803921568)
MediumGray = (0.6627450980392157, 0.6627450980392157, 0.6627450980392157)
DarkGray = (0.4117647058823529, 0.4117647058823529, 0.4117647058823529)

lightpurple= (0.7450980392156863, 0.7058823529411765, 0.9019607843137255)
darkpurple= (0.5019607843137255, 0.0, 0.5019607843137255)
orange= (1.0, 0.5294117647058824, 0.0)
pink=(1.0, 0.7529411764705882, 0.796078431372549)
green= (0.0, 0.5019607843137255, 0.0)
lighterorange = (1.0, 0.7254901960784313, 0.30196078431372547)

color_dict = {
    lightpurple: ["LDV", "VV", "TV", "VN", "V"],
    darkpurple: ["HDV", "NV", "NN", "PN"],
    orange: ["WW", "NA", "OW"],
    pink: ["G", "PL"],
    green: ["HGM"],
    lighterorange: ["VA"]
}

# change font size
plt.rcParams["font.family"] = "Arial"
plt.rcParams['font.size'] = 10

# set random seed to 5
np.random.seed(5) 

os.chdir(os.getcwd()+'/Figs')

# Define the active GSD used as the "Standard" for each deposit, 
# or all "Large" (GPSR), or all "Small" (MASC)
# Figure 12 = "Standard, Figure 14A = "Small", Figure 14B = "Large"
activeGSD = "Standard" #Standard, Large, or Small

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

# Calculate the empirical cumulative distribution function (ECDF)
def ecdf(data):
    """
    Calculate the empirical cumulative distribution function (ECDF) for a given dataset.

    Parameters:
    data: array-like
        A sequence of numerical values for which the ECDF will be computed.

    Returns:
    x: numpy.ndarray
        Sorted values of the input data.
    y: numpy.ndarray
        Cumulative probabilities corresponding to the sorted values.
    """
    # Sort the data in ascending order
    x = np.sort(data)
    # Compute the cumulative probabilities for each data point
    y = np.arange(1, len(data) + 1) / len(data)
    return x, y

def create1000SHRSsample(deposit_name, lith_percent_df):
    """
    Create a sample DataFrame of 1000 Schmidt Hammer Rock Strength (SHRS) values
    based on lithology percentages using empirical cumulative distribution functions (ECDFs).

    Parameters:
    deposit_name: str
        The name of the deposit for which the SHRS sample is created.
    lith_percent_df: pandas.DataFrame
        A DataFrame containing lithology types and their corresponding fractions.
    SHRS_df: 
        A DataFrame containing lithology types and corresponding strengths.

    Returns:
    new_df: pandas.DataFrame
        A DataFrame containing 1000 SHRS values categorized by lithology.
    """
    # Creating the new DataFrame with a dynamic name
    new_df = pd.DataFrame(columns=SHRS_data.columns)
    
    for lithtype in lith_percent_df['Lithologies']:
        # Calculate ECDF for the original data, filter for D > 25cm and desired lithtype
        x_original, y_original = ecdf(
            SHRS_data.loc[
                (SHRS_data['Stop ID'] == deposit_name) & 
                (SHRS_data['Lithology Category'] == lithtype) & 
                ((SHRS_data['Extra SHRS'] == 'y') | 
                 (SHRS_data['Size (cm)'] >= 25) | 
                 (SHRS_data['Stop ID'] == GPSR))
                ]['Mean_Median SHRS']
            )
        
        # Generate xnumber random samples from a uniform distribution between 0 and 1
        num_samples = int(lith_percent_df.loc[(lith_percent_df['Lithologies'] == lithtype)]['Fraction'].iloc[0])
        random_uniform_samples = np.random.rand(num_samples) #THIS IS A NUMBER THAT SHOULD BE THE FRAC FOR THAT LITHOLOGY OUT OF 1000
        
        # Map the random samples to strength values based on the original ECDF
        mapped_strengths = np.interp(random_uniform_samples, y_original, x_original)
        
        # Convert the array to a DataFrame
        mapped_df = pd.DataFrame(mapped_strengths, columns=['Mean_Median SHRS'])
        
        # add lithtype to the Lithology Category column
        mapped_df['Lithology Category'] = lithtype
        
        # Concatenate the data_df with the test DataFrame
        new_df = pd.concat([new_df, mapped_df], ignore_index=True)

    new_df['Stop ID'] = deposit_name    

    return new_df

#%%

# Prep DFs for use
field_data2023=field_data.copy()

# Convert 'Size (cm)' to numeric, coercing errors to NaN
field_data2023['Size (cm)'] = pd.to_numeric(field_data2023['Size (cm)'], errors='coerce')

# Filter the DataFrame
field_data2023 = field_data2023.loc[(field_data2023['Extra SHRS'] == 'y') | (field_data2023['Size (cm)'] >= 25)]

#prep suiattle liths
merge_liths = { "NP": 'NV', 'GF': 'PL', 'VNV':'NV', 'VC':'VB', "NAL":"NA", "VB":'VV', 'UG':'WW', 'G':'PL'} # Define replacements {old_lith: new_lith}

# group suiattle G and P liths to PL and VV
suiattleSHRS_data.loc[suiattleSHRS_data['Lithology'] == 'G', 'Lithology'] = "PL"
suiattleSHRS_data.loc[suiattleSHRS_data['Lithology'] == 'P', 'Lithology'] = "VV"

suiattleSHRS_data = suiattleSHRS_data[suiattleSHRS_data['Lithology'] != 'MS']

#change suiattle column names for merging with 2023 field data
suiattleSHRS_data=suiattleSHRS_data.rename(columns={"Site Num": "Stop ID", 'Lithology': 'Lithology Category','SHRS median': 'Mean_Median SHRS'})
suiattleSHRS_data['Stop ID'] = suiattleSHRS_data['Stop ID'].astype("string")

# merge 2023 and suiattle data to 1 df and drop any empty data
SHRS_data= pd.concat([field_data2023,suiattleSHRS_data],axis=0)
SHRS_data= SHRS_data[SHRS_data['Mean_Median SHRS'].notna()]

for old_lith, new_lith in merge_liths.items():
    SHRS_data.loc[SHRS_data['Lithology Category'] == old_lith, 'Lithology Category'] = new_lith
    
    
# change values of NAL to NA - had to be entered into excel as NAL as NA would appear as empty to python
SHRS_data.loc[SHRS_data['Lithology Category'] == 'NAL', 'Lithology Category'] = 'NA'

#%%
# combine exposures as needed for modeling
         
SHRS_data= add_combine_exposures(SHRS_data,exposures=['KC1','KC2','KC3','KC4','KC5','KC6','KC7'],
                                deposit_name= MRKC)

SHRS_data= add_combine_exposures(SHRS_data,exposures=['T5A','T5B'],
                                deposit_name="T5")

SHRS_data= add_combine_exposures(SHRS_data,exposures=['T4','T5','T6','T7','T8'],
                                deposit_name= MRWR)

SHRS_data= add_combine_exposures(SHRS_data,exposures=['MM1','MM2','MM3','MM4','MM5','MM6','MM7','MM8','MM9', 'MM10', 'MM11'],
                                deposit_name= MMLR)

SHRS_data= add_combine_exposures(SHRS_data,exposures=['MA1','MA1B','MA2','MA2B','MA3','MA3B'],#removed 4/4A
                                deposit_name= MASC)

SHRS_data= add_combine_exposures(SHRS_data,exposures=['9','10'],
                                deposit_name= GPSR)

#%%
# Lithology fractions for each deposit, obtained from PieChartFig script analysis
# create dfs with the lithology and corresponding fraction for each deposit

# Data for Meager
meager_df = pd.DataFrame({
    'Lithologies': ['LDV', 'PL', 'HDV', 'OW', 'HGM'],
    'Fraction': [62.150056, 29.339306, 4.255319, 3.023516, 1.231803]})

meager_df['Fraction'] = (meager_df['Fraction'] * 10).round(0)

# Data for Suiattle
suiattle_df = pd.DataFrame ({
    'Lithologies': ['VV', 'NV'], #PL
    'Fraction': [56.959707, 43.040293]}) #split PL 1.831502 evenly to VV and NV

suiattle_df['Fraction'] = (suiattle_df['Fraction'] * 10).round(0)

# Data for Tahoma
tahoma_df = pd.DataFrame ({
    'Lithologies': ['PN', 'TV', 'VV', 'WW'],
    'Fraction': [39.344262, 31.876138, 19.125683, 9.653916]})

tahoma_df['Fraction'] = (tahoma_df['Fraction'] * 10).round(0)

# Data for Kautz
kautz_df = pd.DataFrame({
    'Lithologies': ['NV', 'V', 'PL'],
    'Fraction': [89.189189, 9.009009, 1.801802]})

kautz_df['Fraction'] = (kautz_df['Fraction'] * 10).round(0)

# Data for Adams
adams_df = pd.DataFrame({
    'Lithologies': ['NA', 'NN', 'VN', 'VA'],
    'Fraction': [49.846154, 29.230769, 11.076923, 9.846154]})

adams_df['Fraction'] = (adams_df['Fraction'] * 10).round(0)
adams_df.at[2, 'Fraction'] += 1
#%%
# create 1000 weighted shrs dfs
Meager_weightedSHRS_distrib = create1000SHRSsample(MMLR, meager_df, )
Suiattle_weightedSHRS_distrib = create1000SHRSsample(GPSR, suiattle_df, )
Tahoma_weightedSHRS_distrib = create1000SHRSsample(MRWR, tahoma_df, )
Kautz_weightedSHRS_distrib = create1000SHRSsample(MRKC, kautz_df, )
Adams_weightedSHRS_distrib = create1000SHRSsample(MASC, adams_df, )

#combine weighted dfs into one df
weighteddistrib_SHRS_data= pd.concat([Meager_weightedSHRS_distrib, Suiattle_weightedSHRS_distrib, Tahoma_weightedSHRS_distrib, Kautz_weightedSHRS_distrib, Adams_weightedSHRS_distrib], ignore_index=True)

#%%
#Prep GSD
# prep suiattle gsd df for merge with 2023 data
suiattleGSD_data=suiattleGSD_data.drop(columns=('Row Number'))
suiattleGSD_data=suiattleGSD_data.rename(columns={"Stop Number": "Stop ID"})
suiattleGSD_data['Stop ID'] = suiattleGSD_data['Stop ID'].astype("string")

# set up for suiattle %fines
suiattleGSD_data.loc[(suiattleGSD_data['Stop ID'].isin( ['7.0', '7.5','9.0','10.0','11.0'])) & (suiattleGSD_data['Lithology Category'].isin(['F', 'S', 'sand'])), 'Size (cm)'] = 'F'

# merge 2023 field data and suiattle data
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
# Plot Figure 12: Modeled abrasion of the five study deposits sediment over 100 km of downstream transport. 
# The products of abrasion are shown in white. Starting and ending mean SHRS of sediment 
# denoted in white on left and right corners. Modified from Pfeiffer et al. (2022).
# OR Plot Figure 14: Modeled Abrasion for (A) all small GSD (MA-SC) and (B) all large GSD (GP-SR).

# Define figure and subplots
fig, axs = plt.subplots(3, 2, figsize=(6.5, 9), sharey=True) # could be True or False

# Flatten axs array for easy indexing
axs = axs.flatten()

# Desired order of lithologies
desired_order = ['PL', 'HGM', 'NN', 'NV', 'PN', 'HDV', 'TV', 'VN', 'V', 'VV', 'LDV', 'WW', 'OW', 'NA', 'VA']

# Iterate through each deposit_name
for idx, deposit_name in enumerate(deposit_list):
    if activeGSD == "Standard":
        GSD = deposit_name
    if activeGSD == "Large":
        GSD = GPSR
    if activeGSD == 'Small':
        GSD = MASC
        
    # set random seed to 5
    np.random.seed(5) 
    lith_list = weighteddistrib_SHRS_data.loc[weighteddistrib_SHRS_data['Stop ID'] == deposit_name, 'Lithology Category'].unique()
    
    # Reorder lith_list based on desired_order
    lith_list = [lith for lith in desired_order if lith in lith_list]

    gsd_df = gsd_data.loc[gsd_data['Stop ID'] == GSD].copy() #* should be == deposit_name , can replace with 'MMLR' or any deposit gsd
   
    # pull the GSD data
    D_source = np.array(gsd_df['Size (m)'].values)
    #D_source = [.01] #very interesting to watch what it does with a single grain size value

    SHRS = weighteddistrib_SHRS_data.loc[weighteddistrib_SHRS_data['Stop ID'] == deposit_name, 'Mean_Median SHRS'].values
    lith = weighteddistrib_SHRS_data.loc[weighteddistrib_SHRS_data['Stop ID'] == deposit_name, 'Lithology Category'].values

    #k_line_style = ['k-', 'k--', 'k:']
    #letter = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    #letter_now = 0
    
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
    
    #print('mean at 0km is: ', round(min(mean_SHRS),1),' and mean at end is: ', round(max(mean_SHRS),1))
    # How does lith by volume change?
    # Dictionary to store results
    lith_vol = {}
    vol_tot = np.sum(Vol_matrix, axis=1) #total volume of grains at each distance
    # Loop over the lithologies and calculate sums
    for lith_type in lith_list:
        lith_vol[f'vol_{lith_type}'] = np.sum(Vol_matrix[:, np.tile(lith, n) == lith_type], axis=1)
        #print(lith_type, (lith_vol[f'vol_{lith_type}'][-1])/)
    
    vol_initial = np.sum(Vol_matrix[0, :])
    vol_lith_products = 0
    for lith_type in lith_list:
        vol_lith_products += lith_vol[f'vol_{lith_type}']
        
    vol_products = vol_initial - (vol_lith_products)
    
    # Create a list of values where each value in lith_vol is divided by vol_initial
    y_data = [lith_vol[f'vol_{lith_type}'] / vol_initial for lith_type in lith_list] + [vol_products / vol_initial]
    labels = lith_list
    

    print(1-(vol_products[100]/1000))
    
    colors = []
    for lithology in lith_list:
        for color, lithologies in color_dict.items():
            if lithology in lithologies:
                colors.append(color)
                break
    new_colors = colors.copy()
    new_colors.append('white')

    axs[idx].stackplot(dist_km, *y_data, labels=labels, colors=new_colors) #, edgecolor='black', linewidth=1 looks pretty bad though....
    axs[idx].set_ylim(0, 1)
    axs[idx].set_xlim(0, 100) #should be 100
    
    # Set the title inside the subplot box
    #axs[idx].set_title(deposit_name, loc='center', y=0.90)  #adjust y to place below line
    axs[idx].set_title(deposit_name, loc='center', y=0.90, fontdict={'family': 'Arial', 'size': 10})
    
    # Add start and end mean SHRS value to the plot
    axs[idx].text(x=6, y=0.01, s=round(min(mean_SHRS),1), color='white', ha='center')
    axs[idx].text(x=94, y=0.01, s=round(max(mean_SHRS),1), color='white', ha='center')
   
    
   # axs[idx].set_title(deposit_name)
    axs[idx].set_xlabel('Distance from source (km)')
    #axs[idx].legend(loc='upper right')
    axs[idx].grid(False)
    if idx % 2 == 0:
        axs[idx].set_ylabel('Volumetric fraction remaining')

# Hide any unused subplots
for ax in axs[len(deposit_list):]:
    ax.axis('off')

plt.tight_layout()
plt.savefig('abrasionmodelfig.png', dpi=400, bbox_inches="tight")
plt.show()

