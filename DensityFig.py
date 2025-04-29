#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 13:04:05 2023

@author: bpinke

# Plot Figure 11: (A) Density of each deposit as random sampled per exposure. (B) Density of each deposit weighted by lithology fraction, n for each is 100.
"""

# loads necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy.stats import kruskal
import numpy as np
import pandas as pd
import os

# load in 2023 density and abrasion data
path=os.path.join( os.getcwd(), 'Data/PinkeTumbling2023_USE.xlsx')

deposit_density = pd.read_excel (
        path, 
        sheet_name='AllDensity', 
        header = 0)

# define colors
meagercolor = (0.4, 0.2, 0.0) #dark brown
glaciercolor = (0.0, 0.5019607843137255, 0.5019607843137255) #teal
tahomacolor = (0.4, 0.6, 0.8) #steel blue
kautzcolor = (0.8627450980392157, 0.0784313725490196, 0.23529411764705882) #crimson
adamscolor = (0.5019607843137255, 0.5019607843137255, 0.0) #olive

# change text font and size
plt.rcParams["font.family"] = "Arial"
plt.rcParams['font.size'] = 10

# set random seed to 5
np.random.seed(5) 

# change the directory so figs are saved in proper location
os.chdir(os.getcwd()+'/Figs')

#%%
# define functions

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

def create100DensitySample(deposit_name, lith_percent_df, density_df):
    """
    Create a sample DataFrame of 100 Density values
    based on lithology percentages using empirical cumulative distribution functions (ECDFs).

    Parameters:
    deposit_name: str
        The name of the deposit for which the Density sample is created.
    lith_percent_df: pandas.DataFrame
        A DataFrame containing lithology types and their corresponding fractions.
    density_df: 
            A DataFrame containing lithology types and corresponding densities.

    Returns:
    new_df: pandas.DataFrame
        A DataFrame containing 100 Density values categorized by lithology.
    """
    # Creating the new DataFrame with a dynamic name
    new_df = pd.DataFrame(columns=density_df.columns)
    
    for lithtype in lith_percent_df['Lithologies']:
        # Calculate ECDF for the original data
        x_original, y_original = ecdf(
            density_df.loc[
                (density_df['Deposit'] == deposit_name) & 
                (density_df['Lithology'] == lithtype)]
                ['Density (kg/m3)']
                )
        
        # Generate xnumber random samples from a uniform distribution between 0 and 1
        num_samples = int(lith_percent_df.loc[(lith_percent_df['Lithologies'] == lithtype)]['Fraction'].iloc[0])
        random_uniform_samples = np.random.rand(num_samples) #THIS IS A NUMBER THAT SHOULD BE THE FRAC FOR THAT LITHOLOGY OUT OF 1000
        
        # Map the random samples to strength values based on the original ECDF
        mapped_strengths = np.interp(random_uniform_samples, y_original, x_original)
        
        # Convert the array to a DataFrame
        mapped_df = pd.DataFrame(mapped_strengths, columns=['Density (kg/m3)'])
        
        # add lithtype to the Lithology Category column
        mapped_df['Lithology'] = lithtype
        
        # Concatenate the data_df with the test DataFrame
        new_df = pd.concat([new_df, mapped_df], ignore_index=True)

    new_df['Deposit'] = deposit_name    

    return new_df

def run_density_violin_by_plot(df, by, data, colors, title=None, ax=None):
    """
    df: df of choice
    by: string of column name you want to group the data by for plotting ("Stop ID" or "Lithology Category")
    data: a list of strings of the categories you want plotted. Eg the Stop ID names or the Lithology Categories of interest. 
    colors: list of colors corresponding to each category
    title: default is none, give a string if desired.
    ax: matplotlib axis object to plot on. If None, a new figure is created.
    
    Creates a violin plot of desired data, with count printed on fig and min, median, max printed in console.
    Performs Kruskal-Wallis test to check for statistical differences between the distributions.
    """
    # Use provided axis or create a new figure if ax is None
    if ax is None:
        plt.figure(figsize=(4.3, 3))  # inches
        ax = plt.gca()
    
    # Create a temporary DataFrame grouping the given df as desired and selecting for desired data
    datas = df.loc[df[by].isin(data)]
    
    # Create the violin plot
    sns.violinplot(data=datas, x=by, y='Density (kg/m3)', saturation=1, palette=colors, order=data, ax=ax)
    
    # Update x tick labels to include count for each category
    new_xticklabels = []
    for category in data:
        count = datas[datas[by] == category].shape[0]
        new_xticklabels.append(f'{category}\n(n={count})')
    
    ax.set_xticklabels(new_xticklabels)
    ax.set(ylabel=r'Clast density,$\rho_s$ (kg/m$^3$)', xlabel=None)
    
    # Perform Kruskal-Wallis test
    category_data = [datas[datas[by] == category]['Density (kg/m3)'] for category in data]
    kruskal_result = kruskal(*category_data)

    # Get the median, min, and max values for each category represented
    for category in data:
        category_data = datas[datas[by] == category]['Density (kg/m3)']
        median_val = category_data.median()
        min_val = category_data.min()
        max_val = category_data.max()

        # Print out min, max, and median values for each category
        print(f"{category} - Min: {min_val:.2f}, Max: {max_val:.2f}, Median: {median_val:.2f}")
        
    # Print Kruskal-Wallis test results
    print(f"Kruskal-Wallis H-test result: H-statistic = {kruskal_result.statistic:.2f}, p-value = {kruskal_result.pvalue:.4f}")

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['font.size'] = 10
   
    if ax is None:
        plt.savefig("Density" + title + ".png", dpi=400, bbox_inches="tight")
        plt.show()

#%%
# prep dfs for analysis

# remove P density samples from df, remove MA4 exposure samples as this data is thrown out
deposit_density = deposit_density[deposit_density['Deposit'] != 'P']
deposit_density = deposit_density[~((deposit_density['Deposit'] == 'MA') & (deposit_density['Sample'].between(27, 30)))]
deposit_names = {'Tahoma':'MR-WR', 'KC':'MR-KC', 'MM':'MM-LR', 'MA':'MA-SC'}#
deposit_density['Deposit'] = deposit_density['Deposit'].replace(deposit_names)

#%%

# Add missing lithology density data from the abrasion tumbler experiment to density
new_data = [
    {'Deposit': 'MM-LR', 'Lithology': 'HDV', 'Density (kg/m3)': 2372.395486},
    {'Deposit': 'MA-SC', 'Lithology': 'VA', 'Density (kg/m3)': 2065.337521}
]

# Create a new DataFrame from the new data
new_data_df = pd.DataFrame(new_data)

# Concatenate the new DataFrame with the existing deposit_density DataFrame
model_deposit_density = pd.concat([deposit_density, new_data_df], ignore_index=True)

#merge liths appropriately:

merge_liths = { "NP": 'NV', 'GF': 'PL', 'VNV':'NV', 'VC':'VB', "NAL":"NA", "VB":'VV', 'UG':'WW', 'G':'PL'} # Define replacements {old_lith: new_lith}
for old_lith, new_lith in merge_liths.items():
    model_deposit_density.loc[model_deposit_density['Lithology'] == old_lith, 'Lithology'] = new_lith

# Lithology fractions for each deposit, obtained from PieChartFig script analysis
# Create dfs with the lithology and corresponding fraction for each deposit

# Data for Meager
meager_df = pd.DataFrame({
    'Lithologies': ['LDV', 'PL', 'HDV', 'OW', 'HGM'], # don't have HDV in deposit density, added from shrs/density measurements
    'Fraction': [62.150056, 29.339306, 4.255319, 3.023516, 1.231803]})

meager_df['Fraction'] = (meager_df['Fraction'] * 1).round(0)
meager_df.at[2, 'Fraction'] += 1

# Data for Tahoma
tahoma_df = pd.DataFrame ({
    'Lithologies': ['PN', 'TV', 'VV', 'WW'],
    'Fraction': [39.344262, 31.876138, 19.125683, 9.653916]})

tahoma_df['Fraction'] = (tahoma_df['Fraction'] * 1).round(0)

# Data for Kautz
kautz_df = pd.DataFrame({
    'Lithologies': ['NV', 'V', 'PL'],
    'Fraction': [89.189189, 9.009009, 1.801802]})

kautz_df['Fraction'] = (kautz_df['Fraction'] * 1).round(0)

# Data for Adams
adams_df = pd.DataFrame({
    'Lithologies': ['NA', 'NN', 'VN', 'VA'], # Don't have VA in deposit density, added from shrs/density measurements
    'Fraction': [49.846154, 29.230769, 11.076923, 9.846154]})

adams_df['Fraction'] = (adams_df['Fraction'] * 1).round(0)

#%%
# create 1000 weighted shrs dfs
Meager_weightedDensity_distrib = create100DensitySample('MM-LR', meager_df, model_deposit_density)
Tahoma_weightedDensity_distrib = create100DensitySample('MR-WR', tahoma_df, model_deposit_density)
Kautz_weightedDensity_distrib = create100DensitySample('MR-KC', kautz_df, model_deposit_density)
Adams_weightedDensity_distrib = create100DensitySample('MA-SC', adams_df, model_deposit_density)

#combine weighted dfs into one df
weighteddistrib_density_data= pd.concat([Meager_weightedDensity_distrib, Tahoma_weightedDensity_distrib, Kautz_weightedDensity_distrib, Adams_weightedDensity_distrib], ignore_index=True)

# Use desired deposit names
# Define the mapping of values to be replaced
replace_dict = {
    "MM-LR": "Lillooet R.",
    "MR-WR": "White R.",
    "MR-KC": "Kautz C.",
    "MA-SC": "Salt C.",
}

# Use the replace function to update the values in the "Deposit" column
deposit_density['Deposit'] = deposit_density['Deposit'].replace(replace_dict)
weighteddistrib_density_data['Deposit'] = weighteddistrib_density_data['Deposit'].replace(replace_dict)

#%%
# Plot Figure 11: (A) Density of each deposit as random sampled per exposure. (B) Density of each deposit weighted by lithology fraction, n for each is 100.
# Create a figure with two subplots
fig, axs = plt.subplots(2, 1, figsize=(4.5, 7), sharey=True)

# Plot on the first subplot
run_density_violin_by_plot(deposit_density, 
                           by="Deposit", 
                           data=['Lillooet R.', 'White R.', 'Kautz C.', 'Salt C.'], 
                           colors=[meagercolor, tahomacolor, kautzcolor, adamscolor], 
                           title='Density by Deposit',
                           ax=axs[0])

# Plot on the second subplot
run_density_violin_by_plot(weighteddistrib_density_data, 
                           by="Deposit", 
                           data=['Lillooet R.', 'White R.', 'Kautz C.', 'Salt C.'], 
                           colors=[meagercolor, tahomacolor, kautzcolor, adamscolor], 
                           title='Density Weighted by Deposit',
                           ax=axs[1])

# Adjust layout to avoid overlap
plt.tight_layout()

plt.savefig("Density" + 'ObservedWeighted' + ".png", dpi=400, bbox_inches="tight")
# Show the plot
plt.show()
