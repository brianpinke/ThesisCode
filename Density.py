#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 13:04:05 2023

@author: bpinke
"""

# loads necessary libraries
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np
import pandas as pd
import os


# load in 2023 density and abrasion data
path=os.path.join( os.getcwd(), 'Data/PinkeTumbling2023_USE.xlsx')

density_abrasion_2023 = pd.read_excel (
        path, 
        sheet_name='Density', 
        header = 0)

# load just 2023 abrasion data
abr_data_2023 = pd.read_excel (
        path, 
        sheet_name='Abrasion', 
        header = 0)

# load in Suiattle SHRS and abrasion
path=os.path.join( os.getcwd(), 'Data/SHRS_Abrasion_Data.xlsx')

suiattle_data = pd.read_excel (
        path, 
        sheet_name='ToPython', 
        header = 0)

# load in Finn Coffin Abrasion Density data
path=os.path.join( os.getcwd(), 'Data/FinnTumblingData.xlsx')

finn_data = pd.read_excel (
        path, 
        sheet_name='AbrasionDensity', 
        header = 0)

# define colors
lightpurple= (0.7450980392156863, 0.7058823529411765, 0.9019607843137255)
darkpurple= (0.5019607843137255, 0.0, 0.5019607843137255)
orange= (1.0, 0.5294117647058824, 0.0)
pink=(1.0, 0.7529411764705882, 0.796078431372549)
green= (0.0, 0.5019607843137255, 0.0)
lighterorange =(1.0, 0.7254901960784313, 0.30196078431372547)

# change font size
plt.rcParams.update({'font.size': 14})
#plt.rcParams['pdf.fonttype'] = 42
#plt.rcParams['ps.fonttype'] = 42

# change the directory so figs are saved in proper location
os.chdir(os.getcwd()+'/Figs')

#%%
# define functions

def run_density_shrs_plot(data,title=None):
    """
    data: df of interest
    title= default is none, give string if desired
    
    Choose data and title to plot a density vs shrs plot, grouped by "Lith" category

    """

    fig, ax = plt.subplots()
    #exponential fit
    a,b = np.polyfit(np.log(data['SHRS median'][:-2]),data['Density (kg/m3)'][:-2],1)
    x_plotfit = np.linspace(20,80,20)
    y_plotfit = a*np.log(x_plotfit) + b
    print(title, "density best fit b:", b, "m:", a)
    
    plt.plot(x_plotfit,y_plotfit,'-',color = 'grey',zorder = 1)
    
    
    colors = {'VV': lightpurple, 'VN': lightpurple, "LDV": lightpurple, "TV":lightpurple, "NA": orange, "HDV": darkpurple, "NV":darkpurple, "NN": darkpurple, "PN": darkpurple, "G":pink, "PL":pink, "UG":orange, "VA":lighterorange}
    grouped = data.groupby('Lith')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', y='Density (kg/m3)', x='SHRS median',  color=colors[key], label=key)

    plt.ylabel(r'Clast density,$\rho_s$ (kg/m$^3$)')
    plt.xlabel('Schmidt Hammer Rock Strength')
    plt.ylim(1500, 2800)
    plt.title(title)
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    
    plt.savefig("Density"+title+".png", dpi=400, bbox_inches="tight")
    plt.show()

def run_density_shrs_method_plot(data,title=None):
    """
    data: df of interest
    title: default is none, give string if desired
    
    Choose data and title to plot a density vs shrs plot, grouped by density "method" category

    """

    fig, ax = plt.subplots()
    #exponential fit
    a,b = np.polyfit(np.log(data['SHRS median'][:-2]),data['Density (kg/m3)'][:-2],1)
    x_plotfit = np.linspace(20,80,20)
    y_plotfit = a*np.log(x_plotfit) + b
    
    plt.plot(x_plotfit,y_plotfit,'-',color = 'grey',zorder = 1)
    
    colors = {1: 'blue', 2:'green'}
    grouped = data.groupby('Method')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter',label=key, y='Density (kg/m3)', x='SHRS median',  color=colors[key])

    plt.ylabel(r'Clast density,$\rho_s$ (kg/m$^3$)')
    plt.xlabel('Schmidt Hammer Rock Strength')
    plt.ylim(1500, 2800)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def run_abrasion_shrs_plot(data,title=None):
    """
    data: df of interest
    title: default is none, give string if desired
    Choose data and title to plot as an abrasion vs shrs plot

    """

    fig, ax = plt.subplots()

    #exponential fit
    a,b = np.polyfit(data['SHRS median'],np.log(data['Abrasion Avg']),1)
    x_plotfit = np.linspace(20,80,20)
    y_plotfit = np.exp(b)*np.exp(a*x_plotfit)
    print(title, "abrasion best fit b:", np.exp(b), "m:", a)
    
# =============================================================================
#     # this code allows you to find the necessary values to add the hard coded lines below
#     # Print the equation
#     equation = f'y = {np.exp(a):.4f} * exp({np.exp(b):.4f} * x)'
#     print(title, "abrasion best fit equation:", equation)
# 
# =============================================================================
    plt.plot(x_plotfit,y_plotfit,'-',color = 'grey',zorder = 1, label= 'All Data line')
    
    # Suiattle additional line
    x_additional = np.linspace(20, 80, 20)
    y_additional = 3.0714532806118964 * np.exp(-0.13637476779600016 * x_additional)

    # Plot the additional line
    plt.plot(x_additional, y_additional, '--', color='blue', label='Suiattle Line', zorder=2)

    # Just 2023 Data additional line
    x_additional2 = np.linspace(20, 80, 20)
    y_additional2 = 0.0271715170885713 * np.exp(-0.05050364086399952 * x_additional2)

    # Plot the 2023 additional line
    plt.plot(x_additional2, y_additional2, '--', color='red', label='2023 Data Line', zorder=3)

    
    colors = {'VV': lightpurple, 'VN': lightpurple, "LDV": lightpurple, "TV":lightpurple, "NA": orange, "HDV": darkpurple, "NV":darkpurple, "NN": darkpurple, "PN": darkpurple, "G":pink, "PL":pink, "UG":orange, "OW": orange}
    grouped = data.groupby('Lith')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', y='Abrasion Avg', x='SHRS median', color=colors[key]) #label=key,
    print(data)


    plt.yscale('log')
    plt.ylabel(r'Tumbler abrasion rate, $\alpha_t$ (1/km)')
    plt.xlabel('Schmidt Hammer Rock Strength')
    plt.title(title)
    plt.tight_layout()
    plt.legend()
    
    # Set x and y-axis limits
    plt.xlim(17, 83)  # Set your desired x-axis limits
    plt.ylim(0.00012194674843383384, 0.2791458876150631)  # Set your desired y-axis limits
    
    
    # Calculate R-squared value
    x_values = data['SHRS median']
    y_values = np.log(data['Abrasion Avg'])
    slope, intercept, r_value, p_value, std_err = linregress(x_values, y_values)
    r_squared = r_value**2
    print("R-squared value:", r_squared)
    
    # Annotate points with 'sample' values if the column exists
    if 'Sample' in data.columns:
        for _, row in data.iterrows():
            ax.text(row['SHRS median'], row['Abrasion Avg'], row['Sample'],
                    fontsize=10)
                    #fontsize=8, ha='center', va='center')
            
    plt.savefig("Abrasion"+title+".png", dpi=400, bbox_inches="tight") #pdf for illustrator
    plt.show()
    
#%%
# prep dfs for analysis

# change 2023 density abrasion data values of NAL to NA
## "Lith" column is "grouped liths", "Lithology" column is actual lithologies
density_abrasion_2023.loc[density_abrasion_2023['Lith'] == 'NAL', 'Lith'] = 'NA'
# select for density method 2 to new df dabrade2
dabrade2=density_abrasion_2023.loc[density_abrasion_2023['Method']==2].copy()
# select desired columns
dabrade2=dabrade2[['Sample','Lith', 'SHRS median','Density (kg/m3)','Abrasion Avg']].copy()
# remove na values...not sure why I have this / if it is necessary
dabrade2 = dabrade2[dabrade2['Abrasion Avg'].notna()]

# prep Suiattle data for concat to 2023 data
suiattle_data=suiattle_data[['Lith','SHRS median', 'Density mean', 'Abrasion Avg']].copy()
suiattle_data=suiattle_data.rename(columns={"Density mean": "Density (kg/m3)"})

# concat 2023 density/abrasion and suiattle density/abrasion data
density_abrasion= pd.concat([dabrade2,suiattle_data],axis=0)

# change 2023 abrasion values of NAL to NA
abr_data_2023.loc[abr_data_2023['Lith'] == 'NAL', 'Lith'] = 'NA'
abr_data_2023 = abr_data_2023[abr_data_2023['Abrasion Avg'].notna()]

#%%
#run plots

# compare density method 1 vs 2
run_density_shrs_method_plot(density_abrasion_2023, 'method 1 vs 2')

# plot density vs shrs for diff datasets
run_density_shrs_plot(dabrade2, '2023 Data')
run_density_shrs_plot(suiattle_data, 'Suiattle Data')
run_density_shrs_plot(density_abrasion, 'Density All Data')

# plot abrasion vs shrs for diff datasets
run_abrasion_shrs_plot(dabrade2, '2023 Data')
run_abrasion_shrs_plot(suiattle_data, 'Suiattle Data')
run_abrasion_shrs_plot(density_abrasion, 'Abrasion All Data')
run_abrasion_shrs_plot(finn_data, 'Finn Abrasion')


#%%
# Below is a series of plots to analyze measured and expected abrasion and shrs values to try and determine disparity in expected trend lines
#%%
# Create a new column - Expected Abrasion
# Calculate Expected Abrasion using Suiattle Equation with 2023 SHRS values as Residual Column

abr_data_2023['Expected Abrasion'] = 3.0714532806118964 * np.exp(-0.13637476779600016 * abr_data_2023['SHRS median'])
abr_data_2023['Residual'] = abr_data_2023['Abrasion Avg'] - abr_data_2023['Expected Abrasion']
abr_data_2023['% Residual'] = (abr_data_2023['Abrasion Avg'] - abr_data_2023['Expected Abrasion']) / abr_data_2023['Expected Abrasion'] * 100
# Plot Residual vs Grain size grouped by Sample
# Create scatter plot
fig, ax = plt.subplots()

# Select colormap
colormap = plt.cm.get_cmap('tab20')

# Generate colors from the colormap for each sample number
sample_colors = {sample_num: colormap(sample_num / 22) for sample_num in range(1, 223)}

for sample_num, group in abr_data_2023.groupby('Sample'):
    group.plot(ax=ax, kind='scatter', y= 'Residual', x='Size (cm)', color = sample_colors[sample_num], label=f"Sample {sample_num}")


# Add labels and title
plt.xlabel('Size (cm)')
plt.ylabel('Residual (Found - Expected)')
plt.title('Residual vs. Grain Size')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)


# Show the plot
plt.show()
#%%

# Plot Residual vs SHRS
# Create scatter plot
fig, ax = plt.subplots()

for sample_num, group in abr_data_2023.groupby('Sample'):
    group.plot(ax=ax, kind='scatter', y= 'Residual', x='SHRS median', color = sample_colors[sample_num], label=f"Sample {sample_num}")


# Add labels and title
plt.xlabel('SHRS')
plt.ylabel('Residual (Found - Expected)')
plt.title('Residual vs. SHRS')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)


# Show the plot
plt.show()

#%%
# Plot %Residual vs Grain size grouped by Sample
# Create scatter plot
fig, ax = plt.subplots()

for sample_num, group in abr_data_2023.groupby('Sample'):
    group.plot(ax=ax, kind='scatter', y= '% Residual', x='Size (cm)', color = sample_colors[sample_num], label=f"Sample {sample_num}")


# Add labels and title
plt.xlabel('Size (cm)')
plt.ylabel('% Residual (Found - Expected)')
plt.title('% Residual vs. Grain Size')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
plt.ylim(-200, 700)


# Show the plot
plt.show()

#%%

# Plot % Residual vs SHRS
# Create scatter plot
fig, ax = plt.subplots()

for sample_num, group in abr_data_2023.groupby('Sample'):
    group.plot(ax=ax, kind='scatter', y= 'Residual', x='SHRS median', color = sample_colors[sample_num], label=f"Sample {sample_num}")

# plt.semilogy(abr_data_2023['SHRS median'],abr_data_2023['Residual'],'.')

for i in range(len(abr_data_2023.groupby('Sample'))):
    ax.annotate(abr_data_2023['Sample'][i], (abr_data_2023['SHRS median'][i],abr_data_2023['Residual'][i]))
    
# Add labels and title
plt.xlabel('SHRS')
plt.ylabel('Residual (Found - Expected)')
ax.semilogy()
plt.title('Residual vs. SHRS')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
#plt.ylim(-200, 700)
plt.ylim(.0001, 10)

# Show the plot
plt.show()

