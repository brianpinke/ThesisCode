#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 15:09:10 2024

@author: bpinke

# Plot Figure 10(A): SHRS vs. abrasion rate. Modified from Pfeiffer et al. (2022), trendline is for Pfeiffer et al. (2022) and this study’s combined datasets.
# Plot Figure 10(B): SHRS vs. Density. Modified from Pfeiffer et al. (2022), trendline is for Pfeiffer et al. (2022) and this study’s combined datasets.
"""

# loads necessary libraries
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
import os

# load in 2023 density and abrasion data
path=os.path.join( os.getcwd(), 'Data/PinkeTumbling2023_USE.xlsx')

# load just 2023 abrasion data
abr_data_2023 = pd.read_excel (
        path, 
        sheet_name='Abrasion', 
        header = 0)

# load in Suiattle SHRS, abrasion, and density
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

# change text font and size
plt.rcParams["font.family"] = "Arial"
plt.rcParams['font.size'] = 10

# change the directory so figs are saved in proper location
os.chdir(os.getcwd()+'/Figs')

#%%
def exp_fit(data, title, dot_color, fillcolor=0.2, subdata='abrasion'):
    """
    data (DataFrame): The input data containing 'SHRS median' and 'Abrasion Avg' or 'Density (kg/m3)'.
    title (str): The title for the trendline in the plot legend.
    dot_color (str): The color of the trendline to match the corresponding data points.
    fillcolor (float): The alpha value for the confidence interval fill color.
    subdata (str): Specify 'abrasion' for exponential fit and 'density' for logarithmic fit.
    
    This function calculates and prints the R squared value and p-value for the data, 
    and plots the appropriate trendline (exponential or logarithmic) on the scatter plot.
    """
    
    if subdata == 'abrasion':
        x = data['SHRS median']
        y = data['Abrasion Avg']
        log_y = np.log(y)  # Log of abrasion avg
        a, b = np.polyfit(x, log_y, 1)  # Fit log(y) to a linear model with x
        print(title, "abrasion best fit b:", b, "m:", a)



        # Calculate R squared value
        y_pred = np.exp(b) * np.exp(a * x)  # Calc predicted y values (abrasion)
        ss_tot = sum((y - np.mean(y))**2)  # Total sum of squares
        ss_res = sum((y - y_pred)**2)  # Residual sum of squares
        r_squared = 1 - (ss_res / ss_tot)  # R squared formula

        # Calculate p-value
        n = len(x)  # Get number of data points
        mean_x = np.mean(x)  # Calculate mean of x (shrs median)
        ss_xx = sum((x - mean_x)**2)  # Calc sum of squares of shrs median
        se = np.sqrt(sum((log_y - (a * x + b))**2) / (n - 2))  # Standard error of the residuals
        se_a = se / np.sqrt(ss_xx)  # Standard error of slope
        t_stat = a / se_a  # t-statistic for slope
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 2))  # Two-tailed p-value

        print(f'{title} R squared value: {r_squared}, p-value: {p_value}')

        x_plotfit = np.linspace(20, 80, 100)  # Generate x values for the trendline
        y_plotfit = np.exp(b) * np.exp(a * x_plotfit)  # Calculate y values for the trendline
        plt.plot(x_plotfit, y_plotfit, label=title, linestyle='-', color=dot_color, alpha=0.6, zorder=1) #'--'
        
        # Calculate confidence interval
        y_fit = np.exp(b) * np.exp(a * x)
        residuals = log_y - (a * x + b)
        s_res = np.sqrt(np.sum(residuals**2) / (n - 2))
        x_range = np.linspace(min(x), max(x), 100)
        y_fit = np.exp(b) * np.exp(a * x_range)

        conf_interval = 1.96 * s_res * np.sqrt(1/n + (x_range - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
        lower_bound = y_fit * np.exp(-conf_interval)
        upper_bound = y_fit * np.exp(conf_interval)
        plt.fill_between(x_range, lower_bound, upper_bound, color=dot_color, alpha=fillcolor, linewidth=0)

    elif subdata == 'density':
        x = data['SHRS median']
        y = data['Density (kg/m3)']
        log_x = np.log(x)  # Log of SHRS median
        a, b = np.polyfit(log_x, y, 1)  # Fit y to a linear model with log(x)
        
        print(title, "density best fit b:", b, "m:", a)

        # Calculate R squared value
        y_pred = a * log_x + b  # Calc predicted y values (density)
        ss_tot = sum((y - np.mean(y))**2)  # Total sum of squares
        ss_res = sum((y - y_pred)**2)  # Residual sum of squares
        r_squared = 1 - (ss_res / ss_tot)  # R squared formula

        # Calculate p-value
        n = len(x)  # Get number of data points
        mean_log_x = np.mean(log_x)  # Calculate mean of log(x)
        ss_xx = sum((log_x - mean_log_x)**2)  # Calc sum of squares of log(x)
        se = np.sqrt(sum((y - (a * log_x + b))**2) / (n - 2))  # Standard error of the residuals
        se_a = se / np.sqrt(ss_xx)  # Standard error of slope
        t_stat = a / se_a  # t-statistic for slope
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 2))  # Two-tailed p-value

        print(f'{title} R squared value: {r_squared}, p-value: {p_value}')

        x_plotfit = np.linspace(20, 80, 100)  # Generate x values for the trendline
        y_plotfit = a * np.log(x_plotfit) + b  # Calculate y values for the trendline
        plt.plot(x_plotfit, y_plotfit, label=title, linestyle='-', color=dot_color, alpha=0.6, zorder=1) #'--'
        
        # Calculate confidence interval
        y_fit = a * log_x + b
        residuals = y - y_fit
        s_res = np.sqrt(np.sum(residuals**2) / (n - 2))
        x_range = np.linspace(min(x), max(x), 100)
        log_x_range = np.log(x_range)
        y_fit = a * log_x_range + b

        conf_interval = 1.96 * s_res * np.sqrt(1/n + (log_x_range - mean_log_x)**2 / np.sum((log_x - mean_log_x)**2))
        lower_bound = y_fit - conf_interval
        upper_bound = y_fit + conf_interval
        plt.fill_between(x_range, lower_bound, upper_bound, color=dot_color, alpha=fillcolor, linewidth=0)
        
#%%
# prep dfs for analysis

# change 2023 density abrasion data values of NAL to NA
# "Lith" column is "grouped liths", "Lithology" column is actual lithologies

# merge liths and drop liths as neccessary
merge_liths = { "NP": 'NV', 'GF': 'PL', 'VNV':'NV', 'VC':'VB', "NAL":"NA", "VB":'VV', 'UG':'NV','TV':'VV', 'PN':'NV','HDV':'NV', 'LDV':'VV','G':'PL', 'VN':'VV','NN':'NV', 'G':'PL'} # Define replacements {old_lith: new_lith}

for old_lith, new_lith in merge_liths.items():
    suiattle_data.loc[suiattle_data['Lith'] == old_lith, 'Lith'] = new_lith

# prep Suiattle SHRS data
suiattle_data=suiattle_data[['Lith','SHRS median', 'Density mean', 'Max Density', 'Min Density', 'Abrasion Avg', 'Max Abrasion', 'Min Abrasion', 'SHRS stdev']].copy()
suiattle_data=suiattle_data.rename(columns={"Density mean": "Density (kg/m3)"})
suiattle_data['deposit'] = "suiattle"


# drop NA values from finn_data
finn_data = finn_data[finn_data['Abrasion Avg'].notna()]
# change 2023 abrasion values of NAL to NA
abr_data_2023.loc[abr_data_2023['Lith'] == 'NAL', 'Lith'] = 'NA'
abr_data_2023 = pd.concat([abr_data_2023, finn_data], axis =0)
abr_data_2023 = abr_data_2023[abr_data_2023['Abrasion Avg'].notna()]


# merge liths and drop liths as neccessary

for old_lith, new_lith in merge_liths.items():
    abr_data_2023.loc[abr_data_2023['Lith'] == old_lith, 'Lith'] = new_lith

#%%
# Plot Figure 10(A): SHRS vs. abrasion rate. Modified from Pfeiffer et al. (2022), trendline is for Pfeiffer et al. (2022) and this study’s combined datasets.

size = abr_data_2023['Size (cm)']
#merge suiattle to abr_data_2023 as 'all_abr'
all_abr = pd.concat([abr_data_2023, suiattle_data], axis =0)
all_abr = all_abr.reset_index(drop=True)

# Extracting the data from suiattle_data
x_suiattle = suiattle_data['SHRS median']
y_suiattle = suiattle_data['Abrasion Avg']
xerr_suiattle =suiattle_data['SHRS stdev']#/ np.sqrt(3)

# Get y-error for suiattle_data
yerr_lower_suiattle = suiattle_data['Min Abrasion']
yerr_upper_suiattle = suiattle_data['Max Abrasion']

# Identify subset where Size (cm) >= 25
abr_25 = abr_data_2023[size >= 25]

# Extracting data for abr_25
x_abr_25 = abr_25['SHRS median']
y_abr_25 = abr_25['Abrasion Avg']
xerr_abr_25 = abr_25['SHRS stdev'] 
yerr_abr_25 = abr_25['Abr Error']

# Combine abr_25 and suiattle data for trendline
abr25_suiattle = pd.concat([abr_25, suiattle_data], ignore_index=True)

# Create a scatter plot
plt.figure(figsize=(4.3, 3)) #inches

# Plot abr data with error bars and trendlines matching dot colors
plt.errorbar(x_abr_25, y_abr_25, xerr=xerr_abr_25, yerr= yerr_abr_25, fmt='o', alpha=1, color='blue', ecolor='gray',markersize=0,elinewidth=.6, zorder=2) #label=All Data
plt.errorbar(x_suiattle, y_suiattle, xerr=xerr_suiattle, yerr=[yerr_lower_suiattle, yerr_upper_suiattle], fmt='o',ecolor='gray', alpha=1, markersize=0, elinewidth=.6,zorder=2) #label=Glacier Peak (Suiattle)

plt.plot(x_abr_25, y_abr_25, 'o', markersize=6, color='black', label='This Study',zorder=4)
plt.plot(x_suiattle, y_suiattle, 's', markerfacecolor='white', markeredgecolor='black', markersize=6, markeredgewidth=.7, color='black', label='Pfeiffer 2022',zorder=3)
    
# Plot exponential trendlines with dot colors
#exp_fit(all_abr, 'allabr' , 'red', 0.05)
exp_fit(abr25_suiattle, None , 'black', 0.05)
#exp_fit(suiattle_data, None, 'brown', 0.05)

# Set y-axis to log scale
plt.yscale('log')

# Set x and y-axis limits
plt.xlim(17, 83)  # Set your desired x-axis limits
plt.ylim(0.00012194674843383384, 0.2791458876150631)

# Labels and legend
plt.xlabel('Schmidt Hammer Rock Strength')
plt.ylabel(r'Tumbler abrasion rate, $\alpha_t$ (1/km)')
plt.legend()
#plt.title('Abrasion vs SHRS, Suiattle, >=25cm')
plt.savefig("Abrasion_SHRSFINAL"+".png", dpi=400, bbox_inches='tight')
plt.show()

#%%
# Plot Figure 10(B): SHRS vs. Density. Modified from Pfeiffer et al. (2022), trendline is for Pfeiffer et al. (2022) and this study’s combined datasets.

#merge suiattle to abr_data_2023 as 'all_dens'
all_dens = pd.concat([abr_data_2023, suiattle_data], axis =0)
all_dens = all_dens.reset_index(drop=True)

dens25_suiattle = pd.concat([abr_25, suiattle_data], ignore_index=True)

# Extracting the density data from suiattle_data
# drop Volcanic Breccia. Pfeiffer 2022 did not use for density
suiattle_data = suiattle_data.drop(index=[12])
# define for density
x_suiattle = suiattle_data['SHRS median']
y_suiattle = suiattle_data['Density (kg/m3)']
xerr_suiattle =suiattle_data['SHRS stdev']

yerr_lower_suiattle = suiattle_data['Min Density']
yerr_upper_suiattle = suiattle_data['Max Density']

#no y err for density unless combine the sub samples but thats not how allison plotted them in 2022 paper

# Extracting y data for density
y_abr_25 = abr_25['Density (kg/m3)']

# Create a scatter plot
plt.figure(figsize=(4.3, 3)) #inches

# Plot abr data with error bars and trendlines matching dot colors
plt.errorbar(x_abr_25, y_abr_25, xerr=xerr_abr_25, fmt='o', alpha=1, color='blue', ecolor='gray',elinewidth=.6, markersize=0,zorder=2) #label=All Data
plt.errorbar(x_suiattle, y_suiattle, xerr=xerr_suiattle, yerr=[yerr_lower_suiattle, yerr_upper_suiattle], fmt='o',ecolor='gray', alpha=1, markersize=0,elinewidth=.6, zorder=2) #label=Glacier Peak (Suiattle)

plt.plot(x_abr_25, y_abr_25, 'o', markersize=6, color='black', label='This Study',zorder=4)
plt.plot(x_suiattle, y_suiattle, 's', markerfacecolor='white', markeredgecolor='black', markersize=6, markeredgewidth=.7, color='black', label='Pfeiffer 2022',zorder=3)

# Plot exponential trendlines with dot colors
#exp_fit(all_dens, 'alldens', 'red', 0.05, 'density')
exp_fit(dens25_suiattle, '25dens', 'black', 0.05, 'density')
#exp_fit(suiattle_data, None, 'brown', 0.05, 'density')

# Set x and y-axis limits
plt.xlim(17, 83)  # Set your desired x-axis limits
plt.ylim(1500, 2800)

# Labels and legend
plt.xlabel('Schmidt Hammer Rock Strength')
plt.ylabel(r'Density (kg/m$^3$)')
#plt.title('Abrasion vs SHRS, Suiattle, >=25cm')
#plt.legend()
plt.savefig("Density_SHRSFINAL"+".png", dpi=400, bbox_inches='tight')
plt.show()
