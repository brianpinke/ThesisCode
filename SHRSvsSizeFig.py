#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 14:39:55 2024

@author: bpinke

# Plot Figure 13: Sample clast size vs. Strength colored by lithology.
"""

# loads necessary libraries
import matplotlib.pyplot as plt
import pandas as pd
import os

# load in 2023 field data
path=os.path.join( os.getcwd(), 'Data/fieldbook_data_2023_USE.xlsx')

field_data2023 = pd.read_excel (
        path, 
        sheet_name='1 - Measurements', 
        header = 0)

# define colors
lightpurple= (0.7450980392156863, 0.7058823529411765, 0.9019607843137255)
darkpurple= (0.5019607843137255, 0.0, 0.5019607843137255)
orange= (1.0, 0.5294117647058824, 0.0)
pink=(1.0, 0.7529411764705882, 0.796078431372549)
green= (0.0, 0.5019607843137255, 0.0)
lighterorange = (1.0, 0.7254901960784313, 0.30196078431372547)


# change font size
plt.rcParams["font.family"] = "Arial"
plt.rcParams['font.size'] = 10

# change the directory so figs are saved in proper location
os.chdir(os.getcwd()+'/Figs')

#%%
# define functions

def add_combine_exposures(data,exposures,deposit_name):
    """
    data: df of interest
    exposures: strings of exposures to be combined
    deposit_name: string of name of combined exposures
    Add combined exposures as a deposit to df
    
    combines exposures in a given dataset given a new "deposit_name", appended to existing dataframe.
    
    """
    deposit = data.loc[data['Stop ID'].isin(exposures)]
    deposit['Stop ID'] = deposit_name
    data= pd.concat([data,deposit],axis=0)
    
    return data 

# Define a function to assign a color to each lithology category
def assign_color(lithology_category):
    """
    This function assigns a color to a given lithology category based on the predefined `color_dict` mapping.
    It returns the color (as an RGB tuple) if the category is found in the dictionary, otherwise it returns black.
    
    Parameters:
    lithology_category (str): The lithology category to assign a color to.
    
    Returns:
    tuple: The RGB color tuple for the category, or black (0, 0, 0) if not found.
    """
    
    # Iterate through color_dict to find a matching category
    for color, lithologies in color_dict.items():
        if lithology_category in lithologies:
            return color
    
    # Return black if no match is found
    return (0, 0, 0)

#%%
# Convert 'Size (cm)' to numeric, coercing errors to NaN
field_data2023['Size (cm)'] = pd.to_numeric(field_data2023['Size (cm)'], errors='coerce')

# Filter the DataFrame
field_data2023 = field_data2023.loc[(field_data2023['Extra SHRS'] == 'y') | (field_data2023['Size (cm)'] >= 25)]

#prep liths
merge_liths = { "NP": 'NV', 'GF': 'PL', 'VNV':'NV', 'VC':'VB', "NAL":"NA", "VB":'VV', 'UG':'WW', 'G':'PL'} # Define replacements {old_lith: new_lith}

for old_lith, new_lith in merge_liths.items():
    field_data2023.loc[field_data2023['Lithology Category'] == old_lith, 'Lithology Category'] = new_lith
    
    
# change values of NAL to NA - had to be entered into excel as NAL as NA would appear as empty to python
field_data2023.loc[field_data2023['Lithology Category'] == 'NAL', 'Lithology Category'] = 'NA'

SHRS_data = field_data2023.copy()

SHRS_data= add_combine_exposures(SHRS_data,exposures=['KC1','KC2','KC3','KC4','KC5','KC6','KC7'],
                                deposit_name= 'MRKC')

SHRS_data= add_combine_exposures(SHRS_data,exposures=['T5A','T5B'],
                                deposit_name="T5")

SHRS_data= add_combine_exposures(SHRS_data,exposures=['T4','T5','T6','T7','T8'],
                                deposit_name= 'MRWR')

SHRS_data= add_combine_exposures(SHRS_data,exposures=['MM1','MM2','MM3','MM4','MM5','MM6','MM7','MM8','MM9', 'MM10', 'MM11'],
                                deposit_name= 'MMLR')

SHRS_data= add_combine_exposures(SHRS_data,exposures=['MA1','MA1B','MA2','MA2B','MA3','MA3B'],#removed 4/4A
                                deposit_name= 'MASC')
    
    
filtered_df = SHRS_data[SHRS_data['Stop ID'].isin(['MRKC', 'MRWR', 'MMLR', 'MASC'])]

color_dict = {
    lightpurple: ["LDV", "VV", "TV", "VN", "V", 'breccia'],
    darkpurple: ["HDV", "NV", "NN", "PN", 'A'],
    orange: ["WW", "NA", "OW"],
    pink: ["G", "PL"],
    green: ["HGM"],
    lighterorange: ["VA"]
}

colorlist = [lighterorange, orange, lightpurple, darkpurple, green, pink]

categories = [
    "Vesicular Altered",    
    "Non-Vesicular Altered",
    "Vesicular",            
    "Non-Vesicular",        
    "Metamorphic",          
    "Plutonic"
]
#%%
# Plot Figure 13: Sample clast size vs. Strength colored by lithology.

# Apply the color assignment
filtered_df['color'] = filtered_df['Lithology Category'].apply(assign_color)

# Create the scatter plot
plt.figure(figsize=(4.3, 3))
plt.scatter(filtered_df['Mean_Median SHRS'], filtered_df['Size (cm)'], c=filtered_df['color'])

# Labels and title

plt.xlabel('SHRS')
plt.ylabel('Size (cm)')
#plt.title('Scatter Plot of Size vs SHRS Colored by Lithology Category')

# Create custom legend using the provided colors and categories
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=category) 
           for color, category in zip(colorlist, categories)]

# Add the legend to the plot
#plt.legend(handles=handles, loc='upper right')

plt.savefig('SizeVsSHRSScatter.png', dpi=400, bbox_inches="tight")
# Show the plot
plt.show()

