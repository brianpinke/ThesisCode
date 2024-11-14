#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 1 16:08:23 2024

@author: bpinke

# Plot Figure 4: Lithologic percents for the coarse material fraction of each deposit. Clasts <0.5 cm are denoted as fines in gray.
"""

# load necessary libraries
import matplotlib.pyplot as plt

import pandas as pd
import os

# load in 2023 field data
path=os.path.join( os.getcwd(), 'Data/fieldbook_data_2023_USE.xlsx')

pinke2023_data = pd.read_excel (
        path, 
        sheet_name='1 - Measurements', 
        header = 0)

# load in Suiattle data
path=os.path.join( os.getcwd(), 'Data/SuiattleFieldData_Combined20182019.xlsx')

suiattleGSD_data = pd.read_excel (
        path, 
        sheet_name='3 - Grain size df deposit', 
        header = 0)

# define colors
lightpurple= (0.7450980392156863, 0.7058823529411765, 0.9019607843137255)
darkpurple= (0.5019607843137255, 0.0, 0.5019607843137255)
orange= (1.0, 0.5294117647058824, 0.0)
pink=(1.0, 0.7529411764705882, 0.796078431372549)
green= (0.0, 0.5019607843137255, 0.0)
lighterorange =(1.0, 0.7254901960784313, 0.30196078431372547)

color_dict = {
    lightpurple: ["LDV", "VV", "TV", "VN", "V"],
    darkpurple: ["HDV", "NV", "NN", "PN"],
    orange: ["WW", "NA", "OW"],
    pink: ["G", "PL"],
    green: ["HGM"],
    lighterorange: ["VA"],
    'lightgray': ['F']
}

# Change text font and size
plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({'font.size': 10})

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
    """
    deposit = data.loc[data['Stop ID'].isin(exposures)]
    deposit['Stop ID'] = deposit_name
    data= pd.concat([data,deposit],axis=0)
    
    return data    

def lith_piechart(deposit_name, dataframe, color_dict, lithpercenttext='no', title='no', ax=None):
    """
    deposit_name: a string for the name of the deposit.
    dataframe: a dataframe with the deposit names and lithology data.
    color_dict: a dictionary where keys are colors and values are lists of lithologies.
    lithpercenttext: default is 'no', give string 'yes' for lith and percent printed on figure.
    title: default is 'no', give string 'yes' for title.
    
    Give deposit of interest and dictionary of colors desired in order largest percent lith to smallest.
    Returns a pie chart with coarse fraction lithology percents largest to smallest and fines in gray last.
    Prints in the console the deposit name, lithology type, and corresponding fraction.
    """
    deposit = dataframe.loc[dataframe['Stop ID'] == deposit_name].copy()  # Copy to avoid SettingWithCopyWarning
    lithpercent = deposit['Lithology Category'].value_counts() / deposit['Lithology Category'].count() * 100
    
    # Ensure 'F' is last
    if 'F' in lithpercent.index:
        lithpercent = lithpercent.reindex([idx for idx in lithpercent.index if idx != 'F'] + ['F'])
    
    # Create a color list based on the lithology categories in lithpercent
    color_list = []
    for lithology in lithpercent.index:
        for color, lithologies in color_dict.items():
            if lithology in lithologies:
                color_list.append(color)
                break
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))  # Adjust the size as needed
    
    if lithpercenttext == 'no':
        wedges, texts = ax.pie(
            lithpercent, colors=color_list, startangle=90, 
            wedgeprops=dict(edgecolor='black'), labeldistance=0.82
        )
    else:
        wedges, texts, autotexts = ax.pie(
            lithpercent, colors=color_list, startangle=90, #add labels=lithpercent.index, if want lith text too
            wedgeprops=dict(edgecolor='black'), autopct='%1.1f%%', labeldistance=1.2
        )
    
    # Customize the edge color for the 'F' category
    for i, wedge in enumerate(wedges):
        if lithpercent.index[i] == 'F':
            wedge.set_edgecolor('gray')
    
    # Create a white circle to make the pie chart hollow
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    ax.add_artist(centre_circle)
    
    ax.axis('equal')
    if title == 'yes':
        ax.set_title(deposit_name)
    if ax is None:
        plt.savefig("Pie" + deposit_name + ".png", dpi=400, bbox_inches="tight")
        plt.show()
    
    # Print lithpercent in console
    print(deposit_name, lithpercent)

def plot_multiple_pie_charts(deposit_names, dataframe, color_dict, fig_name, lithpercenttext='no', title='no'):
    """
    deposit_name: a string for the name of the deposit.
    dataframe: a dataframe with the deposit names and lithology data.
    color_dict: a dictionary where keys are colors and values are lists of lithologies.
    lithpercenttext: default is 'no', give string 'yes' for lithology percent printed on figure.
    title: default is 'no', give string 'yes' for title.
    
    Give deposit of interest and dictionary of colors desired in order largest percent lith to smallest.
    Returns a pie chart with coarse fraction lithology percents largest to smallest and fines in gray last.
    """
    n = len(deposit_names)
    cols = 3  # Number of columns
    rows = (n // cols) + (n % cols > 0)  # Calculate rows needed
    
    # Calculate the height based on the number of rows and the aspect ratio
    fig_height = 9 * (rows / cols)  # Extend height proportionally to the number of rows
    fig_width = 6.5  # Keep width fixed
    
    fig, axs = plt.subplots(rows, cols, figsize=(fig_width, fig_height))  # Adjust the size as needed
    axs = axs.flatten()  # Flatten the 2D array of axes to a 1D array for easy iteration

    # Generate the pie charts
    for i, deposit_name in enumerate(deposit_names):
        lith_piechart(deposit_name, dataframe, color_dict, lithpercenttext, title, ax=axs[i])

    # Hide any unused subplots
    for j in range(len(deposit_names), len(axs)):
        fig.delaxes(axs[j])  
    
    # Adjust layout
    plt.tight_layout()

    # Save the entire figure
    plt.savefig("Pie" + fig_name + ".png", dpi=400, bbox_inches="tight")

    # Show the plot
    plt.show()

#%%
# dataset cleanup and prep for analysis

# Prep suiattle df for concat to field data
suiattleGSD_data=suiattleGSD_data.drop(columns=('Row Number'))
suiattleGSD_data=suiattleGSD_data.rename(columns={"Stop Number": "Stop ID"})
suiattleGSD_data['Stop ID'] = suiattleGSD_data['Stop ID'].astype("string")

# group suiattle G and P liths to PL and VV
suiattleGSD_data.loc[suiattleGSD_data['Lithology Category'] == 'G', 'Lithology Category'] = "PL"
suiattleGSD_data.loc[suiattleGSD_data['Lithology Category'] == 'P', 'Lithology Category'] = "VV"

# concat field data and suiattle data to lith data
lithpie_data= pd.concat([pinke2023_data,suiattleGSD_data],axis=0)

# drop extra rows that are from extra shrs measurements and not part of the random distribution
lithpie_data.drop(lithpie_data[lithpie_data['Extra SHRS'] == 'y'].index, inplace = True)

# merge liths and drop liths as neccessary

merge_liths = { "NP": 'NV', 'GF': 'PL', 'VNV':'NV', 'VC':'VB', "NAL":"NA", "VB":'VV', 'UG':'WW', 'G':'PL'} #'G': 'PL' # Define replacements {old_lith: new_lith}

for old_lith, new_lith in merge_liths.items():
    lithpie_data.loc[lithpie_data['Lithology Category'] == old_lith, 'Lithology Category'] = new_lith
    
# if want to plot fines
lithpie_data.loc[(lithpie_data['Stop ID'].isin(['7.0','7.5','9.0','10.0','11.0'])) & (lithpie_data['Lithology Category'].isin(['F', 'S', 'sand'])), 'Lithology Category'] = 'F'  
lithpie_data.loc[(lithpie_data['Size (cm)'].isin(['F'])), 'Lithology Category'] = 'F'
lithpie_data = lithpie_data[lithpie_data['Lithology Category'].notna()]
remove_liths = ['A', 'breccia', 'AN', 'C', 'P'] # removes unwanted/rare liths, 'P'
lithpie_data = lithpie_data[~lithpie_data['Lithology Category'].isin(remove_liths)]
lithpie_data = lithpie_data[['Stop ID', 'Lithology Category']]
#%%
# combine exposures as needed for plotting
lithpie_data= add_combine_exposures(lithpie_data,exposures=['T5A','T5B'],
                                deposit_name="T5")

lithpie_data= add_combine_exposures(lithpie_data,exposures=['KC1','KC2','KC3','KC4','KC5','KC6','KC7'],
                                deposit_name="Kautz Creek")

lithpie_data= add_combine_exposures(lithpie_data,exposures=['T4','T5','T6','T7','T8'],
                                deposit_name="Little Tahoma")

lithpie_data= add_combine_exposures(lithpie_data,exposures=['MM1','MM2','MM3','MM4','MM5','MM6','MM7','MM8','MM9', 'MM10', 'MM11'],
                                deposit_name="Mt. Meager")

lithpie_data= add_combine_exposures(lithpie_data,exposures=['MA1','MA1B','MA2','MA2B','MA3','MA3B'],
                                deposit_name="Mt. Adams")

lithpie_data= add_combine_exposures(lithpie_data,exposures=['7.5','7.0','9.0','10.0','11.0'],
                                deposit_name="Suiattle River")

#%%
# Plot Figure 4: Lithologic percents for the coarse material fraction of each deposit. Clasts <0.5 cm are denoted as fines in gray.
alldeposits = ['Mt. Meager', 'Suiattle River', 'Little Tahoma', 'Kautz Creek','Mt. Adams',]
plot_multiple_pie_charts(alldeposits,lithpie_data, color_dict, 'All Deposits')

