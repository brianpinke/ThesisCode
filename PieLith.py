#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 21:48:48 2023

@author: bpinke
"""

# loads necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
import os

# load in 2023 field data
path=os.path.join( os.getcwd(), 'Data/fieldbook_data_2023_USE.xlsx')

field_data = pd.read_excel (
        path, 
        sheet_name='1 - Measurements', 
        header = 0)

# load in Suiattle data
path=os.path.join( os.getcwd(), 'Data/SuiattleFieldData_Combined20182019.xlsx')

suiattle_data = pd.read_excel (
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

# set text size to 14
plt.rcParams.update({'font.size': 14})

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


def lith_piechart(deposit_name, color_list, lithpercenttext='no', title='no', ):
    """
    deposit_name: a string for the name of the deposit.
    color_list: choose the appropriate list name for the deposit.
    lithpercenttext: default is 'no' give string 'yes' for lith and percent printed on figure.
    title: default is 'no', give string 'yes' for title.
    
    Give deposit of interest and list of colors desired in order largest percent lith to smallest.
    Returns a hollow pie chart and prints lithpercent series in the console and on the plot.
    """
    lithpercent = lith_data.loc[(lith_data['Stop ID'] == deposit_name)]['Lithology Category'].value_counts() / lith_data.loc[(lith_data['Stop ID'] == deposit_name)]['Lithology Category'].count() * 100
    
    colors = color_list  # Used in order of df which is sorted big to small percent.
    fig1, ax1 = plt.subplots(figsize=(6, 6))  # Adjust the tuple (width, height) as needed

    # Create the pie chart with adjusted labels and autopct for percentages inside wedges
    
    # without percent and lith labels
    if lithpercenttext == 'no':
        wedges, texts, autotexts = ax1.pie(lithpercent, colors=colors, startangle=90, wedgeprops=dict(edgecolor='black'), autopct='', labeldistance = .82)
    #percent and lith labels
    else:
        wedges, texts, autotexts = ax1.pie(lithpercent, labels=lithpercent.index, colors=colors, startangle=90, wedgeprops=dict(edgecolor='black'), autopct='%1.1f%%', labeldistance= .82) 
   

    # Adjust label positions to prevent overlap
    #for text in texts:
        #text.set_horizontalalignment('center')

    # Create a white circle to make the pie chart hollow
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig1.gca().add_artist(centre_circle)
    
    ax1.axis('equal')
    plt.tight_layout()
    if title == 'yes':
        plt.title(deposit_name)
    plt.savefig("Pie" + deposit_name + ".png", dpi=400, bbox_inches="tight")
    plt.show()
    
    #print lithpercent in console
    print(lithpercent)

#%%
# dataset cleanup and prep for analysis

# Prep suiattle df for concat to field data
suiattle_data=suiattle_data.drop(columns=('Row Number'))
suiattle_data=suiattle_data.rename(columns={"Stop Number": "Stop ID"})
suiattle_data['Stop ID'] = suiattle_data['Stop ID'].astype("string")

# group suiattle G and P liths to PL and VV
suiattle_data.loc[suiattle_data['Lithology Category'] == 'G', 'Lithology Category'] = "PL"
suiattle_data.loc[suiattle_data['Lithology Category'] == 'P', 'Lithology Category'] = "VV"

# concat field data and suiattle data to lith data
lith_data= pd.concat([field_data,suiattle_data],axis=0)

# drop na lith category values
lith_data = lith_data[lith_data['Lithology Category'].notna()]

# drop extra rows that are from extra shrs measurements and not part of the random distribution
lith_data.drop(lith_data[lith_data['Extra SHRS'] == 'y'].index, inplace = True)


#%%
# merge liths and drop liths as neccessary

merge_liths = { "NP": 'NV', 'GF': 'PL', 'VNV':'NV', 'VC':'VB', "NAL":"NA", "VB":'VV'} #'G': 'PL' # Define replacements {old_lith: new_lith}

for old_lith, new_lith in merge_liths.items():
    lith_data.loc[lith_data['Lithology Category'] == old_lith, 'Lithology Category'] = new_lith

remove_liths = ['A', 'breccia', 'AN', 'C', 'S', 'sand', 'F', 'P']
lith_data = lith_data[~lith_data['Lithology Category'].isin(remove_liths)]

#%%
# combine exposures as needed for plotting
         
lith_data= add_combine_exposures(lith_data,exposures=['KC1','KC2','KC3','KC4','KC5','KC6','KC7'],
                                deposit_name="Kautz Creek")

lith_data= add_combine_exposures(lith_data,exposures=['T4','TNUV1','T5A','T5B','T6','T7','T8'],
                                deposit_name="Little Tahoma")

lith_data= add_combine_exposures(lith_data,exposures=['MM1','MM2','MM3','MM4','MM5','MM6','MM7','MM8','MM9', 'MM10', 'MM11'],
                                deposit_name="Mt. Meager")

lith_data= add_combine_exposures(lith_data,exposures=['MA1','MA1B','MA2','MA2B','MA3','MA3B','MA4','MA4A'],
                                deposit_name="Mt. Adams")

lith_data= add_combine_exposures(lith_data,exposures=['10.0','7.5','7.0','9.0','11.0'],
                                deposit_name="Suiattle River")


#%%
#define colors for each pie chart
# set colors for each deposit in high to low percent order
kautzcolors= [darkpurple, lightpurple, pink]
meagercolors = [lightpurple, pink, darkpurple, orange, green]
tahomacolors = [darkpurple, lightpurple, lightpurple, orange]
adamscolors = [orange, darkpurple, lightpurple, lighterorange]
suiattlecolors = [lightpurple, darkpurple, pink]

#%%
# create pie charts
lith_piechart('Mt. Meager', meagercolors)
lith_piechart('Suiattle River', suiattlecolors)
lith_piechart('Little Tahoma', tahomacolors)
lith_piechart('Kautz Creek', kautzcolors)
lith_piechart('Mt. Adams', adamscolors)



