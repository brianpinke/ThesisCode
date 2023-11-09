#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 13:04:05 2023

@author: bpinke
"""

# loads necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

#os.chdir('C:\\Users\\pinkeb\\OneDrive - Western Washington University\\Thesis\\ThesisCode')
os.chdir('/Users/bpinke/Library/CloudStorage/OneDrive-WesternWashingtonUniversity/Thesis/ThesisCode')

path=os.path.join( os.getcwd(), 'Densitydata.xlsx')
# loads in SHRS data text files: (1) metadata (2) lithology (3) medians
data_2023 = pd.read_excel (
        path, 
        sheet_name='data', 
        header = 0)

data1=data_2023.loc[data_2023['Method']==1].copy()
data1=data1[['Lith', 'SHRS median','Density (kg/m3)']].copy()

path=os.path.join( os.getcwd(), 'SHRS_Abrasion_Data.xlsx')
# loads in SHRS data text files: (1) metadata (2) lithology (3) medians
suiattle_data = pd.read_excel (
        path, 
        sheet_name='ToPython', 
        header = 0)

suiattle_data=suiattle_data[['Lith','SHRS median', 'Density mean']].copy()
suiattle_data=suiattle_data.rename(columns={"Density mean": "Density (kg/m3)"})

data2= pd.concat([data1,suiattle_data],axis=0)


#%%

def run_density_shrs_plot(data,title=None):
    """
    Choose data and title to plot as a density vs shrs plot

    """
   # LithCat = ['VV', 'NN','G']
    #colors = {'VV': 'blue', 'NN':'green', 'G':'red'}
# =============================================================================
#     plt.scatter(data['SHRS median'], data['Density (kg/m3)'], c=data['Lith'].map(colors))
#     
#     plt.xlabel(r'Clast density,$\rho_s$ (kg/m$^3$)',fontsize = 10)
#     plt.ylabel('Schmidt Hammer Rock Strength',fontsize = 10)
#     plt.legend()
# =============================================================================
    fig, ax = plt.subplots()
    #exponential fit
    a,b = np.polyfit(np.log(data['SHRS median'][:-2]),data['Density (kg/m3)'][:-2],1)
    x_plotfit = np.linspace(20,80,20)
    y_plotfit = a*np.log(x_plotfit) + b
    plt.plot(x_plotfit,y_plotfit,'-',color = 'grey',zorder = 1)
    
    colors = {'VV': 'blue', 'NN':'green', 'G':'red', 'TV':'orange'}
    grouped = data.groupby('Lith')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', y='Density (kg/m3)', x='SHRS median', label=key, color=colors[key])

    plt.ylabel(r'Clast density,$\rho_s$ (kg/m$^3$)')
    plt.xlabel('Schmidt Hammer Rock Strength')
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
#%%
run_density_shrs_plot(data_2023, '2023 Data')
run_density_shrs_plot(suiattle_data, 'Suiattle Data')
run_density_shrs_plot(data2, 'All Data')


