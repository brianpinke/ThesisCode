# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 20:44:04 2023

@author: pinkeb

# Plot Figure 8: Coarse fraction grain size distribution, excluding clasts <0.5 cm. Individual exposures in gray. 
# Print out D 16, 50, 95 and %fines of each deposit in console.
"""

# loads necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FormatStrFormatter

# load in 2023 field data
path=os.path.join( os.getcwd(), 'Data/fieldbook_data_2023_USE.xlsx')

field_data2023 = pd.read_excel (
        path, 
        sheet_name='1 - Measurements', 
        header = 0)

# load in Suiattle Debris Flow deposit GSD data
path=os.path.join( os.getcwd(), 'Data/SuiattleFieldData_Combined20182019.xlsx')

suiattleGSD_data = pd.read_excel (
        path, 
        sheet_name='3 - Grain size df deposit', 
        header = 0)

# define colors

LightGray = (0.8274509803921568, 0.8274509803921568, 0.8274509803921568)
MediumGray = (0.6627450980392157, 0.6627450980392157, 0.6627450980392157)
DarkGray = (0.4117647058823529, 0.4117647058823529, 0.4117647058823529)

meagercolor = (0.4, 0.2, 0.0) #dark brown
glaciercolor = (0.0, 0.5019607843137255, 0.5019607843137255) #teal
tahomacolor = (0.4, 0.6, 0.8) #steel blue
kautzcolor = (0.8627450980392157, 0.0784313725490196, 0.23529411764705882) #crimson
adamscolor = (0.5019607843137255, 0.5019607843137255, 0.0) #olive

# change text font and size
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

def process_size(x, fines = 'no'):
    """
    x is the size cm of all data collected FOR ONE SITE
    1. set 0 to nans and drop nans, drops fines
    2. sort values
    3. cm/100= m 
    
    If fines = 'yes', it keeps fines as 0.5cm

    """
    #set non numerics to nan 
    size_numeric = pd.to_numeric(x,errors="coerce") 
    
    if fines == "no":
        #set 0 to nan 
        size_numeric[size_numeric==0] = np.nan 
    else:
        # Replace NaN values with 0.5 cm (fines)
        size_numeric = size_numeric.fillna(0.5)
        
        #set 0 to 0.5 (fines) 
        size_numeric[size_numeric==0] = 0.5
    
    #drop nans, sort, cm-->m 
    return size_numeric.dropna().sort_values()/100 

def get_DX(exposure, percentile=50, df=None, fines = 'no'):
    """
    exposure: exposures in Stop ID of interest.
    percentile: desired percentile of interest. Default is 50%, give as a number 0-100.
    df: df of interest, default is gsd_data. Give different df if desired. must have "Stop ID"  and Size (cm) columns with gsd in cm.
    
    Prints in the console and returns the Dx of an exposure for a given X percentile in meters.

    """
    if df == None:
        df = gsd_data
        
    dx = df.loc[df['Stop ID']==exposure]["Size (cm)"]
    
    if fines == 'no':
        dx = process_size(dx) #use if want to remove fines
    else:
        dx = process_size(dx, 'yes') #use if want to keep fines
    
    dx = np.percentile(dx, percentile)
    
    print("The D",percentile,"of", exposure, "is", round(np.median(dx),3), "m")
    return dx

def percent_fines(data, exposure):
    """
    data: df of interest with "Stop ID" and "Size (cm) columns"
    exposure: string of the deposit or exposure of interest
    
    Prints and returns the percent fines in a given exposure or deposit.

    """
    # To get % fines, I need to get # of rows in deposit, 
    #and # of rows in deposit that are fines, then do percentage.
    
   # Give it df and deposit or exposure of interest
    # new temp df of the desired exposure grain size values
    dx = data.loc[data['Stop ID']== exposure]["Size (cm)"]
    
    # Check if there are any 'F' values in the filtered DataFrame
    if 'F' not in dx.values:
        print(f"There are no fines in the exposure {exposure}.")
        return None
    
    # counts number of fines found in the exposure and total gsd measurements
    fines_count= dx.value_counts()['F']
    total_count = len(dx.index)
    percent_fines = fines_count / total_count * 100
    percent_fines = round(percent_fines, 2)
    
    print("The percent fines of", exposure, "is:", percent_fines, "%")
    return percent_fines

def run_grain_size_plot(data, exposures, colors=None, line_types=None, title=None, ax=None):
    """
    data: df of interest
    exposures: list of strings of exposures (stop ID column) to be used in plot
    colors: list of strings of colors. Must be equal number of colors as exposures. nth color is used for nth exposure.
    line_types: list of strings of line types. Must be equal number of line types as exposures. nth line type is used for nth exposure.
    title: default is none, give a string if desired.
    ax: matplotlib axis object to plot on. If None, a new figure is created.
    """
    GsdBySite = data.groupby('Stop ID').agg({'Size (cm)': process_size}).loc[exposures]
    
    if ax is None:
        fig, ax = plt.subplots()
    
    for i, (site, size) in enumerate(GsdBySite.iterrows()):
        count = np.linspace(0, 100, len(size[0]))
        ax.semilogx(size[0], count, label=str(site), color=colors[i], linestyle=line_types[i], zorder=2)
        
    print(exposures)
    a =[16,50,95]

    for i in exposures:
        for b in a:
               get_DX(i, b)
               percent_fines(gsd_data, i)
                
    ax.set_xlabel('Grain Size (m)')
    ax.set_ylabel('Cumulative % Finer')
    ax.set_ylim([0, 100])
    ax.set_xlim(0.0035229004419047457, 7.806067884544429)
    
    # add grain size delineation lines
    ax.axvline(x = .256, color ='black', linestyle = '-', linewidth=.9, label ='axvline - full height', zorder=1) #boulder
    ax.axvline(x = .064, color ='black', linestyle = '-', linewidth=.9, label ='axvline - full height', zorder=1) #cobble
    ax.axvline(x = .005, color ='black', linestyle = '-', linewidth=.9, label ='axvline - full height', zorder=1) #gravel
    ax.axhline(y = 50, color ='black', linestyle = '--', linewidth=.5, label ='axvline - full height', zorder=1) #D50
    #ax.axvline(x = .002, color ='b', label ='axvline - full height') #gravel
    
    #ax.legend()
    #ax.set_title(title)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    
    if ax is None:
        plt.savefig("GSD" + title + ".png", dpi=400, bbox_inches="tight")
        plt.show()
        
        
def run_grain_size_subplots(data, plot_details):
    """
    data: df of interest
    plot_details: list of dictionaries containing 'exposures', 'colors', 'line_types', and 'title' for each subplot.
    Figure of 6 plots using the run_grain_size_plot function
    """
    fig, axs = plt.subplots(3, 2, figsize=(6.5, 9))  # 3 rows, 2 columns
    axs = axs.flatten()  # Flatten the 2D array of axes for easy iteration
    
    for ax, detail in zip(axs, plot_details):
        run_grain_size_plot(data, 
                            exposures=detail['exposures'], 
                            colors=detail['colors'], 
                            line_types=detail['line_types'], 
                            title=detail['title'], 
                            ax=ax)
    
    plt.tight_layout()
    plt.savefig("GSD_Subplots.png", dpi=400, bbox_inches="tight")
    plt.show()

#%%  
# prep suiattle gsd df for merge with 2023 data
suiattleGSD_data=suiattleGSD_data.drop(columns=('Row Number'))
suiattleGSD_data=suiattleGSD_data.rename(columns={"Stop Number": "Stop ID"})
suiattleGSD_data['Stop ID'] = suiattleGSD_data['Stop ID'].astype("string")
#%%
# merge 2023 and suiattle data
gsd_data= pd.concat([field_data2023,suiattleGSD_data],axis=0)
    
#Set all extra shrs measurements to nan
gsd_data.drop(gsd_data[gsd_data['Extra SHRS'] == 'y'].index, inplace = True)

# combine exposures as needed for plotting
gsd_data= add_combine_exposures(gsd_data,exposures=['KC1','KC2','KC3','KC4','KC5','KC6','KC7'],
                                deposit_name="Kautz Creek")

gsd_data= add_combine_exposures(gsd_data,exposures=['T5A','T5B'],
                                deposit_name="T5")

gsd_data= add_combine_exposures(gsd_data,exposures=['T4','T5','T6','T7','T8'],
                                deposit_name="Little Tahoma")

gsd_data= add_combine_exposures(gsd_data,exposures=['MM1','MM2','MM3','MM4','MM5','MM6','MM7','MM8','MM9', 'MM10', 'MM11'],
                                deposit_name="Mt. Meager")

gsd_data= add_combine_exposures(gsd_data,exposures=['MA1','MA1B','MA2','MA2B','MA3','MA3B'],#removed 4A/4B
                                deposit_name="Mt. Adams")

gsd_data= add_combine_exposures(gsd_data,exposures=['7.0', '7.5','9.0','10.0','11.0'],
                                deposit_name="Suiattle River")

#%%
plot_details = [
    
    {'exposures': ['Mt. Meager', 'Suiattle River', 'Little Tahoma', 'Kautz Creek', 'Mt. Adams'],
     'colors': [meagercolor, glaciercolor, tahomacolor, kautzcolor, adamscolor],
     'line_types': ['solid'] * 5,
     'title': 'All Deposits GSD'},
 
    {'exposures': ['MM1', 'MM2', 'MM3', 'MM4', 'MM5', 'MM6', 'MM7', 'MM8', 'MM9', 'MM10', 'MM11', 'Mt. Meager'],
     'colors': [DarkGray] +[DarkGray] +[LightGray] +[LightGray] +[DarkGray] +[DarkGray] +[DarkGray] +[LightGray] +[DarkGray] +[LightGray] +[LightGray] + [meagercolor],
     'line_types': ['solid'] * 12,
     'title': 'Mt. Meager GSD'},
    
    {'exposures': ['10.0', '7.5', '7.0', '9.0', '11.0', 'Suiattle River'],
     'colors': [MediumGray] * 5 + [glaciercolor],
     'line_types': ['solid'] * 6,
     'title': 'Suiattle River GSD'},

    {'exposures': ['T4', 'T5', 'T6', 'T7', 'T8', 'Little Tahoma'],
     'colors': [DarkGray] +[LightGray] +[LightGray] +[LightGray] +[DarkGray] + [tahomacolor],
     'line_types': ['solid'] * 6,
     'title': 'Little Tahoma GSD'},
    
    {'exposures': ['KC1', 'KC2', 'KC3', 'KC4', 'KC5', 'KC6', 'KC7', "Kautz Creek"],
     'colors': [DarkGray] * 4 + [LightGray] * 3 + [kautzcolor],
     'line_types': ['solid'] * 8,
     'title': 'Kautz Creek GSD'},
    
    {'exposures': ['MA1', 'MA1B', 'MA2', 'MA2B', 'MA3', 'MA3B', 'Mt. Adams'],
     'colors': [LightGray] * 2 + [MediumGray] * 2 + [DarkGray] * 2 + [adamscolor],
     'line_types': ['solid'] * 7,
     'title': 'Mt. Adams GSD'}
    
  
]
# Plot Figure 8: Coarse fraction grain size distribution, excluding clasts <0.5 cm. Individual exposures in gray. 
run_grain_size_subplots(gsd_data, plot_details)
