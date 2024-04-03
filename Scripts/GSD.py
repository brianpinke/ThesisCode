# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 20:44:04 2023

@author: pinkeb
"""

# loads necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

#os.chdir('C:\\Users\\pinkeb\\OneDrive - Western Washington University\\Thesis\\ThesisCode')
os.chdir('/Users/bpinke/Library/CloudStorage/OneDrive-WesternWashingtonUniversity/Thesis/ThesisCode')

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
Red = (1.0, 0.0, 0.16)
OrangeRed = (1.0, 0.35, 0.0)
Orange = (1.0, 0.65, 0.0)
YellowOrange = (1.0, 0.85, 0.0)
YellowGreen = (0.65, 1.0, 0.0)
Green = (0.0, 1.0, 0.0)
BlueGreen = (0.0, 0.65, 0.35)
Cyan = (0.0, 1.0, 1.0)
Blue = (0.0, 0.0, 1.0)
Indigo = (0.3, 0.0, 0.6)
Brown = (0.6, 0.3, 0.0)
Black = (0.0, 0.0, 0.0)

LightGray = (0.8274509803921568, 0.8274509803921568, 0.8274509803921568)
MediumGray = (0.6627450980392157, 0.6627450980392157, 0.6627450980392157)
DarkGray = (0.4117647058823529, 0.4117647058823529, 0.4117647058823529)

lightpurple= (0.7450980392156863, 0.7058823529411765, 0.9019607843137255)
darkpurple= (0.5019607843137255, 0.0, 0.5019607843137255)
orange= (1.0, 0.5294117647058824, 0.0)
pink=(1.0, 0.7529411764705882, 0.796078431372549)
green= (0.0, 0.5019607843137255, 0.0)
lighterorange = (1.0, 0.7254901960784313, 0.30196078431372547)

meagercolor = (0.4, 0.2, 0.0) #dark brown
glaciercolor = (0.0, 0.5019607843137255, 0.5019607843137255) #teal
tahomacolor = (0.8627450980392157, 0.0784313725490196, 0.23529411764705882) #steel blue
kautzcolor = (0.4, 0.6, 0.8) #crimson
adamscolor = (0.5019607843137255, 0.5019607843137255, 0.0) #olive

# change font size
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
    
    combines exposures in a given dataset given a new "deposit_name", appended to existing dataframe.
    
    """
    deposit = data.loc[data['Stop ID'].isin(exposures)]
    deposit['Stop ID'] = deposit_name
    data= pd.concat([data,deposit],axis=0)
    
    return data 

def process_size(x):
    """
    x is the size cm of all data collected FOR ONE SITE
    1. set 0 to nans and drop nans
    2. sort values
    3. cm/100= m 

    """
    #set non numerics to nan 
    size_numeric = pd.to_numeric(x,errors="coerce") 
    
    #set 0 to nan 
    size_numeric[size_numeric==0] = np.nan 
    
    #drop nans, sort, cm-->m 
    return size_numeric.dropna().sort_values()/100 

def run_grain_size_plot(data,exposures,colors=None,line_types=None,title=None):
    
    """
    data: df of interest
    exposures:s list of strings of exposures (stop ID column) to be used in plot
    colors: list of strings of colors. Must be equal number of colors as exposures. nth color is used for nth exposure.
    line_types: list of strings of line types. Must be equal number of line types as exposures. nth line type is used for nth exposure.
    title: default is none, give a string if desired.
        
    Plot cumulative percent finer gsd line for each exposure/deposit specified in the given dataset "Stop ID" column. Grain size is given as cm and this code changes it to m.
    """
    
    #Groups the given df by exposures present in the Stop ID column and runs "process size" function on the data to clean the gsd for plotting.
    GsdBySite = data.groupby('Stop ID').agg({'Size (cm)':process_size}).loc[exposures]
    
    #plot line for each exposure (site), array of all sizes in that site, itterate through array - each size
    i=0
    for site,size in GsdBySite.iterrows():  
        #print(site,size)
        # return evenly spaced numbers between 0 and 100 for the count of variables in each Size array
        count = np.linspace(0,100,len(size[0])) 
        plt.semilogx(size[0],count,
                    label = str(site),color=colors[i],linestyle=line_types[i])
        i+=1
        
        
    plt.xlabel('Grain Size (m)')
    plt.ylabel('Cumulative % Finer')
    plt.ylim([0,100])
    plt.xlim(0.0035229004419047457, 7.806067884544429)
    #plt.legend()
    #plt.title(title)
    
    plt.savefig("GSD"+title+".png", dpi=400, bbox_inches="tight")
    plt.show()
    
def get_DX(exposure, percentile=50, df=None):
    """
    exposure: exposures in Stop ID of interest.
    percentile: desired percentile of interest. Default is 50%, give as a number 0-100.
    df: df of interest, default is gsd_data. Give different df if desired. must have "Stop ID"  and Size (cm) columns with gsd in cm.
    
    Prints in the console and returns the Dx of an exposure for a given X percentile in meters.

    """
    if df == None:
        df = gsd_data
        
    dx = df.loc[df['Stop ID']==exposure]["Size (cm)"]
    dx = process_size(dx)
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
    
    # counts number of fines found in the exposure and total gsd measurements
    fines_count= dx.value_counts()['F']
    total_count = len(dx.index)
    percent_fines = fines_count / total_count * 100
    percent_fines = round(percent_fines, 2)
    
    print("The percent fines of", exposure, "is:", percent_fines, "%")
    return percent_fines
    
# =============================================================================
# def run_violin_by_plot(gsd_data,by,exposures,colors,title=None, xlabel=None):
#     """
#     returns a violin plot of the gsd for a given set of exposures.
#     """
# 
#  
#     if by =="Stop ID":
#         xlabel="Exposure"
#     if by =='Lithology Category':
#         xlabel="Lithology"
# 
#     grain_data = gsd_data.drop(gsd_data[gsd_data['Size (cm)'] == 'F'].index)
#     grain_data = grain_data.drop(grain_data[grain_data['Size (cm)'] == '0'].index)
#     grain_data = grain_data[grain_data['Size (cm)'].notna()]
#     grain_data[["Size (cm)"]] = grain_data[["Size (cm)"]].apply(pd.to_numeric)/100
#     print(grain_data)
#                                
#         
#     data= grain_data.loc[grain_data[by].isin(exposures)]
#     
#     sns.violinplot(data=data,x=by,y='Size (cm)', saturation=1, palette=colors).set(title=title, xlabel=xlabel, ylabel='Grain Size (m)')
#     plt.yscale('log')
#     
#     plt.savefig("GSD_violin"+title+".png")
#     plt.show()
# =============================================================================
    
#%%  
# prep dfs for plotting and analysis

# prep suiattle df for merge with 2023 data
suiattleGSD_data=suiattleGSD_data.drop(columns=('Row Number'))
suiattleGSD_data=suiattleGSD_data.rename(columns={"Stop Number": "Stop ID"})
suiattleGSD_data['Stop ID'] = suiattleGSD_data['Stop ID'].astype("string")

# merge 2023 and suiattle data
gsd_data= pd.concat([field_data2023,suiattleGSD_data],axis=0)
    
#Set all extra shrs measurements to nan
gsd_data.drop(gsd_data[gsd_data['Extra SHRS'] == 'y'].index, inplace = True)

# combine exposures as needed for plotting
gsd_data= add_combine_exposures(gsd_data,exposures=['KC1','KC2','KC3','KC4','KC5','KC6','KC7'],
                                deposit_name="Kautz Creek")

gsd_data= add_combine_exposures(gsd_data,exposures=['T2A','T2B','T2C'],
                                deposit_name="T2")

gsd_data= add_combine_exposures(gsd_data,exposures=['T1','T2','T3'],
                                deposit_name="Tahoma 2022")

gsd_data= add_combine_exposures(gsd_data,exposures=['T4','TNUV1','T5A','T5B','T6','T7','T8'],
                                deposit_name="Tahoma 2023")

gsd_data= add_combine_exposures(gsd_data,exposures=['T1','T2','T4','TNUV1','T5A','T5B','T6','T7','T8'],
                                deposit_name="Little Tahoma")

gsd_data= add_combine_exposures(gsd_data,exposures=['MM1','MM2','MM3','MM4','MM5','MM6','MM7','MM8','MM9', 'MM10', 'MM11'],
                                deposit_name="Mt. Meager")

gsd_data= add_combine_exposures(gsd_data,exposures=['MA1','MA1B','MA2','MA2B','MA3','MA3B','MA4','MA4A'],
                                deposit_name="Mt. Adams")

gsd_data= add_combine_exposures(gsd_data,exposures=['10.0','7.5','7.0','9.0','11.0'],
                                deposit_name="Suiattle River")

# set up for suiattle %fines
gsd_data.loc[(gsd_data['Stop ID'] == 'Suiattle River') & (gsd_data['Lithology Category'].isin(['F', 'S', 'sand'])), 'Size (cm)'] = 'F'

#%%
# Plot gsd datafor each deposit and all deposits.

run_grain_size_plot(gsd_data,exposures=['MM1','MM2','MM3','MM4','MM5','MM6','MM7','MM8','MM9', 'MM10', 'MM11','Mt. Meager'],
                    colors =[MediumGray,MediumGray,MediumGray,MediumGray,MediumGray,MediumGray,MediumGray,MediumGray,MediumGray,MediumGray,MediumGray,meagercolor],
                    line_types=['solid','solid','solid','solid','solid','solid','solid','solid','solid','solid','solid','solid']
                    ,title='Mt. Meager GSD')

run_grain_size_plot(gsd_data,exposures=['10.0','7.5','7.0','9.0','11.0','Suiattle River'],
                    colors =[MediumGray,MediumGray,MediumGray,MediumGray,MediumGray, glaciercolor],
                    line_types=['solid','solid','solid','solid','solid','solid']
                    ,title='Suiattle River GSD')

run_grain_size_plot(gsd_data,exposures=['T1','T2','T4','TNUV1','T5A','T5B','T6','T7','T8','Little Tahoma'],
                    colors =[MediumGray,MediumGray,MediumGray,MediumGray,MediumGray,MediumGray,MediumGray,MediumGray,MediumGray,tahomacolor],
                    line_types=['solid','solid','solid','solid','solid','solid','solid','solid','solid', 'solid']
                    ,title='Little Tahoma GSD')

run_grain_size_plot(gsd_data,exposures=['KC1','KC2','KC3','KC4','KC5','KC6','KC7',"Kautz Creek"],
                    colors =[DarkGray,DarkGray,DarkGray,DarkGray, LightGray,LightGray, LightGray, kautzcolor],
                    line_types=['solid','solid','solid','solid','solid','solid','solid','solid']
                    ,title='Kautz Creek GSD')

run_grain_size_plot(gsd_data,exposures=['MA1','MA1B','MA2','MA2B','MA3','MA3B','MA4','MA4A','Mt. Adams'],
                    colors =[MediumGray,MediumGray,MediumGray,MediumGray,MediumGray,MediumGray,MediumGray,MediumGray,adamscolor],
                    line_types=['solid','solid','solid','solid','solid','solid','solid','solid','solid']
                    ,title='Mt. Adams GSD')

run_grain_size_plot(gsd_data,exposures=[ 'Mt. Meager', 'Suiattle River', 'Little Tahoma', 'Kautz Creek','Mt. Adams'],
                    colors =[meagercolor,glaciercolor, tahomacolor, kautzcolor, adamscolor],
                    line_types=['solid','solid','solid','solid','solid']
                    ,title='All Deposits GSD')

#print(gsd_data['Stop ID'].unique())

#%%
# get %fines of each deposit...
percent_fines(gsd_data, "Mt. Meager")
percent_fines(gsd_data, "Suiattle River")
percent_fines(gsd_data, "Little Tahoma")
percent_fines(gsd_data, "Kautz Creek")
percent_fines(gsd_data, "Mt. Adams")

#%%
# Compare tahoma 2022 vs 2023
run_grain_size_plot(gsd_data,exposures=['T1', 'T2', 'T3', "Tahoma 2022"],
                    colors =[Red, Blue, Green, Black],
                    line_types=['solid','solid','solid','dashed']
                    ,title='Tahoma 2022 GSD')

run_grain_size_plot(gsd_data,exposures=['T4','TNUV1','T5A','T5B','T6','T7','T8','Tahoma 2023'],
                    colors =[Red, Orange, YellowGreen, Green, Blue, Indigo, Brown, Black],
                    line_types=['solid','solid','solid','solid','solid','solid','solid','dashed']
                    ,title='Tahoma 2023 GSD')

#%%
# Examining hummmocks vs non hummocks at mt. meager and little tahoma
# Examining location specific similarities of Mt. Adams and Kautz Creek

# plot Mt. Meager data hummock vs not hummock MM6? shear?

#hummock mound: MM3, MM4, MM8, MM10, MM11

#mass exposure: MM1, MM2, MM5, MM6, MM7, MM9

gsd_data= add_combine_exposures(gsd_data,exposures=['MM1','MM2','MM5','MM6','MM7','MM9'],
                                deposit_name="Meager Scarps")

gsd_data= add_combine_exposures(gsd_data,exposures=['MM3','MM4','MM8','MM10','MM11'],
                                deposit_name="Meager Hummocks")

run_grain_size_plot(gsd_data,exposures=['MM1','MM2','MM5','MM6','MM7','MM9','MM3','MM4','MM8', 'MM10', 'MM11','Meager Scarps','Meager Hummocks'],
                    colors =['blue','blue','blue','blue', 'blue', 'blue', 'red', 'red', 'red','red','red','black','green'],
                    line_types=['dashed','dashed','dashed','dashed','dashed','dashed','dashed','dashed','dashed','dashed','dashed','solid','solid']
                    ,title='Meager Scarps vs. Hummocks GSD')


# plot Tahoma data hummock vs not hummock
#hummock mound: T5A, T5B, T6, T7

#mass exposure: T4, T8, 

gsd_data= add_combine_exposures(gsd_data,exposures=['T4','T8'],
                                deposit_name="Tahoma Scarps")

gsd_data= add_combine_exposures(gsd_data,exposures=['T5A', 'T5B','T6','T7'],
                                deposit_name="Tahoma Hummocks")

run_grain_size_plot(gsd_data,exposures=['T4', 'T8', 'T5A','T5B', 'T6','T7','Tahoma Scarps', "Tahoma Hummocks"],
                    colors =['blue','blue','red','red', 'red','red','black','green'],
                    line_types=['dashed','dashed','dashed','dashed','dashed','dashed','solid','solid']
                    ,title='Tahoma Scarps vs Hummocks GSD')


# plot proximal, medial, distal for kautz data

#gsd_data= add_combine_exposures(gsd_data, exposures=['KC1'],
         #                       deposit_name='Proximal')

gsd_data= add_combine_exposures(gsd_data, exposures=['KC1','KC2','KC3','KC4'],
                                deposit_name='Medial')

gsd_data= add_combine_exposures(gsd_data, exposures=['KC5','KC6','KC7'],
                                deposit_name='Distal')

run_grain_size_plot(gsd_data,exposures=['KC1','KC2','KC3','KC4','KC5','KC6','KC7','Medial','Distal'], #proximal
                    colors =['blue','red','red','red','green','green','green','blue','red','green'],
                    line_types=['dashed','dashed','dashed','dashed','dashed','dashed','dashed','solid','solid','solid']
                    ,title='Kautz Creek GSD by Distance')

percent_fines(gsd_data, "Medial")
percent_fines(gsd_data, "Distal")

# plot Mt. Adams data by Salt Creek Valley outer west, inner west, inner east, outer east

gsd_data= add_combine_exposures(gsd_data, exposures=['MA1','MA1B'],
                                deposit_name='Outer West')

gsd_data= add_combine_exposures(gsd_data, exposures=['MA2','MA2B'],
                                deposit_name='Inner West')

gsd_data= add_combine_exposures(gsd_data, exposures=['MA3','MA3B'],
                                deposit_name='Inner East')

gsd_data= add_combine_exposures(gsd_data, exposures=['MA4','MA4A'],
                                deposit_name='Outer East')

run_grain_size_plot(gsd_data,exposures=['MA1','MA1B','MA2','MA2B','MA3','MA3B','MA4','MA4A','Outer West','Inner West','Inner East','Outer East'],
                    colors =['blue','blue','red','red', 'green', 'green', 'orange', 'orange', 'blue','red','green','orange'],
                    line_types=['dashed','dashed','dashed','dashed','dashed','dashed','dashed','dashed','solid','solid','solid','solid']
                    ,title='Mt. Adams GSD by Salt Creek Valley')

#%%


# =============================================================================
# #%%
# #plot gsd by violin plots
# 
# 
# 
# run_violin_by_plot(gsd_data,by="Stop ID", exposures=['Kautz Creek','Little Tahoma','Mt. Adams','Mt. Meager','Suiattle River'],
#                     colors =['blue','purple','red','orange','green'],
#                     title='All Deposits GSD')
# =============================================================================


#%%
# grab DX 16, 25, 50, 84, 95 of the exposures

get_DX("Kautz Creek",16)
get_DX("Kautz Creek",25)
get_DX("Kautz Creek",50)
get_DX("Kautz Creek",84)
get_DX("Kautz Creek",95)

get_DX("Little Tahoma",16)
get_DX("Little Tahoma",25)
get_DX("Little Tahoma",50)
get_DX("Little Tahoma",84)
get_DX("Little Tahoma",95)

get_DX("Mt. Meager",16)
get_DX("Mt. Meager",25)
get_DX("Mt. Meager",50)
get_DX("Mt. Meager",84)
get_DX("Mt. Meager",95)

get_DX("Mt. Adams", 16)
get_DX("Mt. Adams", 25)
get_DX("Mt. Adams", 50)
get_DX("Mt. Adams", 84)
get_DX("Mt. Adams", 95)

get_DX("Suiattle River", 16)
get_DX("Suiattle River", 25)
get_DX("Suiattle River", 50)
get_DX("Suiattle River", 84)
get_DX("Suiattle River", 95)



print('DONE!!!!')
