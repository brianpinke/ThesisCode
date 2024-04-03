#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 13:04:20 2024

This code models transport-dependent abrasion of a sediment source of given
strength and grain size distribution. Because we aren't certain of the magnitude
of the 'transport-dependent' effect, we do a sensitivity analysis playing with
different values of these variables (k and the alpha multiplier). 

@author: Brian Pinke **Modified From Allison Pfeiffer
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# %% Read in the source material SHRS data  and fix up data to work in script
#os.chdir('C:\\Users\\pinkeb\\OneDrive - Western Washington University\\Thesis\\ThesisCode')
os.chdir('/Users/bpinke/Library/CloudStorage/OneDrive-WesternWashingtonUniversity/Thesis/ThesisCode')

path=os.path.join( os.getcwd(), 'Data/SuiattleFieldData_Combined20182019.xlsx')
# loads in Suiattle data text files: (1) metadata (2) lithology (3) medians
suiattleGSD_data = pd.read_excel (
        path, 
        sheet_name='3 - Grain size df deposit', 
        header = 0)

suiattleGSD_data=suiattleGSD_data.drop(columns=('Row Number'))
suiattleGSD_data=suiattleGSD_data.rename(columns={"Stop Number": "Stop ID"})
suiattleGSD_data['Stop ID'] = suiattleGSD_data['Stop ID'].astype("string")

path=os.path.join( os.getcwd(), 'Data/SuiattleFieldData_Combined20182019.xlsx')
suiattleSHRS_data = pd.read_excel (
        path, 
        sheet_name='4-SHRSdfDeposit', 
        header = 0)

path=os.path.join( os.getcwd(), 'Data/fieldbook_data_2023_USE.xlsx')
# loads in SHRS data text files: (1) metadata (2) lithology (3) medians
field_data = pd.read_excel (
        path, 
        sheet_name='1 - Measurements', 
        header = 0)

#active_df = field_data.copy()

os.chdir(os.getcwd()+'/Figs')

#%% 
# Modify field_data for use here, drop fines, anything = 0, na values, and convert xm to m
field_data = field_data[['Stop ID','Lithology Category','Size (cm)', 'Mean_Median SHRS', 'Extra SHRS']].copy()


field_data = field_data.drop(field_data[field_data['Size (cm)'] == 'F'].index)
field_data = field_data.drop(field_data[field_data['Size (cm)'] == '0'].index) #2809
field_data = field_data[field_data['Size (cm)'].notna()]

# change grain size to meters
field_data[["Size (cm)"]] = field_data[["Size (cm)"]].apply(pd.to_numeric)/100
field_data=field_data.rename(columns={"Size (cm)": "Size (m)"})

# rename to median shrs
#field_data=field_data.rename(columns={"Mean_Median SHRS": "Median SHRS"})



#%% Create Functions

def add_combine_exposures(data,exposures,deposit_name):
    """
    Add combined exposures as a 'Stop ID' to the table
    """
    deposit = data.loc[data['Stop ID'].isin(exposures)]
    deposit['Stop ID'] = deposit_name
    data= pd.concat([data,deposit],axis=0)
    
    return data 


#%% Combine Kautz

field_data= add_combine_exposures(field_data,exposures=['KC1','KC2','KC3','KC4','KC5','KC6','KC7'],
                                deposit_name="Kautz Creek")

#field_data= add_combine_exposures(field_data,exposures=['T2A','T2B','T2C'],
                               # deposit_name="T2")

#field_data= add_combine_exposures(field_data,exposures=['T1','T2','T3'],
                               # deposit_name="Tahoma 2022")

field_data= add_combine_exposures(field_data,exposures=['T4','TNUV1','T5A','T5B','T6','T7','T8'],
                                deposit_name="Tahoma 2023")

#field_data= add_combine_exposures(field_data,exposures=['T1','T2','T4','TNUV1','T5A','T5B','T6','T7','T8'],
                                #deposit_name="Little Tahoma")

field_data= add_combine_exposures(field_data,exposures=['MM1','MM2','MM3','MM4','MM5','MM6','MM7','MM8','MM9', 'MM10', 'MM11'],
                                deposit_name="Mt. Meager")

field_data= add_combine_exposures(field_data,exposures=['MA1','MA1B','MA2','MA2B','MA3','MA3B','MA4','MA4A'],
                                deposit_name="Mt. Adams")


Suiattle_lith_list = ['NV', 'VV']
Adams_lith_list = ['NN', 'NA', 'VN', 'VA']
Meager_lith_list = ['OW', 'HGM','HDV','G','LDV']
Kautz_lith_list = ['G', 'NV', 'V']
Tahoma23_lith_list = ['PN','UG','TV','VV']

#define colors
lightpurple= (0.7450980392156863, 0.7058823529411765, 0.9019607843137255)
darkpurple= (0.5019607843137255, 0.0, 0.5019607843137255)
orange= (1.0, 0.5294117647058824, 0.0)
pink=(1.0, 0.7529411764705882, 0.796078431372549)
green= (0.0, 0.5019607843137255, 0.0)

lighterorange =(1.0, 0.7254901960784313, 0.30196078431372547)
TahomaColors = [darkpurple, orange, lightpurple, lightpurple, 'white']
KautzColors = [pink, darkpurple,lightpurple, 'white']

#%% FOR WEIGHTED VERSION OF MODEL
gsd_data = field_data[['Stop ID','Lithology Category','Mean_Median SHRS']].copy()
gsd_data= gsd_data[gsd_data['Mean_Median SHRS'].notna()]

# Data for Meager
meager_data = {
    'Lithologies': ['LDV', 'G', 'HDV', 'OW', 'HGM'],
    'Fraction': [62.064677, 29.975124, 4.726368, 2.487562, 0.746269]
}

meager_df = pd.DataFrame(meager_data, columns=['Lithologies', 'Fraction'])

meager_df['Fraction'] = (meager_df['Fraction'] * 10).round(0)

# Data for Suiattle
suiattle_data = {
    'Lithologies': ['VV', 'NV', 'PL'],
    'Fraction': [56.043956, 42.124542, 1.831502]
}

suiattle_df = pd.DataFrame(suiattle_data, columns=['Lithologies', 'Fraction'])

suiattle_df['Fraction'] = (suiattle_df['Fraction'] * 10).round(0)

# Data for Tahoma
tahoma_data = {
    'Lithologies': ['PN', 'TV', 'VV', 'UG'],
    'Fraction': [37.500000, 34.375000, 18.923611, 9.201389]
}

tahoma_df = pd.DataFrame(tahoma_data, columns=['Lithologies', 'Fraction'])

tahoma_df['Fraction'] = (tahoma_df['Fraction'] * 10).round(0)

# Data for Kautz
kautz_data = {
    'Lithologies': ['NV', 'V', 'G'],
    'Fraction': [89.189189, 9.009009, 1.801802]
}

kautz_df = pd.DataFrame(kautz_data, columns=['Lithologies', 'Fraction'])

kautz_df['Fraction'] = (kautz_df['Fraction'] * 10).round(0)
kautz_df.at[2, 'Fraction'] += 1

# Data for Adams
adams_data = {
    'Lithologies': ['NA', 'NN', 'VN', 'VA'],
    'Fraction': [49.340369, 31.662269, 9.498681, 9.498681]
}

adams_df = pd.DataFrame(adams_data, columns=['Lithologies', 'Fraction'])

adams_df['Fraction'] = (adams_df['Fraction'] * 10).round(0)

merge_liths = { "NP": 'NV', 'GF': 'PL', 'VNV':'NV', 'VC':'VB', "NAL":"NA", "VB":'VV'} #'G': 'PL' # Define replacements {old_lith: new_lith}

for old_lith, new_lith in merge_liths.items():
    gsd_data.loc[gsd_data['Lithology Category'] == old_lith, 'Lithology Category'] = new_lith


# Calculate the empirical cumulative distribution function (ECDF)
def ecdf(data):
    # Sort the data in ascending order
    x = np.sort(data)
    # Compute the cumulative probabilities for each data point
    y = np.arange(1, len(data) + 1) / len(data)
    return x, y

def create1000SHRSsample(deposit_name, lith_percent_df):
    # Creating the new DataFrame with a dynamic name
    new_df = pd.DataFrame(columns=['Mean_Median SHRS','Lithology Category', 'Stop ID'])
    
    for lithtype in lith_percent_df['Lithologies']:
        # Calculate ECDF for the original data
        x_original, y_original = ecdf(gsd_data.loc[(gsd_data['Stop ID'] == deposit_name) & (gsd_data['Lithology Category'] == lithtype)]['Mean_Median SHRS'])
        
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

Meager_weightedSHRS_distrib = create1000SHRSsample('Mt. Meager', meager_df, )
#Suiattle_weightedSHRS_distrib = create1000SHRSsample('Suiattle', suiattle_df, )
Tahoma_weightedSHRS_distrib = create1000SHRSsample('Tahoma 2023', tahoma_df, )
Kautz_weightedSHRS_distrib = create1000SHRSsample('Kautz Creek', kautz_df, )
Adams_weightedSHRS_distrib = create1000SHRSsample('Mt. Adams', adams_df, ) 	


#%%
### NEED TO CHANGE: deposit_name, lith_list, SHRS, lith to requisite deposit variables.
deposit_name = "Tahoma 2023"
lith_list = Tahoma23_lith_list


active_df = field_data.loc[field_data['Stop ID'] == deposit_name].copy()

# pull GSD data, Set all extra shrs measurements to nan
gsd_df = active_df[['Size (m)', 'Extra SHRS']].copy()
gsd_df.drop(gsd_df[gsd_df['Extra SHRS'] == 'y'].index, inplace = True)

D_source = np.array(gsd_df['Size (m)'].values)

SHRS = Tahoma_weightedSHRS_distrib['Mean_Median SHRS'].values
lith = Tahoma_weightedSHRS_distrib['Lithology Category'].values

# =============================================================================
# #%% choose variables:
#     
# deposit_name = "Meager"
# lith_list = Meager_lith_list
# 
# 
# #%% this is essentially a function, choose deposit name, colors, and liths above and then run.
# 
# 
# if deposit_name == "Suiattle":
#     SHRS = suiattleSHRS_data['SHRS median'].values
#     lith = suiattleSHRS_data['Lithology'].values
#     
#     suiattleGSD_data = suiattleGSD_data.replace(np.nan,0) # most sand grain size marked as "nan"
#     suiattleGSD_data = suiattleGSD_data[suiattleGSD_data['Size (cm)'] != 0] # remove sand... 
# 
#     D_source = suiattleGSD_data['Size (cm)'].values
#     D_source = np.array(D_source)/100 # now in m
# else:
#     active_df = field_data.loc[field_data['Stop ID'] == deposit_name].copy()
# 
#     # pull GSD data, Set all extra shrs measurements to nan
#     gsd_df = active_df[['Size (m)', 'Extra SHRS']].copy()
#     gsd_df.drop(gsd_df[gsd_df['Extra SHRS'] == 'y'].index, inplace = True)
# 
#     D_source = np.array(gsd_df['Size (m)'].values)
#     
#     # pull SHRS and Lith data
#     # drop na shrs and lith values
#     SHRS_Lith_df = active_df[['Lithology Category', 'Median SHRS']].copy()
#     SHRS_Lith_df = SHRS_Lith_df[SHRS_Lith_df['Median SHRS'].notna()]
#     SHRS_Lith_df = SHRS_Lith_df[SHRS_Lith_df['Lithology Category'].notna()]
#     print(SHRS_Lith_df.size)
#     # merge liths and drop liths as neccessary
# 
#     merge_liths = { "NP": 'NV', 'GF': 'PL', 'VNV':'NV', 'VC':'VB', "NAL":"NA", "VB":'VV'} #'G': 'PL' # Define replacements {old_lith: new_lith}
# 
#     for old_lith, new_lith in merge_liths.items():
#         SHRS_Lith_df.loc[SHRS_Lith_df['Lithology Category'] == old_lith, 'Lithology Category'] = new_lith
#     
# 
#     SHRS_Lith_df = SHRS_Lith_df[SHRS_Lith_df['Lithology Category'] != 'A']
#     SHRS_Lith_df = SHRS_Lith_df[SHRS_Lith_df['Lithology Category'] != 'breccia']
#     SHRS_Lith_df = SHRS_Lith_df[SHRS_Lith_df['Lithology Category'] != 'AN']
#     SHRS_Lith_df = SHRS_Lith_df[SHRS_Lith_df['Lithology Category'] != 'A']
#     SHRS_Lith_df = SHRS_Lith_df[SHRS_Lith_df['Lithology Category'] != 'C']
#     SHRS_Lith_df = SHRS_Lith_df[SHRS_Lith_df['Lithology Category'] != 'S']
#     SHRS_Lith_df = SHRS_Lith_df[SHRS_Lith_df['Lithology Category'] != 'sand']
#     SHRS_Lith_df = SHRS_Lith_df[SHRS_Lith_df['Lithology Category'] != 'F']
#     SHRS_Lith_df = SHRS_Lith_df[SHRS_Lith_df['Lithology Category'] != 'P']
# 
#     SHRS = SHRS_Lith_df['Median SHRS'].values
#     lith = SHRS_Lith_df['Lithology Category'].values
#     print(lith.size)
# 
# =============================================================================
#%%

# set lithology classifications, change AN to PN (very few data - non-vesicular)
#data['Lithology Category'] = np.where((data['Lithology Category'] == 'AN'), 'PN', data['Lithology Category'])
#lith = data['Lithology Category'].values


# %%  

ks = [0,15,45] # UNKNOWN coefficient in transport dependent abrasion equation. Let's try these three values, which mimic Attal and Lave conceptual diagram
alphamultipliers = [1,2,4] # UNKNOWN tumbler correction value, literature suggests something in this range.

k_line_style = ['k-','k--','k:']
#fig1,axVol = plt.subplots(3,3,sharex=True,sharey=True,figsize=(7.5,7.5),dpi=600)


letter = ['a','b','c','d','e','f','g','h','i'] # for labeling panels on figure
letter_now = 0

# These nested loops run the transport-dependent abrasion equation for the range of k and alphamultiplier values above

k = 15

alphamultiplier = 2

# %% 
# PART 1: Simple Sternberg abrasion calculation, based on our data.
measured_alphas = alphamultiplier*3.0715*np.exp(-0.136*SHRS) 
density = 923.63*np.log(SHRS) - 1288.2 

#dist_150km = np.arange(0,151) # array to represent distance downstream from source
dist_km = np.arange(0,101)
#dist_150km = dist_65km # MAKING IT = 65km as 150 was not the fig from pfeiffer 2022
Sternb_mass_abradingsourcematerial = np.exp(
        -np.expand_dims(measured_alphas,axis=0) 
        * (np.expand_dims(dist_km,axis=1))
        ) # classic abrasion calculation 

Sternb_volume_abradingsourcematerial = (Sternb_mass_abradingsourcematerial/density)
Sternb_NormVolume_downstream = np.sum(Sternb_volume_abradingsourcematerial,axis = 1)/np.sum(Sternb_volume_abradingsourcematerial[0,:])
Sternb_NormMass_downstream = np.sum(Sternb_mass_abradingsourcematerial,axis=1)/np.sum(Sternb_mass_abradingsourcematerial[0,:])

if alphamultiplier == 1:
    Sternb_NormVolume_downstream_alpha1 = np.sum(Sternb_volume_abradingsourcematerial,axis = 1)/np.sum(Sternb_volume_abradingsourcematerial[0,:])
    Sternb_NormMass_downstream_alpha1 = np.sum(Sternb_mass_abradingsourcematerial,axis=1)/np.sum(Sternb_mass_abradingsourcematerial[0,:])           

d_alpha = 0.01 # np.median(measured_alphas)/3
D_50 = 0.14*np.exp(-d_alpha*dist_km)

# %% THE NEXT STEP: low density grains higher transport rate--> higher abrasion rate
n = 10 # number of repeated distributions
n = 1 # SET TO 1 FOR WEIGHTED VERSION BECAUSE INPUT IS 1000 SHRS AND LITH ALREADY
density_n = 923.63*np.log(np.tile(SHRS,n)) - 1288.2 # NOTE: this regression ignores 2 VV outlier points. Submerged density (to match mass measurements)
SHRS_n = np.tile(SHRS,n)
lith_n = np.tile(lith,n)
D_matrix = np.full([np.size(dist_km),n*np.size(SHRS)],0.)
Mass_matrix = np.full([np.size(dist_km),n*np.size(SHRS)],0.)
Vol_matrix = np.full([np.size(dist_km),n*np.size(SHRS)],0.)
print(SHRS_n, lith_n, Vol_matrix)
D_matrix[0,:] = np.random.choice(D_source,np.shape(SHRS_n)) # randomly sample from the grain size data..
Vol_matrix[0,:] = 1.0 # all 1.0 m3 "parcels"
Mass_matrix[0,:] = Vol_matrix[0,:]*density_n

# ASSUMPTION: we're going to abrade these without considering selective transport. Represents a single upstream source (not terrible here), with no channel storage/aggradation/lag

tauS_50 = 0.045 # a representative Shields stress for transport of D50..
rho = 1000 
g = 9.81

tau = np.full(np.shape(dist_km),fill_value=np.nan)
mean_SHRS = np.full(np.shape(dist_km),fill_value=np.nan)
# %% Calculate abrsion at each downstream distance
for i in range(np.size(dist_km[1:])):
    # let's assume that a representative tau is twice that required to transport D50 of mean density
    tau[i] = 2.0*tauS_50*((np.mean(density_n)-rho)*g*D_50[i]) 
    tauS = tau[i]/((density_n-rho)*g*D_matrix[i,:])  
    tauS_tauSc = tauS/tauS_50
    
    alpha = alphamultiplier* 3.0715*np.exp(-0.136*SHRS_n) # the constant alpha
    
    alpha_HighTransport = alpha*(k
                           *(density_n-rho)/rho
                           *D_matrix[i,:]
                           *(tauS_tauSc-3.3) 
                           + 1
                           )
    
    alpha[tauS_tauSc>3.3]=alpha_HighTransport[tauS_tauSc>3.3]
    alpha[D_matrix[i,:]<0.002]=0 # don't abrade sand

    # Now, 
    Mass_matrix[i+1,:] = (
            Mass_matrix[i,:]
            *np.exp(
                    -(dist_km[i+1]-dist_km[i])
                    *alpha
                    )
            )
    
    D_matrix[i+1,:] = D_matrix[i,:]*(
            (Mass_matrix[i+1,:]/density_n)
            /(Mass_matrix[i,:]/density_n)
            )**(1/3)
    
    # Volumetrically weighted mean strength at each distance ds
    vol_fraction = (Mass_matrix[i,:]/density_n)/np.sum(Mass_matrix[i,:]/density_n)
    mean_SHRS[i] = np.sum(SHRS_n*vol_fraction)
    

Vol_matrix = Mass_matrix/density_n
Vol_matrix[D_matrix <=0.002] = 0 #When we measure bed material, we don't measure sand. 
mass_norm = np.sum(Mass_matrix,axis=1)/np.sum(Mass_matrix[0,:])
vol_norm = np.sum(Vol_matrix,axis=1)/np.sum(Vol_matrix[0,:])

# %% How does lith by volume change?
# list of unique liths
#lith_list = list(set(lith))
#lith_list = ['NV','VV']
   
# Dictionary to store results
lith_vol = {}

vol_tot = np.sum(Vol_matrix,axis=1) #total volume of grains at each distance
# Loop over the lithologies and calculate sums
for lith_type in lith_list:
    lith_vol[f'vol_{lith_type}'] = np.sum(Vol_matrix[:, np.tile(lith, n) == lith_type], axis=1)

# or...
# for i in lith_list:
    #print( lith_frac[f'vol_{i}'])

# =============================================================================
#         # %% Which grains disappear?
#         
#         # Dictionary to store results
#         lith_frac = {}
#         
#         grain_frac_lost_end = Vol_matrix[0,:]-Vol_matrix[-1,:]/Vol_matrix[0,:]
#         # Loop over the lithologies and calculate fractions
#         for i in lith_list:
#             lith_frac[f'{i}_frac_lost_end'] = grain_frac_lost_end[lith_n == i]
#         
#         # call via -> lith_frac['XX_frac_lost_end'] 
#         
# =============================================================================
        
# %% Plot

## ***CHANGE FOR LITHOLOGIC VARIABLES***
vol_initial = np.sum(Vol_matrix[0,:])
vol_lith_products = 0
for lith_type in lith_list:
    vol_lith_products += lith_vol[f'vol_{lith_type}']
vol_products = vol_initial-(vol_lith_products)

# Create a list of values where each value in lith_vol is divided by vol_initial
y_data = [lith_vol[f'vol_{lith_type}'] / vol_initial for lith_type in lith_list] + [vol_products / vol_initial]

labels = lith_list

colors = TahomaColors
# Ensure the number of unique lithologies is less than or equal to the number of colors

num_colors_needed = min(len(y_data), len(colors))

# Create a new list of colors based on the number of colors needed
new_colors = colors[:num_colors_needed]
# Make the last color white
new_colors[-1] = 'white'

plt.stackplot(dist_km,
              
              *y_data, 
              labels=labels,
              colors=new_colors,
              ) #, edgecolor='black', linewidth=1 looks pretty bad though....

plt.ylim(0,1)
#axVol[j,u].set(xlim=[0,150])
plt.xlim(0,100) # changed to x axis of 60
#plt.text(30,0.85,'k='+str(k)+', '+
                #str(alphamultiplier)+r'$\alpha$') #75, 0.85
######plt.text(5,.9,letter[letter_now],size=14)  #5, 0.9 # THIS PRINTS A B C D for each subplot
letter_now +=1
    
# %% Save figure 2
            
plt.figure(1)        
plt.title(deposit_name)
plt.ylabel('Volumetric fraction remaining')
plt.xlabel('Distance from source (km)')

# =============================================================================
# plt.annotate(r'increasing baseline $\alpha$',
#             #xy=(0.8,0.845),xycoords='figure fraction',
#             #xytext=(0.15,0.84),textcoords = 'figure fraction',
#             xy=(0.8,0.915),xycoords='figure fraction',
#             xytext=(0.3,0.91),textcoords = 'figure fraction',
#             arrowprops= dict(arrowstyle="->")
#             )
# 
# plt.annotate(r'increasingly transport-dependent $\alpha$',
#              #xy=(0.875,0.15),xycoords='figure fraction',
#              #xytext=(0.87,0.35),textcoords = 'figure fraction',rotation=-90,
#              xy=(0.93,0.15),xycoords='figure fraction',
#              xytext=(0.92,0.35),textcoords = 'figure fraction',rotation=-90,
#             arrowprops= dict(arrowstyle="->")
#             )
# =============================================================================

#plt.legend(bbox_to_anchor=(1.48, 3.63))
#plt.legend()
# Get the handles and labels of the plotted areas
handles, labels = plt.gca().get_legend_handles_labels()

# Reverse the order of labels and handles
reversed_labels = labels[::-1]
reversed_handles = handles[::-1]

# Create a legend with reversed order
plt.legend(reversed_handles, reversed_labels)

plt.savefig('Fig2_BaseTransportAbrasion' + deposit_name +'.png', dpi=400, bbox_inches="tight")