#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 20:02:30 2023

@author: bpinke
"""

# loads necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns 

#os.chdir('C:\\Users\\pinkeb\\OneDrive - Western Washington University\\Thesis\\ThesisCode')
os.chdir('/Users/bpinke/Library/CloudStorage/OneDrive-WesternWashingtonUniversity/Thesis/ThesisCode')

path=os.path.join( os.getcwd(), 'fieldbook_data_2023_USE.xlsx')
# loads in SHRS data text files: (1) metadata (2) lithology (3) medians
field_data = pd.read_excel (
        path, 
        sheet_name='1 - Measurements', 
        header = 0)

#gsd_data = field_data.copy()

path=os.path.join( os.getcwd(), 'SuiattleFieldData_Combined20182019.xlsx')
# loads in SHRS data text files: (1) metadata (2) lithology (3) medians
suiattleSHRS_data = pd.read_excel (
        path, 
        sheet_name='4-SHRSdfDeposit', 
        header = 0)

#Stop ID, Lithology Category, Mean_Median SHRS
#Site Num, Lithology, SHRS median
field_data = field_data[['Stop ID','Lithology Category','Mean_Median SHRS']].copy()
suiattleSHRS_data = suiattleSHRS_data[['Site Num', 'Lithology','SHRS median']].copy()

suiattleSHRS_data=suiattleSHRS_data.rename(columns={"Site Num": "Stop ID", 'Lithology': 'Lithology Category','SHRS median': 'Mean_Median SHRS'})
suiattleSHRS_data['Stop ID'] = suiattleSHRS_data['Stop ID'].astype("string")

gsd_data= pd.concat([field_data,suiattleSHRS_data],axis=0)
gsd_data= gsd_data[gsd_data['Mean_Median SHRS'].notna()]

    #%%
def process_size(x):
    """
    x is the size cm of all data collected FOR ONE SITE
    1. set 0 to nans and drop nans
    2. sort values
    3. cm/100= m 

    """
    
    size_numeric = pd.to_numeric(x,errors="coerce") #set non numerics to nan 
    size_numeric[size_numeric==0] = np.nan #set 0 to nan 
    
    return size_numeric.dropna().sort_values()/100 #drop nans, sort, cm-->m 

# =============================================================================
# 
# def run_shrs_by_exposure_plot(gsd_data,exposures):
#     
#     """
#     Groubpy Stop ID for each stop id apply the process size function for all sizes 
#     Then we choose the stop ids we care about 
#     """
#     n= len(exposures)
#     
#     data= gsd_data.loc[gsd_data['Stop ID'].isin(exposures)]
#     
#     sns.boxplot(data=data,x='Stop ID',y='Mean_Median SHRS',orient='v')
#     
# 
# 
# def run_shrs_by_lithology_plot(gsd_data,lithology):
#     
#     """
#     Groubpy Stop ID for each stop id apply the process size function for all sizes 
#     Then we choose the stop ids we care about 
#     """
#     n= len(exposures)
#     
#     data= gsd_data.loc[gsd_data['Lithology Category'].isin(lithology)]
#     
#     sns.boxplot(data=data,x='Lithology Category',y='Mean_Median SHRS',orient='v')
#     
#     
# =============================================================================

    
def add_combine_exposures(data,exposures,deposit_name):
    """
    Add combined exposures as a site to table
    """
    deposit = data.loc[data['Stop ID'].isin(exposures)]
    deposit['Stop ID'] = deposit_name
    data= pd.concat([data,deposit],axis=0)
    
    return data 

def run_shrs_by_plot(gsd_data,by,data,colors,title=None, xlabel=None):
    #print(dict(zip(data,colors))['KC1'])
    if by =="Stop ID":
        xlabel="Exposure"
    if by =='Lithology Category':
        xlabel="Lithology"
        
    data= gsd_data.loc[gsd_data[by].isin(data)]
    
    sns.violinplot(data=data,x=by,y='Mean_Median SHRS', saturation=1, palette=colors).set(title=title, xlabel=xlabel, ylabel='SHRS')
    
    plt.show()
    
    #pltviolin = data.violinplot(column = 'Mean_Median SHRS', by=by, return_type = 'both', patch_artist = True)
    
    #sns.boxplot(data=data,x=by,y='Mean_Median SHRS',) #palette=dict(zip(data,colors))
    
    #bp_dict= data.boxplot(column='Mean_Median SHRS',by=by,return_type='both',patch_artist = True)
    
# =============================================================================
#     for row_key, (ax,row) in bp_dict.iteritems():
#         ax.set_xlabel('')
#         for i,box in enumerate(row['boxes']):
#             box.set_facecolor(colors[i])
# =============================================================================

# combine exposures as needed for plotting
         
gsd_data= add_combine_exposures(gsd_data,exposures=['KC1','KC2','KC3','KC4','KC5','KC6','KC7'],
                                deposit_name="Kautz")

gsd_data= add_combine_exposures(gsd_data,exposures=['T2A','T2B','T2C'],
                                deposit_name="T2")

gsd_data= add_combine_exposures(gsd_data,exposures=['T1','T2','T3'],
                                deposit_name="Tahoma 2022")

gsd_data= add_combine_exposures(gsd_data,exposures=['T4','TNUV1','T5A','T5B','T6','T7','T8'],
                                deposit_name="Tahoma 2023")

gsd_data= add_combine_exposures(gsd_data,exposures=['T1','T2','T4','TNUV1','T5A','T5B','T6','T7','T8'],
                                deposit_name="Tahoma")

gsd_data= add_combine_exposures(gsd_data,exposures=['MM1','MM2','MM3','MM4','MM5','MM6','MM7','MM8','MM9', 'MM10', 'MM11'],
                                deposit_name="Mt. Meager")

gsd_data= add_combine_exposures(gsd_data,exposures=['MA1','MA1B','MA2','MA2B','MA3','MA3B','MA4','MA4A'],
                                deposit_name="Mt. Adams")

gsd_data= add_combine_exposures(gsd_data,exposures=['9','10'],
                                deposit_name="Suiattle")

# plot SHRS by exposure


run_shrs_by_plot(gsd_data,by="Stop ID", data=['KC1','KC2','KC3','KC4','KC5','KC6','KC7',"Kautz"],
                    colors =['blue','purple','red','orange','green','gray','black','maroon'],
                    title='Kautz Creek SHRS')

run_shrs_by_plot(gsd_data,by="Stop ID", data=['T1', 'T2', 'T3', "Tahoma 2022"],
                    colors =['blue','purple','red','orange'],
                    title='Tahoma 2022 SHRS')

run_shrs_by_plot(gsd_data,by="Stop ID", data=['T4','TNUV1','T5A','T5B','T6','T7','T8','Tahoma 2023'],
                    colors =['blue','purple','red','orange', 'green', 'gray', 'black', 'maroon'],
                    title='Tahoma 2023 SHRS')

run_shrs_by_plot(gsd_data,by="Stop ID", data=['MM1','MM2','MM3','MM4','MM5','MM6','MM7','MM8','MM9', 'MM10', 'MM11','Mt. Meager'],
                    colors =['blue','purple','red','orange', 'green', 'gray', 'black', 'maroon', 'pink','olive','cyan','greenyellow'],
                    title='Mt. Meager SHRS')

run_shrs_by_plot(gsd_data,by="Stop ID", data=['MA1','MA1B','MA2','MA2B','MA3','MA3B','MA4','MA4A','Mt. Adams'],
                    colors =['blue','purple','red','orange', 'green', 'gray', 'black', 'maroon', 'pink'],
                    title='Mt. Adams SHRS')

run_shrs_by_plot(gsd_data,by="Stop ID", data=['9', '10', 'Suiattle'],
                    colors =['blue','purple','red','orange', 'green', 'gray'],
                    title='Suiattle SHRS')

run_shrs_by_plot(gsd_data,by="Stop ID", data=['Tahoma', 'Kautz', 'Mt. Meager', 'Mt. Adams', 'Suiattle'],
                    colors =['blue','purple','red','orange', 'green', 'gray'],
                    title='All Deposits SHRS')

# plot SHRS by Lithology per Deposit
#Kautzdf= gsd_data['Stop ID']

run_shrs_by_plot(gsd_data.loc[gsd_data['Stop ID']=='Kautz'],by="Lithology Category", data=['NV', 'G','V', 'A',"Kautz"],
                    colors =['blue','purple','red','orange','green','gray','black','maroon'],
                    title='Kautz Creek SHRS by Lithology')

run_shrs_by_plot(gsd_data.loc[gsd_data['Stop ID']=='Tahoma 2023'], by="Lithology Category", data=['AN', 'PN','VV', 'TV',"breccia"],
                    colors =['blue','purple','red','orange','green','gray','black','maroon'],
                    title='Tahoma23 SHRS by Lithology')

run_shrs_by_plot(gsd_data.loc[gsd_data['Stop ID']=='Mt. Meager'],by="Lithology Category", data=['LDV','G','HDV','OW','HGM','H'],
                    colors =['blue','purple','red','orange','green','gray','black','maroon'],
                    title='Mt. Meager SHRS by Lithology')

run_shrs_by_plot(gsd_data.loc[gsd_data['Stop ID']=='Mt. Adams'],by="Lithology Category", data=['NN','NAL','VN','VA','C','P'],
                    colors =['blue','purple','red','orange','green','gray','black','maroon'],
                    title='Mt. Adams SHRS by Lithology')

run_shrs_by_plot(gsd_data.loc[gsd_data['Stop ID']=='Suiattle'],by="Lithology Category", data=['VV','NV'],
                    colors =['blue','purple','red','orange','green','gray','black','maroon'],
                    title='Suiattle SHRS by Lithology')

#run_shrs_by_plot(gsd_data,by='Lithology Category',data=['KC1','KC2','KC3','KC4','KC5','KC6','KC7'])


#gsd_data['Stop ID']["Kautz"].groupby("lithology Category").unique_values()

print('DONE!!!')
#print(gsd_data.loc[gsd_data['Stop ID']=="Kautz"]["Lithology Category"].unique())


