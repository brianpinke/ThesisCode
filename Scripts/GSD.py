# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 20:44:04 2023

@author: pinkeb
"""

# loads necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

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
suiattleGSD_data = pd.read_excel (
        path, 
        sheet_name='3 - Grain size df deposit', 
        header = 0)

suiattleGSD_data=suiattleGSD_data.drop(columns=('Row Number'))
suiattleGSD_data=suiattleGSD_data.rename(columns={"Stop Number": "Stop ID"})
suiattleGSD_data['Stop ID'] = suiattleGSD_data['Stop ID'].astype("string")

gsd_data= pd.concat([field_data,suiattleGSD_data],axis=0)
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


def run_grain_size_plot(data,exposures,colors=None,line_types=None,title=None):
    
    """
    Groubpy Stop ID for each stop id apply the process size function for all sizes 
    Then we choose the stop ids we care about 
    """
    
    
    
    
    

    GsdBySite = data.groupby('Stop ID').agg({'Size (cm)':process_size}).loc[exposures]
    
    
    i=0
    
    for site,size in GsdBySite.iterrows():  #for each site, array of all sizes in that site, itterate through array - each size
        #print(site,size)
        
        count = np.linspace(0,100,len(size[0])) # return evenly spaced numbers between 0 and 100 for the count of variables in each Size array
        plt.semilogx(size[0],count,
                    label = str(site),color=colors[i],linestyle=line_types[i])
        
        i+=1
        
        

        
    plt.xlabel('Grain Size (m)',fontsize = 10)
    plt.ylabel('Cumulative % Finer',fontsize = 10)
    plt.ylim([0,100])
    plt.legend()
    
    plt.title(title)
    plt.show()
    
    
def add_combine_exposures(data,exposures,deposit_name):
    """
    Add combined exposures as a site to table
    """
    deposit = data.loc[data['Stop ID'].isin(exposures)]
    deposit['Stop ID'] = deposit_name
    data= pd.concat([data,deposit],axis=0)
    
    return data 
   
    
   
    
#Set all extra shrs measurements to nan

gsd_data.drop(gsd_data[gsd_data['Extra SHRS'] == 'y'].index, inplace = True)

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
#%%

gsd_data= add_combine_exposures(gsd_data,exposures=['10.0','7.5','7.0','9.0','11.0'],
                                deposit_name="Suiattle")
#%%
# Plot gsd data

run_grain_size_plot(gsd_data,exposures=['KC1','KC2','KC3','KC4','KC5','KC6','KC7',"Kautz"],
                    colors =['blue','purple','red','orange','green','gray','black','maroon'],
                    line_types=['solid','solid','solid','solid','solid','solid','solid','dashed']
                    ,title='Kautz Creek GSD')

run_grain_size_plot(gsd_data,exposures=['T1', 'T2', 'T3', "Tahoma 2022"],
                    colors =['blue','purple','red','orange'],
                    line_types=['solid','solid','solid','dashed']
                    ,title='Tahoma 2022 GSD')

run_grain_size_plot(gsd_data,exposures=['T4','TNUV1','T5A','T5B','T6','T7','T8','Tahoma 2023'],
                    colors =['blue','purple','red','orange', 'green', 'gray', 'black', 'maroon'],
                    line_types=['solid','solid','solid','solid','solid','solid','solid','dashed']
                    ,title='Tahoma 2023 GSD')

run_grain_size_plot(gsd_data,exposures=['MM1','MM2','MM3','MM4','MM5','MM6','MM7','MM8','MM9', 'MM10', 'MM11','Mt. Meager'],
                    colors =['blue','purple','red','orange', 'green', 'gray', 'black', 'maroon', 'pink','olive','cyan','greenyellow'],
                    line_types=['solid','solid','solid','solid','solid','solid','solid','solid','solid','solid','solid','dashed']
                    ,title='Mt. Meager GSD')

run_grain_size_plot(gsd_data,exposures=['MA1','MA1B','MA2','MA2B','MA3','MA3B','MA4','MA4A','Mt. Adams'],
                    colors =['blue','purple','red','orange', 'green', 'gray', 'black', 'maroon', 'pink'],
                    line_types=['solid','solid','solid','solid','solid','solid','solid','solid','dashed']
                    ,title='Mt. Adams GSD')

run_grain_size_plot(gsd_data,exposures=[ 'Tahoma', 'Kautz', 'Mt. Meager', 'Mt. Adams','Suiattle'],
                    colors =['blue','purple','red','orange', 'green', 'gray', 'black'],
                    line_types=['solid','solid','solid','solid','solid']
                    ,title='All Deposits GSD')

#print(gsd_data['Stop ID'].unique())



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

print('DONE!!!!')


print('Kautz D50 is ', np.median(process_size(gsd_data.loc[gsd_data['Stop ID']=="Kautz"]['Size (cm)'])), 'm')




    

