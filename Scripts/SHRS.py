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
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn

import statsmodels.api as sm 
from statsmodels.formula.api import ols 

#os.chdir('C:\\Users\\pinkeb\\OneDrive - Western Washington University\\Thesis\\ThesisCode')
os.chdir('/Users/bpinke/Library/CloudStorage/OneDrive-WesternWashingtonUniversity/Thesis/ThesisCode')

# load in 2023 field data
path=os.path.join( os.getcwd(), 'Data/fieldbook_data_2023_USE.xlsx')

field_data2023 = pd.read_excel (
        path, 
        sheet_name='1 - Measurements', 
        header = 0)


# load in Suiattle Debris Flow deposit SHRS data
path=os.path.join( os.getcwd(), 'Data/SuiattleFieldData_Combined20182019.xlsx')

suiattleSHRS_data = pd.read_excel (
        path, 
        sheet_name='4-SHRSdfDeposit', 
        header = 0)

# load in Suiattle boulder bar SHRS data
suiattleBoulder_data = pd.read_excel (
        path,
        sheet_name= '2 - SHRS boulders on bars',
        header = 0)
suiattleSHRS_data =pd.concat([suiattleSHRS_data, suiattleBoulder_data], axis=0)

# load in Global SHRS data from Gouidie 2006
path=os.path.join(os.getcwd(), 'Data/Goudie2006RockStrength.xlsx')

GoudieSHRS = pd.read_excel(
        path,
        sheet_name= 'Table 1',
        header=1)


# define colors
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

def run_shrs_by_plot(df, by, data, colors, title=None, xaxislabel=None):
    """
    df: df of choice
    by: string of column name you want to group the data by for plotting ("Stop ID" or "Lithology Category")
    data: a list of strings of the categories you want plotted. Eg the Stop ID names or the Lithology Categories of interest. 
    title: default is none, give a string if desired.
    xaxislabel: default is none. If "by" variable is "Stop ID" or "Lithology Category", the xlabel will be "Deposit" or "Lithology" respectively. If no xaxis label is desired, set = "No".

    Creates violin plot of desired data, with n printed on fig and min, median, max printed in console.
    """
   # set the xlabel given these conditions
    if by == "Stop ID":
        xlabel = "Deposit"
    if by == 'Lithology Category' or 'Lithology':
        xlabel = "Lithology"
    if xaxislabel == "No":
        xlabel = None
    
    # new temporary df grouping the given df as desired and selecting for desired data.
    datas = df.loc[df[by].isin(data)]
    
    # creates violin plot
    ax = sns.violinplot(data=datas, x=by, y='Mean_Median SHRS', saturation=1, palette=colors, order=data)
    
    # Update x tick labels to include count for each category
    new_xticklabels = []
    for category in data:
        count = datas[datas[by] == category].shape[0]
        new_xticklabels.append(f'{category}\n(n={count})')
    
    ax.set_xticklabels(new_xticklabels)
    
    ax.set(ylabel='SHRS', xlabel=xlabel)
    plt.ylim(0, 90)
    plt.title(title)#######
    plt.xticks(rotation=45, ha="right", fontsize=8)
    
    # get the median, min, and max values for each category represented
    for category in data:
        category_data = datas[datas[by] == category]['Mean_Median SHRS']
        median_val = category_data.median()
        min_val = category_data.min()
        max_val = category_data.max()

        # Print out min, max, and median values for each category
        print(f"{category} - Min: {min_val:.2f}, Max: {max_val:.2f}, Median: {median_val:.2f}")
        
    plt.savefig("SHRS" + title + ".png", dpi=400, bbox_inches="tight")
    plt.show()

#%%
# Prep DFs for use

#prep suiattle liths
merge_liths = { 'GF': 'PL', 'VC':'VV'} #'G': 'PL' # Define replacements {old_lith: new_lith}

for old_lith, new_lith in merge_liths.items():
    suiattleSHRS_data.loc[suiattleSHRS_data['Lithology'] == old_lith, 'Lithology'] = new_lith
    
suiattleSHRS_data = suiattleSHRS_data[suiattleSHRS_data['Lithology'] != 'MS']

# simplify df's to desired columns
pinke_data = field_data2023[['Stop ID','Lithology Category','Mean_Median SHRS']].copy()
suiattleSHRS_data = suiattleSHRS_data[['Site Num', 'Lithology','SHRS median']].copy()

#change suiattle column names for merging with 2023 field data
suiattleSHRS_data=suiattleSHRS_data.rename(columns={"Site Num": "Stop ID", 'Lithology': 'Lithology Category','SHRS median': 'Mean_Median SHRS'})
suiattleSHRS_data['Stop ID'] = suiattleSHRS_data['Stop ID'].astype("string")

# merge 2023 and suiattle data to 1 df and drop any empty data
SHRS_data= pd.concat([pinke_data,suiattleSHRS_data],axis=0)
SHRS_data= SHRS_data[SHRS_data['Mean_Median SHRS'].notna()]

# change values of NAL to NA - had to be entered into excel as NAL as NA would appear as empty to python
SHRS_data.loc[SHRS_data['Lithology Category'] == 'NAL', 'Lithology Category'] = 'NA'

#%%
# combine exposures as needed for plotting
         
SHRS_data= add_combine_exposures(SHRS_data,exposures=['KC1','KC2','KC3','KC4','KC5','KC6','KC7'],
                                deposit_name="Kautz") #SHOULD I INCLUDE SHRS1 and 2???

SHRS_data= add_combine_exposures(SHRS_data,exposures=['T2A','T2B','T2C'],
                                deposit_name="T2")

SHRS_data= add_combine_exposures(SHRS_data,exposures=['T1','T2','T3'],
                                deposit_name="Tahoma 2022")

SHRS_data= add_combine_exposures(SHRS_data,exposures=['T4','TNUV1','T5A','T5B','T6','T7','T8'],
                                deposit_name="Tahoma 2023")

SHRS_data= add_combine_exposures(SHRS_data,exposures=['T1','T2','T4','TNUV1','T5A','T5B','T6','T7','T8'],
                                deposit_name="Tahoma")

SHRS_data= add_combine_exposures(SHRS_data,exposures=['MM1','MM2','MM3','MM4','MM5','MM6','MM7','MM8','MM9', 'MM10', 'MM11'],
                                deposit_name="Mt. Meager")

SHRS_data= add_combine_exposures(SHRS_data,exposures=['MA1','MA1B','MA2','MA2B','MA3','MA3B','MA4','MA4A'],
                                deposit_name="Mt. Adams")

SHRS_data= add_combine_exposures(SHRS_data,exposures=['9','10', '1','2', '3', '4', '5','6','7', '8','11','12','13','14','15','16','17' ],
                                deposit_name="Suiattle")

#%%
# plot SHRS by exposure

run_shrs_by_plot(SHRS_data,by="Stop ID", data=['KC1','KC2','KC3','KC4','KC5','KC6','KC7',"Kautz"],
                    colors =['blue','purple','red','orange','green','gray','black','maroon'],
                    title='Kautz Creek SHRS')

run_shrs_by_plot(SHRS_data,by="Stop ID", data=['T1', 'T2', 'T3', "Tahoma 2022"],
                    colors =['blue','purple','red','orange'],
                    title='Tahoma 2022 SHRS')

run_shrs_by_plot(SHRS_data,by="Stop ID", data=['T4','TNUV1','T5A','T5B','T6','T7','T8','Tahoma 2023'],
                    colors =['blue','purple','red','orange', 'green', 'gray', 'black', 'maroon'],
                    title='Tahoma 2023 SHRS')

run_shrs_by_plot(SHRS_data,by="Stop ID", data=['MM1','MM2','MM3','MM4','MM5','MM6','MM7','MM8','MM9', 'MM10', 'MM11','Mt. Meager'],
                    colors =['blue','purple','red','orange', 'green', 'gray', 'black', 'maroon', 'pink','olive','cyan','greenyellow'],
                    title='Mt. Meager SHRS')

run_shrs_by_plot(SHRS_data,by="Stop ID", data=['MA1','MA1B','MA2','MA2B','MA3','MA3B','MA4','MA4A','Mt. Adams'],
                    colors =['blue','purple','red','orange', 'green', 'gray', 'black', 'maroon', 'pink'],
                    title='Mt. Adams SHRS')

run_shrs_by_plot(SHRS_data,by="Stop ID", data=['9', '10', 'Suiattle'],
                    colors =['blue','purple','red','orange', 'green', 'gray'],
                    title='Suiattle SHRS')

run_shrs_by_plot(SHRS_data,by="Stop ID", data=['Tahoma', 'Kautz', 'Mt. Meager', 'Mt. Adams', 'Suiattle'],
                    colors =['blue','purple','red','orange', 'green', 'gray'],
                    title='All Deposits SHRS')

#%%
# plot SHRS by Lithology per Deposit
run_shrs_by_plot(SHRS_data.loc[SHRS_data['Stop ID']=='Kautz'],by="Lithology Category", data=['V', 'NV','G'],
                    colors =[lightpurple, darkpurple, pink],
                    title='Kautz Creek SHRS by Lithology')

run_shrs_by_plot(SHRS_data.loc[SHRS_data['Stop ID']=='Tahoma 2023'], by="Lithology Category", data=[ 'VV','TV','UG', 'PN'],
                    colors =[lightpurple, lightpurple, orange, darkpurple],
                    title='Tahoma SHRS by Lithology')

run_shrs_by_plot(SHRS_data.loc[SHRS_data['Stop ID']=='Mt. Meager'],by="Lithology Category", data=['LDV','OW','HDV','G','HGM'],
                    colors =[lightpurple, orange, darkpurple, pink, green],
                    title='Mt. Meager SHRS by Lithology')

run_shrs_by_plot(SHRS_data.loc[SHRS_data['Stop ID']=='Mt. Adams'],by="Lithology Category", data=['VA','VN','NA','NN'],
                    colors =[lighterorange, lightpurple, orange, darkpurple],
                    title='Mt. Adams SHRS by Lithology')

run_shrs_by_plot(SHRS_data.loc[SHRS_data['Stop ID']=='Suiattle'],by="Lithology Category", data=['VV','NV','PL'],
                    colors =[lightpurple, darkpurple, pink],
                    title='Suiattle SHRS by Lithology')

#run_shrs_by_plot(SHRS_data,by='Lithology Category',data=['KC1','KC2','KC3','KC4','KC5','KC6','KC7'])


#SHRS_data['Stop ID']["Kautz"].groupby("lithology Category").unique_values()

print('DONE!!!')
#print(SHRS_data.loc[SHRS_data['Stop ID']=="Kautz"]["Lithology Category"].unique())

#%%




##### EVERYTHING BELOW IS VERY EXPERIMENTAL / IN PROGRESS###





#%%
#look at trends of just VV across all deposits

vesicularLiths = ('V', 'VV', 'LDV', 'VN')

vv_data = SHRS_data[SHRS_data['Lithology Category'].isin(vesicularLiths)].copy()

run_shrs_by_plot(vv_data,by="Stop ID", data=['Tahoma 2023', 'Kautz', 'Mt. Meager', 'Mt. Adams', 'Suiattle'],
                    colors =[tahomacolor,kautzcolor,meagercolor,adamscolor, glaciercolor],
                    title='VV by Deposit')

# use scipi stats and kruskal to do non parametric anova -> Kruskal-Wallis test




# Perform Kruskal-Wallis test
#statistic, p_value = kruskal(vv_data.groupby('Stop ID')['Tahoma 2023'], group2, group3)


#%%

# Group by 'Stop ID' and create a dictionary of lists for 'Mean_Median SHRS' values
deposit_shrs_dict = dict(tuple(vv_data.groupby('Stop ID')['Mean_Median SHRS'].apply(list).reset_index().values))



#%%
### tahoma kautz and meager are similar. if compare those 3 we fail to reject null hypothesis. if include adams or suiattle, it becomes significantly different.
### Perform Kruskal-Wallis test
statistic, pvalue = kruskal(deposit_shrs_dict['Tahoma 2023'], deposit_shrs_dict['Kautz'], deposit_shrs_dict['Mt. Meager'], deposit_shrs_dict['Mt. Adams'],deposit_shrs_dict['Suiattle'] )
# Print the result
print("Kruskal-Wallis H statistic:", statistic)
print("P-value:", pvalue)

# Check the p-value to determine if there are significant differences
if pvalue < 0.05:
    print("Reject the null hypothesis. There are significant differences between groups.")
else:
    print("Fail to reject the null hypothesis. There are no significant differences between groups.")
    
# Perform Dunn's test for pairwise comparisons
posthoc_results = posthoc_dunn(
    [deposit_shrs_dict[group] for group in deposit_shrs_dict],
    p_adjust='bonferroni'
)

# Display the pairwise comparison results
print(posthoc_results)
#%%
# Group by 'Stop ID' and calculate the mean for each group
mean_mean_median_shrs = vv_data.groupby('Stop ID')['Mean_Median SHRS'].mean()

# Print or use the result as needed
print(mean_mean_median_shrs)

#%%
#run kruskal wallis for just the means

# Perform Kruskal-Wallis test
statistic, pvalue = kruskal(44.3, 33.62, 40.756881, 51.654691, )
# Print the result
print("Kruskal-Wallis H statistic:", statistic)
print("P-value:", pvalue)

# Check the p-value to determine if there are significant differences
if pvalue < 0.05:
    print("Reject the null hypothesis. There are significant differences between groups.")
else:
    print("Fail to reject the null hypothesis. There are no significant differences between groups.")


#%%
# Lithology fractions for each deposit, obtained from PieLith script analysis
# create dfs with the lithology and corresponding fraction for each deposit

# Data for Meager
meager_df = pd.DataFrame({
    'Lithologies': ['LDV', 'G', 'HDV', 'OW', 'HGM'],
    'Fraction': [62.064677, 29.975124, 4.726368, 2.487562, 0.746269]})

meager_df['Fraction'] = (meager_df['Fraction'] * 10).round(0)

# Data for Suiattle
suiattle_df = pd.DataFrame ({
    'Lithologies': ['VV', 'NV', 'PL'],
    'Fraction': [56.043956, 42.124542, 1.831502]})

suiattle_df['Fraction'] = (suiattle_df['Fraction'] * 10).round(0)

# Data for Tahoma
tahoma_df = pd.DataFrame ({
    'Lithologies': ['PN', 'TV', 'VV', 'UG'],
    'Fraction': [37.500000, 34.375000, 18.923611, 9.201389]})

tahoma_df['Fraction'] = (tahoma_df['Fraction'] * 10).round(0)

# Data for Kautz
kautz_df = pd.DataFrame({
    'Lithologies': ['NV', 'V', 'G'],
    'Fraction': [89.189189, 9.009009, 1.801802]})

kautz_df['Fraction'] = (kautz_df['Fraction'] * 10).round(0)
kautz_df.at[2, 'Fraction'] += 1

# Data for Adams
adams_df = pd.DataFrame({
    'Lithologies': ['NA', 'NN', 'VN', 'VA'],
    'Fraction': [49.340369, 31.662269, 9.498681, 9.498681]})

adams_df['Fraction'] = (adams_df['Fraction'] * 10).round(0)

#%%
# create a new df with lithology in the first column, strength in the second column. 
#the lithology should occur as many times as denoted by "Fraction". Strength column should be empty.
#new_df = pd.DataFrame({'Lithology': adams_df['Lithologies'].repeat(adams_df['Fraction'].astype(int)),'Strength': ''})

#%%
# =============================================================================
# testadamsdf = SHRS_data.loc[(SHRS_data['Stop ID'] == 'Mt. Adams') & (SHRS_data['Lithology Category'] == 'NA')]
# =============================================================================

#%%
# Calculate the empirical cumulative distribution function (ECDF)
def ecdf(data):
    # Sort the data in ascending order
    x = np.sort(data)
    # Compute the cumulative probabilities for each data point
    y = np.arange(1, len(data) + 1) / len(data)
    return x, y
# =============================================================================
# #%%
# # Calculate ECDF for the original data
# x_original, y_original = ecdf(testadamsdf['Mean_Median SHRS'])
# 
# # Generate 100 random samples from a uniform distribution between 0 and 1
# random_uniform_samples = np.random.rand(421) #THIS IS A NUMBER THAT SHOULD BE THE FRAC FOR THAT LITHOLOGY OUT OF 1000
# 
# # Map the random samples to strength values based on the original ECDF
# mapped_strengths = np.interp(random_uniform_samples, y_original, x_original)
# 
# # Plot ECDFs for comparison
# plt.step(x_original, y_original, label='Original Data')
# plt.step(np.sort(mapped_strengths), np.arange(1, 422) / 421, label='Mapped Data')
# plt.xlabel('Strength Values')
# plt.ylabel('Cumulative Probability')
# plt.legend()
# plt.show()
# =============================================================================

#%%#%%

def create1000SHRSsample(deposit_name, lith_percent_df):
    # Creating the new DataFrame with a dynamic name
    new_df = pd.DataFrame(columns=['Mean_Median SHRS','Lithology Category', 'Stop ID'])
    
    for lithtype in lith_percent_df['Lithologies']:
        # Calculate ECDF for the original data
        x_original, y_original = ecdf(SHRS_data.loc[(SHRS_data['Stop ID'] == deposit_name) & (SHRS_data['Lithology Category'] == lithtype)]['Mean_Median SHRS'])
        print(deposit_name, lithtype, "x:", x_original,"y", y_original)
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

#%%

Meager_weightedSHRS_distrib = create1000SHRSsample('Mt. Meager', meager_df, )
Suiattle_weightedSHRS_distrib = create1000SHRSsample('Suiattle', suiattle_df, )
Tahoma_weightedSHRS_distrib = create1000SHRSsample('Tahoma 2023', tahoma_df, )
Kautz_weightedSHRS_distrib = create1000SHRSsample('Kautz', kautz_df, )
Adams_weightedSHRS_distrib = create1000SHRSsample('Mt. Adams', adams_df, )

weightedSHRS_distrib= pd.concat([Meager_weightedSHRS_distrib, Suiattle_weightedSHRS_distrib, Tahoma_weightedSHRS_distrib, Kautz_weightedSHRS_distrib, Adams_weightedSHRS_distrib], ignore_index=True)

#%%

run_shrs_by_plot(weightedSHRS_distrib ,by="Stop ID", data=['Tahoma 2023', 'Kautz', 'Mt. Meager', 'Mt. Adams', 'Suiattle'],
                    colors =[tahomacolor,kautzcolor,meagercolor,adamscolor, glaciercolor],
                    title=' Lith Weighted Distributions')

run_shrs_by_plot(SHRS_data,by="Stop ID", data=['Tahoma 2023', 'Kautz', 'Mt. Meager', 'Mt. Adams', 'Suiattle'],
                    colors =[tahomacolor,kautzcolor,meagercolor,adamscolor, glaciercolor],
                    title='All Deposits SHRS')


#weightedSHRS_distrib.replace({'Stop ID': {'Kautz': 'Kautz Creek'}}, inplace=True)
#weightedSHRS_distrib.replace({'Stop ID': {'Tahoma 2023': 'Little Tahoma'}}, inplace=True)

#run_shrs_by_plot(weightedSHRS_distrib ,by="Stop ID", data=['Kautz Creek','Little Tahoma'],
                    #colors =[kautzcolor, tahomacolor],
                    #title='Tahoma vs Kautz')


#%%

vvWeighted_data = weightedSHRS_distrib[weightedSHRS_distrib['Lithology Category'].isin(vesicularLiths)].copy()

run_shrs_by_plot(vvWeighted_data,by="Stop ID", data=['Tahoma 2023', 'Kautz', 'Mt. Meager', 'Mt. Adams', 'Suiattle'],
                    colors =[tahomacolor,kautzcolor,meagercolor,adamscolor, glaciercolor],
                    title='Weighted VV by Deposit')



#%%

# Group by 'Stop ID' and create a dictionary of lists for 'Mean_Median SHRS' values
Weighteddeposit_shrs_dict = dict(tuple(vvWeighted_data.groupby('Stop ID')['Mean_Median SHRS'].apply(list).reset_index().values))



#%%
### tahoma kautz and meager are similar. if compare those 3 we fail to reject null hypothesis. if include adams or suiattle, it becomes significantly different.
### Perform Kruskal-Wallis test
statistic, pvalue = kruskal(Weighteddeposit_shrs_dict['Tahoma 2023'], Weighteddeposit_shrs_dict['Kautz'], Weighteddeposit_shrs_dict['Mt. Meager'], Weighteddeposit_shrs_dict['Mt. Adams'],Weighteddeposit_shrs_dict['Suiattle'] )
# Print the result
print("Kruskal-Wallis H statistic:", statistic)
print("P-value:", pvalue)

# Check the p-value to determine if there are significant differences
if pvalue < 0.05:
    print("Reject the null hypothesis. There are significant differences between groups.")
else:
    print("Fail to reject the null hypothesis. There are no significant differences between groups.")
    


#%%
run_shrs_by_plot(weightedSHRS_distrib.loc[weightedSHRS_distrib['Stop ID']=='Tahoma 2023'], by="Lithology Category", data=[ 'VV','TV','UG', 'PN'],
                    colors =[lightpurple, lightpurple, orange, darkpurple],
                    title='Tahoma Weighted SHRS by Lithology')

run_shrs_by_plot(SHRS_data.loc[SHRS_data['Stop ID']=='Tahoma 2023'], by="Lithology Category", data=[ 'VV','TV','UG', 'PN'],
                    colors =[lightpurple, lightpurple, orange, darkpurple],
                    title='Tahoma SHRS by Lithology')
#%%

#%%

# take the 5 deposits and place in new df with just lithology and shrs. 
#make a new column for "broad lith" and group into igneous and metamorphic. 
#I can later grab all granites separate using the specific liths.
# I need to change names to match GoudieSHRS
# Run it through plot function 
selectexposures = ['Tahoma 2023', 'Kautz', 'Mt. Meager', 'Mt. Adams', 'Suiattle']


SHRSbyLith = SHRS_data.loc[SHRS_data['Stop ID'].isin(selectexposures)]

print(SHRSbyLith['Lithology Category'].unique())

volcanic_rocks = ['NV', 'V', 'A', 'PN', 'AN', 'VV', 'TV', 'OW', 'LDV', 'HDV', 'NN', 'NA', 'VN', 'VA']
plutonic_rocks = ['G', 'UG', 'PL']

SHRSbyLith['Lithology'] = np.nan

SHRSbyLith.loc[SHRSbyLith['Lithology Category'].isin(plutonic_rocks), 'Lithology'] = 'PNW Plutonic'
SHRSbyLith.loc[SHRSbyLith['Lithology Category'].isin(volcanic_rocks), 'Lithology'] = 'PNW Volcanic'
SHRSbyLith.loc[SHRSbyLith['Lithology Category'] == 'HGM', 'Lithology'] = 'PNW Metamorphic'
#%%
#Modify SHRSbyLith for combining with Goudie SHRS

# Drop and grab columns
SHRSbyLith = SHRSbyLith.drop(['Stop ID', 'Lithology Category'], axis=1)
GoudiePNWSHRS = GoudieSHRS[['Mean R Value', 'Lithology']]

# Rename columns
#SHRSbyLith = SHRSbyLith.rename(columns={'Mean_Median SHRS': 'SHRS'})
GoudiePNWSHRS = GoudiePNWSHRS.rename(columns = {'Mean R Value': 'Mean_Median SHRS'})


GoudiePNWSHRS = pd.concat([GoudiePNWSHRS, SHRSbyLith], ignore_index=True)
#%%
GoudiePNWSHRS = GoudiePNWSHRS.dropna()
#%%
# Plot!

run_shrs_by_plot(GoudiePNWSHRS, by="Lithology", data=GoudiePNWSHRS['Lithology'].unique().tolist(),
                    colors =[lightpurple, lightpurple, orange, darkpurple, 'green', 'red'],
                    title='GoudiePNW SHRS')


#%%

# Group by 'Stop ID' and create a dictionary of lists for 'Mean_Median SHRS' values
GoudieShrsByLith_dict = dict(tuple(GoudiePNWSHRS.groupby('Lithology')['Mean_Median SHRS'].apply(list).reset_index().values))



#%%
### tahoma kautz and meager are similar. if compare those 3 we fail to reject null hypothesis. if include adams or suiattle, it becomes significantly different.
### Perform Kruskal-Wallis test
statistic, pvalue = kruskal(GoudieShrsByLith_dict['Volcanic'], GoudieShrsByLith_dict['PNW Volcanic'] )
# Print the result
print("Kruskal-Wallis H statistic:", statistic)
print("P-value:", pvalue)

# Check the p-value to determine if there are significant differences
if pvalue < 0.05:
    print("Reject the null hypothesis. There are significant differences between groups.")
else:
    print("Fail to reject the null hypothesis. There are no significant differences between groups.")
    



#%%

# =============================================================================
# #%%
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# 
# # Function to calculate the empirical cumulative distribution function (ECDF)
# def ecdf(data):
#     x = np.sort(data)
#     y = np.arange(1, len(data) + 1) / len(data)
#     return x, y
# 
# # Original data and random samples
# original_data = testadamsdf['Mean_Median SHRS']
# num_samples = 421
# random_uniform_samples = np.random.rand(num_samples)
# 
# # Calculate ECDF for the original data
# x_original, y_original = ecdf(original_data)
# 
# # Map random samples to strength values based on the original ECDF
# mapped_strengths = np.interp(random_uniform_samples, y_original, x_original)
# 
# # Combine the original and mapped data into a single DataFrame
# data = pd.DataFrame({
#     'Data Type': np.concatenate([['Original Data'] * len(original_data), ['Mapped Data'] * num_samples]),
#     'Strength Values': np.concatenate([original_data, mapped_strengths])
# })
# 
# # Create a violin plot using seaborn
# plt.figure(figsize=(10, 6))
# sns.violinplot(x='Data Type', y='Strength Values', data=data)
# plt.xlabel('Data Type')
# plt.ylabel('Strength Values')
# plt.title('Comparison Violin Plot')
# plt.show()
# 
# =============================================================================









#%%
#x = SHRS_data.loc[(SHRS_data['Stop ID'] == 'Kautz') & (SHRS_data['Lithology Category'] == 'NV')]['Mean_Median SHRS']


#%%
# work to plot for each deposit, the extra shrs for each lith and the in situ shrs for each lith
insitu_extra_shrs = field_data2023[['Stop ID','Lithology Category','Mean_Median SHRS', 'Extra SHRS']].copy()

#%%
# combine exposures as needed
         
insitu_extra_shrs= add_combine_exposures(insitu_extra_shrs,exposures=['KC1','KC2','KC3','KC4','KC5','KC6','KC7', 'KCSH1', 'KCSH2'],
                                deposit_name="Kautz") #SHOULD I INCLUDE SHRS1 and 2???

insitu_extra_shrs= add_combine_exposures(insitu_extra_shrs,exposures=['T2A','T2B','T2C'],
                                deposit_name="T2")

insitu_extra_shrs= add_combine_exposures(insitu_extra_shrs,exposures=['T1','T2','T3'],
                                deposit_name="Tahoma 2022")

insitu_extra_shrs= add_combine_exposures(insitu_extra_shrs,exposures=['T4','TNUV1','T5A','T5B','T6','T7','T8'],
                                deposit_name="Tahoma 2023")

insitu_extra_shrs= add_combine_exposures(insitu_extra_shrs,exposures=['T1','T2','T4','TNUV1','T5A','T5B','T6','T7','T8'],
                                deposit_name="Tahoma")

insitu_extra_shrs= add_combine_exposures(insitu_extra_shrs,exposures=['MM1','MM2','MM3','MM4','MM5','MM6','MM7','MM8','MM9', 'MM10', 'MM11'],
                                deposit_name="Mt. Meager")

insitu_extra_shrs= add_combine_exposures(insitu_extra_shrs,exposures=['MA1','MA1B','MA2','MA2B','MA3','MA3B','MA4','MA4A'],
                                deposit_name="Mt. Adams")


# change values of NAL to NA
insitu_extra_shrs.loc[insitu_extra_shrs['Lithology Category'] == 'NAL', 'Lithology Category'] = 'NA'

#%%

run_shrs_by_plot(insitu_extra_shrs.loc[(insitu_extra_shrs['Stop ID'] == 'Kautz') & (insitu_extra_shrs['Extra SHRS'] == 'y')],by="Lithology Category", data=['V', 'NV','G'],
                    colors =[lightpurple, darkpurple, pink],
                    title='Kautz Creek extra SHRS by Lithology')
run_shrs_by_plot(insitu_extra_shrs.loc[(insitu_extra_shrs['Stop ID'] == 'Kautz') & (insitu_extra_shrs['Extra SHRS'].isna())& (insitu_extra_shrs['Mean_Median SHRS'].notna())],by="Lithology Category", data=['V', 'NV','G'],
                    colors =[lightpurple, darkpurple, pink],
                    title='Kautz Creek insitu SHRS by Lithology')



run_shrs_by_plot(insitu_extra_shrs.loc[(insitu_extra_shrs['Stop ID'] == 'Tahoma 2023') & (insitu_extra_shrs['Extra SHRS'] == 'y')], by="Lithology Category", data=[ 'VV','TV','UG', 'PN'],
                    colors =[lightpurple, lightpurple, orange, darkpurple],
                    title='Tahoma extra SHRS by Lithology')
run_shrs_by_plot(insitu_extra_shrs.loc[(insitu_extra_shrs['Stop ID'] == 'Tahoma 2023') & (insitu_extra_shrs['Extra SHRS'].isna())& (insitu_extra_shrs['Mean_Median SHRS'].notna())], by="Lithology Category", data=[ 'VV','TV','UG', 'PN'],
                    colors =[lightpurple, lightpurple, orange, darkpurple],
                    title='Tahoma insitu SHRS by Lithology')


run_shrs_by_plot(insitu_extra_shrs.loc[(insitu_extra_shrs['Stop ID'] == 'Mt. Meager') & (insitu_extra_shrs['Extra SHRS'] == 'y')],by="Lithology Category", data=['LDV','OW','HDV','G','HGM'],
                    colors =[lightpurple, orange, darkpurple, pink, green],
                    title='Mt. Meager extra SHRS by Lithology')
run_shrs_by_plot(insitu_extra_shrs.loc[(insitu_extra_shrs['Stop ID'] == 'Mt. Meager') & (insitu_extra_shrs['Extra SHRS'].isna())& (insitu_extra_shrs['Mean_Median SHRS'].notna())],by="Lithology Category", data=['LDV','OW','HDV','G','HGM'],
                    colors =[lightpurple, orange, darkpurple, pink, green],
                    title='Mt. Meager insitu SHRS by Lithology')


run_shrs_by_plot(insitu_extra_shrs.loc[(insitu_extra_shrs['Stop ID'] == 'Mt. Adams') & (insitu_extra_shrs['Extra SHRS'] == 'y')],by="Lithology Category", data=['VA','VN','NA','NN'],
                    colors =[lighterorange, lightpurple, orange, darkpurple],
                    title='Mt. Adams extra SHRS by Lithology')
run_shrs_by_plot(insitu_extra_shrs.loc[(insitu_extra_shrs['Stop ID'] == 'Mt. Adams') & (insitu_extra_shrs['Extra SHRS'].isna())& (insitu_extra_shrs['Mean_Median SHRS'].notna())],by="Lithology Category", data=['VA','VN','NA','NN'],
                    colors =[lighterorange, lightpurple, orange, darkpurple],
                    title='Mt. Adams insitu SHRS by Lithology')

#%%

# Rename columns so only one word
weightedSHRS_distrib.rename(columns={'Mean_Median SHRS': 'SHRS'}, inplace=True)
weightedSHRS_distrib.rename(columns={'Stop ID': 'Exposure'}, inplace=True)
weightedSHRS_distrib.rename(columns={'Lithology Category': 'Lithology'}, inplace=True)
#%%
# Performing two-way ANOVA 
model = ols('SHRS ~ Exposure + Lithology + Exposure:Lithology', data=weightedSHRS_distrib).fit() 
anova_table = sm.stats.anova_lm(model, typ=2) 

#%%
#make a weighted shrs distrib df but where all the lith categories are grouped into the "like liths"
ldv_rocks = ['V', 'VV', 'TV', 'OW', 'LDV', 'HDV', 'NN', 'NA', 'VN', 'VA']
hdv_rocks = ['A', 'PN', 'AN', 'HDV', 'NN', 'NV']
vesicularLiths = ('V', 'VV', 'LDV', 'VN')
altered_rocks = ['OW', '']
#metamorphic_rocks = 
plutonic_rocks = ['G', 'UG', 'PL']

SHRSbyLith['Lithology'] = np.nan

#SHRSbyLith.loc[SHRSbyLith['Lithology Category'].isin(plutonic_rocks), 'Lithology'] = 'PNW Plutonic'
#SHRSbyLith.loc[SHRSbyLith['Lithology Category'].isin(volcanic_rocks), 'Lithology'] = 'PNW Volcanic'
#SHRSbyLith.loc[SHRSbyLith['Lithology Category'] == 'HGM', 'Lithology'] = 'PNW Metamorphic'
