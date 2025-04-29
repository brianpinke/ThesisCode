#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 1 17:42:48 2024

@author: bpinke

# Plot Figure 5A: SHRS of each deposit as measured.
# Plot Figure 5B: SHRS of each deposit weighted by lithology fraction, n for each is 1000.
# Plot Figure 6: SHRS of each deposit, broken up by exposure. 
# Plot Figure 7: SHRS of each deposit broken up by lithology type.
# Plot Figure 15: SHRS of different rock types globally (modified from Goudie (2006), each data point is the mean of many measurements on many samples). Data from this study denoted with *. 

"""

# loads necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns 
from scipy.stats import kruskal

# load in 2023 field data
path=os.path.join( os.getcwd(), 'Data/fieldbook_data_2023_USE.xlsx')

pinke2023_data = pd.read_excel (
        path, 
        sheet_name='1 - Measurements', 
        header = 0)


# load in Suiattle Debris Flow deposit SHRS data
path=os.path.join( os.getcwd(), 'Data/SuiattleFieldData_Combined20182019.xlsx')

suiattleSHRS_data = pd.read_excel (
        path, 
        sheet_name='4-SHRSdfDeposit', 
        header = 0)

# load in Global SHRS data from Gouidie 2006
path=os.path.join(os.getcwd(), 'Data/Goudie2006RockStrength.xlsx')

Goudie_data = pd.read_excel(
        path,
        sheet_name= 'Table 1',
        header=1)


# define colors
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
tahomacolor = (0.4, 0.6, 0.8) #steel blue
kautzcolor = (0.8627450980392157, 0.0784313725490196, 0.23529411764705882) #crimson
adamscolor = (0.5019607843137255, 0.5019607843137255, 0.0) #olive

# colors for Goudie
light_gray = (0.827, 0.827, 0.827)
gray = (0.663, 0.663, 0.663)
dim_gray = (0.412, 0.412, 0.412)
dark_gray = (0.314, 0.314, 0.314)
very_dark_gray = (0.161, 0.161, 0.161)

# Change text font and size
plt.rcParams["font.family"] = "Arial"
plt.rcParams['font.size'] = 10

#set random seed to 5
np.random.seed(5) 

# change the directory so figs are saved in proper location
os.chdir(os.getcwd()+'/Figs')
#%%
# Denote what each deposit is called

MMLR = "Lillooet R."
GPSR = "Suiattle R."
MRWR = "White R."
MRKC = "Kautz C."
MASC = "Salt C."

deposit_list = [MMLR, GPSR, MRWR, MRKC, MASC]
#%%   
# define functions

def add_combine_exposures(data,exposures,deposit_name, by='Stop ID'):
    """
    data: df of interest
    exposures: strings of exposures to be combined
    deposit_name: string of name of combined exposures
    Add combined exposures as a deposit to df
    
    combines exposures in a given dataset given a new "deposit_name", appended to existing dataframe.
    
    """
    deposit = data.loc[data['Stop ID'].isin(exposures)]
    deposit[by] = deposit_name
    data= pd.concat([data,deposit],axis=0)
    
    return data 

def add_lithology_subsets(data, name):
    """
    Creates a subset of the 'data' DataFrame where the 'Stop ID' column matches the given 'name',
    changes all values in the 'Lithology Category' column to the 'name',
    and then appends this modified subset to the original 'data' DataFrame.

    Parameters:
    - data: The original DataFrame to which the new subset will be appended
    - name: The value to match in the 'Stop ID' column and use for the 'Lithology Category'

    Returns:
    - Updated DataFrame with the modified subset appended
    """
    # Create a subset where 'Stop ID' is equal to 'name'
    filtered_subset = data.loc[data['Stop ID'] == name].copy()  # Make a copy to avoid SettingWithCopyWarning
    
    # Change all values in 'Lithology Category' column to the 'name'
    filtered_subset['Lithology Category'] = name
    
    # Append the modified subset to the original DataFrame
    updated_data = pd.concat([data, filtered_subset], axis=0)
    
    return updated_data



def run_shrs_by_plot(df, by, data, colors, title=None, xaxislabel='No', titleOn='no', ax=None, width=4.3):
    """
    df: df of choice
    by: string of column name you want to group the data by for plotting ("Stop ID" or "Lithology Category")
    data: a list of strings of the categories you want plotted. Eg the Stop ID names or the Lithology Categories of interest. 
    colors: a list of colors for the plot.
    title: default is none, give a string if desired.
    xaxislabel: default is none. If "by" variable is "Stop ID" or "Lithology Category", the xlabel will be "Deposit" or "Lithology" respectively. If no xaxis label is desired, set = "No".

    Creates violin plot of desired data, with n printed on fig and min, median, max printed in console.
    """
    
    save = 'n'
    # Determine if we are using a provided axis or need to create a new figure
    if ax is None:
        plt.figure(figsize=(width, 3)) #should be 4.3, 3 or 6,3?
        ax = plt.gca()  # Get the current axis
        save = 'y'

    # New temporary df grouping the given df as desired and selecting for desired data
    datas = df.loc[df[by].isin(data)]
    
    # Create the violin plot
    sns.violinplot(data=datas, x=by, y='Mean_Median SHRS', saturation=1, palette=colors, order=data, ax=ax)
    
    # Update x tick labels to include count for each category
    new_xticklabels = []
    for category in data:
        count = datas[datas[by] == category].shape[0]
        new_xticklabels.append(f'{category}\nn:{count} ') #should be f'{category}\n(n={count})'
    ax.set_xticklabels(new_xticklabels, rotation=45, ha='right')  # Apply rotation and alignment
    
    # Set fixed y-axis limits and ticks
    ax.set_yticks([0, 20, 40, 60, 80])  # Set y-axis ticks to 0, 20, 40, 60, 80
    ax.set_ylim(0, 90)  # Set y-axis limits to 0 to 90
    
    ax.set(ylabel='Schmidt Hammer Rock Strength')
    
    # Set title if required
    if titleOn == 'yes':
        ax.set_title(title)
    
    # Set x-axis label if not set to "No"
    if xaxislabel != 'No':
        if by == "Stop ID":
            xlabel = "Deposit"
        elif by in ['Lithology Category', 'Lithology']:
            xlabel = "Lithology"
        else:
            xlabel = xaxislabel  # Use provided xaxislabel if not 'No'
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel('')  # Remove x-axis label to maintain subplot size

    # Ensure consistent subplot size
    for label in ax.get_xticklabels():
        label.set_fontsize(10)  # Set the x-axis label size
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')  # Rotate labels to fit and avoid overlapping
    
    #kruskal wallis test
    groups = [datas[datas[by] == category]['Mean_Median SHRS'].dropna() for category in data]
    kruskal_result = kruskal(*groups)
    
    print(f"Kruskal-Wallis H-test result: H-statistic = {kruskal_result.statistic:.3f}, p-value = {kruskal_result.pvalue:.3e}")
    # get the median, min, and max values for each category represented
    for category in data:
        category_data = datas[datas[by] == category]['Mean_Median SHRS']
        median_val = category_data.median()
        min_val = category_data.min()
        max_val = category_data.max()
        q1_val = category_data.quantile(0.25)
        q3_val = category_data.quantile(0.75)
        q4_val = category_data.quantile(0.9)
        # Print out min, max, median, Q1, and Q3 values for each category
        print(f"{category} - Min: {min_val:.2f}, Max: {max_val:.2f}, Median: {median_val:.2f}, Q25: {q1_val:.2f}, Q75: {q3_val:.2f}, Q90: {q4_val:.2f}")
    
    if save == 'y':
        # Save and show the plot only if we're creating a new figure
        plt.savefig("SHRS" + title + ".png", dpi=400, bbox_inches="tight")
        plt.show()
    
    
def multi_shrs_plots(plot_details, data, by="Stop ID", title=None, CustomLayout="no"):
    """
    Create a series of violin plots as subplots in a single figure.

    plot_details: A list of dictionaries, each containing details for a single plot.
                  Each dictionary should have 'data', 'colors', 'title', and optionally 'deposit' keys.
    data: DataFrame for the full data set.
    by: The column name to group data by ("Stop ID" or "Lithology Category").
    title: The title for the entire figure.
    CustomLayout: Specify 'yes' for custom layout or 'no' for default layout.
    """

    # Define the figure size
    fig_width = 6.5  # Fixed width
    fig_height = 9  # Fixed height

    # Create a figure with a grid of subplots
    fig = plt.figure(figsize=(fig_width, fig_height))

    if CustomLayout == "no":
        # Create a 2x3 grid for the default layout
        axs = fig.subplots(nrows=3, ncols=2, sharey=False)
        axs = axs.flatten()  # Flatten for easy indexing
        
        # Adjust spacing between subplots
        plt.subplots_adjust(wspace=0, hspace=-1)
        
        # Plot each subplot
        for ax, details in zip(axs, plot_details):
            df = data if 'deposit' not in details else data[data['Stop ID'] == details['deposit']]
            run_shrs_by_plot(
                df,
                by=by,
                data=details['data'],
                colors=details['colors'],
                title=details['title'],
                xaxislabel='No',
                titleOn='no',
                ax=ax
            )

        # Remove unused subplots
        for ax in axs[len(plot_details):]:
            fig.delaxes(ax)

    else:  # CustomLayout == "yes"
        # Create a custom layout using gridspec
        ax_big = plt.subplot2grid((3, 2), (0, 0), colspan=2)
        run_shrs_by_plot(
            data,
            by=by,
            data=plot_details[0]['data'],
            colors=plot_details[0]['colors'],
            title=plot_details[0]['title'],
            xaxislabel='No',
            titleOn='no',
            ax=ax_big
        )

        # Create remaining subplots in a 2x2 grid
        for i, details in enumerate(plot_details[1:]):
            ax = plt.subplot2grid((3, 2), (1 + i // 2, i % 2))
            run_shrs_by_plot(
                data,
                by=by,
                data=details['data'],
                colors=details['colors'],
                title=details['title'],
                xaxislabel='No',
                titleOn='no',
                ax=ax
            )

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save and show the plot
    plt.savefig(f"SHRS{title}.png", dpi=400, bbox_inches="tight")
    plt.show()

# Calculate the empirical cumulative distribution function (ECDF)
def ecdf(data):
    """
    Calculate the empirical cumulative distribution function (ECDF) for a given dataset.

    Parameters:
    data: array-like
        A sequence of numerical values for which the ECDF will be computed.

    Returns:
    x: numpy.ndarray
        Sorted values of the input data.
    y: numpy.ndarray
        Cumulative probabilities corresponding to the sorted values.
    """
    
    # Sort the data in ascending order
    x = np.sort(data)
    # Compute the cumulative probabilities for each data point
    y = np.arange(1, len(data) + 1) / len(data)
    return x, y

def create1000SHRSsample(deposit_name, lith_percent_df, SHRS_df):
    """
    Create a sample DataFrame of 1000 Schmidt Hammer Rock Strength (SHRS) values
    based on lithology percentages using empirical cumulative distribution functions (ECDFs).

    Parameters:
    deposit_name: str
        The name of the deposit for which the SHRS sample is created.
    lith_percent_df: pandas.DataFrame
        A DataFrame containing lithology types and their corresponding fractions.
    SHRS_df: 
        A DataFrame containing lithology types and corresponding strengths.

    Returns:
    new_df: pandas.DataFrame
        A DataFrame containing 1000 SHRS values categorized by lithology.
    """
    # Creating the new DataFrame with a dynamic name
    new_df = pd.DataFrame(columns=SHRS_df.columns)
    
    for lithtype in lith_percent_df['Lithologies']:
        # Calculate ECDF for the original data
        x_original, y_original = ecdf(
            SHRS_df.loc[
                (SHRS_df['Stop ID'] == deposit_name) & 
                (SHRS_df['Lithology Category'] == lithtype)
                ]['Mean_Median SHRS']
            )

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
# Prep DFs for use

# Convert 'Size (cm)' to numeric, coercing errors to NaN
pinke2023_data['Size (cm)'] = pd.to_numeric(pinke2023_data['Size (cm)'], errors='coerce')

# Filter the DataFrame
pinke2023_data = pinke2023_data.loc[(pinke2023_data['Extra SHRS'] == 'y') | (pinke2023_data['Size (cm)'] >= 25)]

#change suiattle column names for merging with 2023 field data
suiattleSHRS_data=suiattleSHRS_data.rename(columns={"Site Num": "Stop ID", 'Lithology': 'Lithology Category','SHRS median': 'Mean_Median SHRS'})
suiattleSHRS_data['Stop ID'] = suiattleSHRS_data['Stop ID'].astype("string")

# merge 2023 and suiattle data to 1 df and drop any empty data
SHRS_data= pd.concat([pinke2023_data,suiattleSHRS_data],axis=0)
SHRS_data= SHRS_data[SHRS_data['Mean_Median SHRS'].notna()]

# merge lithologies
merge_liths = { "NP": 'NV', 'GF': 'PL', 'VNV':'NV', 'VC':'VB', "NAL":"NA", "VB":'VV', 'UG':'WW', 'G':'PL'} # Define replacements {old_lith: new_lith}

for old_lith, new_lith in merge_liths.items():
    SHRS_data.loc[SHRS_data['Lithology Category'] == old_lith, 'Lithology Category'] = new_lith
    
    
# change values of NAL to NA - had to be entered into excel as NAL as NA would appear as empty to python
SHRS_data.loc[SHRS_data['Lithology Category'] == 'NAL', 'Lithology Category'] = 'NA'
#%%
# combine exposures as needed for plotting
         
SHRS_data= add_combine_exposures(SHRS_data,exposures=['KC1','KC2','KC3','KC4','KC5','KC6','KC7'],
                                deposit_name= MRKC) #MR-KC

SHRS_data= add_combine_exposures(SHRS_data,exposures=['T5A','T5B'],
                                deposit_name="T5")

SHRS_data= add_combine_exposures(SHRS_data,exposures=['T4','T5','T6','T7','T8'],
                                deposit_name= MRWR) #MR-WR


SHRS_data= add_combine_exposures(SHRS_data,exposures=['MM1','MM2','MM3','MM4','MM5','MM6','MM7','MM8','MM9', 'MM10', 'MM11'],
                                deposit_name= MMLR)

SHRS_data= add_combine_exposures(SHRS_data,exposures=['MA1','MA1B','MA2','MA2B','MA3','MA3B'],#removed 4/4A
                                deposit_name= MASC) #MA-SC

SHRS_data= add_combine_exposures(SHRS_data,exposures=['9','10'],
                                deposit_name= GPSR) #GP-SR

# Create a filtered copy of SHRS_data
SHRS_Goudie = SHRS_data[SHRS_data['Stop ID'].isin(deposit_list)].copy()
SHRS_Lith = SHRS_data[SHRS_data['Stop ID'].isin(deposit_list)].copy()
#%%
# Plot Figure 5A: SHRS of each deposit as measured.
run_shrs_by_plot(SHRS_data,by="Stop ID", data=[MMLR, GPSR, MRWR, MRKC, MASC],
                    colors =[meagercolor,glaciercolor,tahomacolor,kautzcolor,adamscolor],
                    title='All Deposits SHRS extra OR >=25cm', xaxislabel='No', titleOn='no', width = 3)
#%%
# Lithology fractions for each deposit, obtained from PieLith script analysis
# create dfs with the lithology and corresponding fraction for each deposit

# Data for Meager
meager_df = pd.DataFrame({
    'Lithologies': ['LDV', 'PL', 'HDV', 'OW', 'HGM'],
    'Fraction': [62.150056, 29.339306, 4.255319, 3.023516, 1.231803]})

meager_df['Fraction'] = (meager_df['Fraction'] * 10).round(0)

# Data for Suiattle
suiattle_df = pd.DataFrame ({
    'Lithologies': ['VV', 'NV'], #PL
    'Fraction': [56.959707, 43.040293]}) #split PL 1.831502 evenly to VV and NV #56.043956, 42.124542, 1.831502

suiattle_df['Fraction'] = (suiattle_df['Fraction'] * 10).round(0)

# Data for Tahoma
tahoma_df = pd.DataFrame ({
    'Lithologies': ['PN', 'TV', 'VV', 'WW'],
    'Fraction': [39.344262, 31.876138, 19.125683, 9.653916]})

tahoma_df['Fraction'] = (tahoma_df['Fraction'] * 10).round(0)

# Data for Kautz
kautz_df = pd.DataFrame({
    'Lithologies': ['NV', 'V', 'PL'],
    'Fraction': [89.189189, 9.009009, 1.801802]})

kautz_df['Fraction'] = (kautz_df['Fraction'] * 10).round(0)
#kautz_df.at[2, 'Fraction'] += 1

# Data for Adams
adams_df = pd.DataFrame({
    'Lithologies': ['NA', 'NN', 'VN', 'VA'],
    'Fraction': [49.846154, 29.230769, 11.076923, 9.846154]})

adams_df['Fraction'] = (adams_df['Fraction'] * 10).round(0)
adams_df.at[2, 'Fraction'] += 1
#%%

Meager_weightedSHRS_distrib = create1000SHRSsample(MMLR, meager_df,SHRS_data )
Suiattle_weightedSHRS_distrib = create1000SHRSsample(GPSR, suiattle_df,SHRS_data )
Tahoma_weightedSHRS_distrib = create1000SHRSsample(MRWR, tahoma_df,SHRS_data )
Kautz_weightedSHRS_distrib = create1000SHRSsample(MRKC, kautz_df, SHRS_data)
Adams_weightedSHRS_distrib = create1000SHRSsample(MASC, adams_df, SHRS_data)

weighteddistrib_SHRS_data= pd.concat([Meager_weightedSHRS_distrib, Suiattle_weightedSHRS_distrib, Tahoma_weightedSHRS_distrib, Kautz_weightedSHRS_distrib, Adams_weightedSHRS_distrib], ignore_index=True)


# Plot Figure 5B: SHRS of each deposit weighted by lithology fraction, n for each is 1000.

run_shrs_by_plot(weighteddistrib_SHRS_data,by="Stop ID", data=[MMLR,GPSR, MRWR, MRKC, MASC],
                    colors =[meagercolor,glaciercolor,tahomacolor,kautzcolor,adamscolor],
                    title='All Deposits WEIGHTEDSHRS extra OR >=25cm', xaxislabel='No', titleOn='no', width = 3)
#%%
# Plot Figure 6: SHRS of each deposit, broken up by exposure. 
multi_shrs_plots(
    plot_details=[
        {'data': [MMLR, 'MM1', 'MM2', 'MM3', 'MM4', 'MM5', 'MM6', 'MM7', 'MM8', 'MM9', 'MM10', 'MM11'],
         'colors': [meagercolor] + [DarkGray] +[DarkGray] +[LightGray] +[LightGray] +[DarkGray] +[DarkGray] +[DarkGray] +[LightGray] +[DarkGray] +[LightGray] +[LightGray],
         'title': 'Mt. Meager extra SHRS OR >=25cm'
        },
        {'data': [GPSR, '9', '10'],
         'colors': [glaciercolor, MediumGray, MediumGray],
         'title': 'Suiattle SHRS',
         },
        {'data': [MRWR, 'T4', 'T5', 'T6', 'T7', 'T8'],
         'colors': [tahomacolor] + [DarkGray] +[LightGray] +[LightGray] +[LightGray] +[DarkGray],
         'title': 'Tahoma 2023 extra SHRS OR >=25cm',
         },
        {'data': [MRKC, 'KC1', 'KC2', 'KC3', 'KC4', 'KC5', 'KC6', 'KC7'],
         'colors': [kautzcolor, DarkGray, DarkGray, DarkGray, DarkGray, LightGray, LightGray, LightGray],
         'title': 'Kautz Creek extra SHRS OR >=25cm',
         },
        {'data': [MASC, 'MA1', 'MA1B', 'MA2', 'MA2B', 'MA3', 'MA3B'],
         'colors': [adamscolor] + [LightGray] * 2 + [MediumGray] * 2 + [DarkGray] * 2,
         'title': 'Mt. Adams extra SHRS OR >=25cm',
         }
    ],
    data=SHRS_data,  # Define or load your full SHRS DataFrame
    by="Stop ID",  # Define grouping column, could be "Stop ID" or "Lithology Category"
    title="MultiSubExposures",
    CustomLayout = 'yes'
)
#%%
# for the Figure 7 lith plot, add all deposit exposures as a single lith

SHRS_data=add_lithology_subsets(SHRS_data,MMLR)
SHRS_data=add_lithology_subsets(SHRS_data,GPSR)
SHRS_data=add_lithology_subsets(SHRS_data,MRWR)
SHRS_data=add_lithology_subsets(SHRS_data,MRKC)
SHRS_data=add_lithology_subsets(SHRS_data, MASC)
#%%
# Plot Figure 7: SHRS of each deposit broken up by lithology type.

multi_shrs_plots(
    plot_details=[
        {'data': [MMLR,'LDV', 'OW', 'HDV', 'HGM', 'PL'],
         'colors': [meagercolor, lightpurple, orange, darkpurple, green, pink],
         'title': 'Mt. Meager liths',
         'deposit': MMLR},
        
        {'data': [GPSR,'VV', 'NV'],
         'colors': [glaciercolor, lightpurple, darkpurple],
         'title': 'Suiattle liths',
         'deposit': GPSR},
        
        {'data': [MRWR,'VV','TV', 'WW', 'PN'],
         'colors': [tahomacolor, lightpurple, lightpurple, orange, darkpurple],
         'title': 'Tahoma liths',
         'deposit': MRWR},
        
        {'data': [MRKC,'V','NV', 'PL'],
         'colors': [kautzcolor, lightpurple, darkpurple, pink],
         'title': 'Kautz liths',
         'deposit': MRKC},
        
        {'data': [MASC,'VA', 'VN','NA', 'NN'],
         'colors': [adamscolor,lighterorange, lightpurple, orange, darkpurple],
         'title': 'Mt. Adams liths',
         'deposit': MASC}
    ],
     
    data=SHRS_data,  # Define or load your full SHRS DataFrame
    by="Lithology Category", # Define grouping column, could be "Stop ID" or "Lithology Category"
    title='MultiSubLiths'
)
#%%
# prep data for Figure 15: SHRS of different rock types globally (modified from Goudie (2006), each data point is the mean of many measurements on many samples). Data from this study denoted with *. 

V_PNW = 'Volcanic*'
M_PNW = 'Metamorphic*'
PL_PNW = 'Plutonic*'


volcanic_rocks = ['NV', 'V', 'A','PN', 'WW', 'VV', 'TV', 'breccia', 'LDV', 'HDV', 'OW', 'NA', 'VN', 'VA', 'NN']

SHRS_Goudie.loc[SHRS_Goudie['Lithology Category'].isin(volcanic_rocks), 'Lithology Category'] = V_PNW
SHRS_Goudie.loc[SHRS_Goudie['Lithology Category'] == 'PL' , 'Lithology Category'] = PL_PNW
SHRS_Goudie.loc[SHRS_Goudie['Lithology Category'] == 'HGM', 'Lithology Category'] = M_PNW

Goudie_data = Goudie_data[['Mean R Value', 'Lithology']]
Goudie_data = Goudie_data.rename(columns = {'Mean R Value': 'Mean_Median SHRS'})
Goudie_data = Goudie_data.rename(columns = {'Lithology': 'Lithology Category'})
Goudie_data.loc[Goudie_data['Lithology Category'] == 'Miscellaneous Sedimentary', 'Lithology Category'] = 'Sedimentary'

SHRS_Goudie = SHRS_Goudie[['Lithology Category', 'Mean_Median SHRS']]

SHRS_Goudie = pd.concat([SHRS_Goudie, Goudie_data], ignore_index=True)

# Plot FIGURE 15
# set figure width to 6 to allow space for text on right side
run_shrs_by_plot(SHRS_Goudie,by="Lithology Category", data=['Limestone', 'Sedimentary', 'Volcanic', 'Metamorphic', 'Plutonic', V_PNW, M_PNW, PL_PNW],
                    colors =[light_gray, gray, dim_gray, dark_gray, very_dark_gray, darkpurple, green, pink],
                    title='Global SHRS (Goudie) vs. PNW', xaxislabel='No', titleOn='No', width=6)
