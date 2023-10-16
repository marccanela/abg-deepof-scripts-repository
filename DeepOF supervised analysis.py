"""
Version number 1: 24/07/2023
@author: mcanela
DeepOF SUPERVISED ANALYSIS
https://deepof.readthedocs.io/en/latest/tutorial_notebooks/deepof_supervised_tutorial.html
"""

import deepof.data
import deepof.visuals
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
import copy
import numpy as np
import pingouin as pg
import pickle
import statistics

directory_output = '//FOLDER/becell/Lab Projects/ERCstG_HighMemory/Data/Marc/1) SOC/2023-09 - Young males/DeepOF/'

# =============================================================================
# Loading a previously generated project
# =============================================================================

my_deepof_project = deepof.data.load_project(directory_output + "deepof_tutorial_project")

# =============================================================================
# Running the supervised analysis
# =============================================================================

supervised_annotation = my_deepof_project.supervised_annotation()
with open(directory_output + 'supervised_annotation.pkl', 'wb') as file:
    pickle.dump(supervised_annotation, file)

# Alternatively, open an existing supervised analysis
with open(directory_output + 'supervised_annotation.pkl', 'rb') as file:
    supervised_annotation = pickle.load(file)

# =============================================================================
# Generating gant charts of all traits of a specific video
# =============================================================================

def gantt(my_deepof_project, supervised_annotation):

    plt.figure(figsize=(12, 8))

    deepof.visuals.plot_gantt(
        my_deepof_project,
        '20230921_Marc_ERC SOC S1_Males_box ab_02_01_1',
        supervised_annotations=supervised_annotation,
    )

    return plt

# =============================================================================
# Exploring trait enrichment across conditions
# =============================================================================

def enrichment_plot(my_deepof_project, supervised_annotation):

    fig = plt.figure(figsize=(14, 5)).subplot_mosaic(
        mosaic="""
               AAAAB
               AAAAB
               """,
    )
    
    deepof.visuals.plot_enrichment(
        my_deepof_project,
        supervised_annotations=supervised_annotation,
        add_stats="Mann-Whitney",
        plot_proportions=True,
        bin_index=3, # Bin number to select
        bin_size=60, # Size of the bin in seconds
        ax = fig["A"],
        verbose  = True,
        normalize = True, # Actual time in seconds on the y axis
    )
    
    deepof.visuals.plot_enrichment(
        my_deepof_project,
        supervised_annotations=supervised_annotation,
        add_stats="Mann-Whitney",
        plot_proportions=False,
        bin_index=3, # Bin number to select
        bin_size=60, # Size of the bin in seconds
        ax = fig["B"],
        normalize = True, # Actual time in seconds on the y axis
    )
    
    for ax in fig:
        fig[ax].set_xticklabels(fig[ax].get_xticklabels(), rotation=45, ha='right')
        fig[ax].set_title("")
        fig[ax].set_xlabel("")
    
    fig["A"].get_legend().remove()
    
    plt.tight_layout()
    
    return plt

# =============================================================================
# Generate a CSV file to store your data
# =============================================================================

def csv_generator(supervised_annotation, hue, values_column, bin_num, bin_size, duration):

    '''
    Parameters
    ----------
    values_column: str ('lookaround', 'huddle', 'climbing', or 'sniffing')
    hue: str (str contained in the name of the file. E.g. S1, Hab, etc.)
    bin_num: int (number of bin, considering that starts with 0)
    bin_size: int (duration of the bins, in seconds)
    duration: int (total expected duration of the videos, in seconds)
    '''
    
    copy_supervised_annotation = copy.deepcopy(supervised_annotation)
    dict_of_dataframes = {key: value for key, value in copy_supervised_annotation.items() if hue in str(key)}
    
    num_of_bins = {}
    factor = int(duration/bin_size)
            
    for key, value in dict_of_dataframes.items():
                
        value.reset_index(inplace=True)
        value.drop('index', axis=1, inplace=True)
        value.reset_index(inplace=True)
                
        bin_length = int(len(value) // factor)
        cutoffs = [i * bin_length for i in range(1, factor)]
        
        # Determine the minimum and maximum of the 'index' column
        min_value = value['index'].min()
        max_value = value['index'].max() + 1

        # Add the leftmost and rightmost edges for the first and last bins
        cutoffs = [min_value] + cutoffs + [max_value]
        
        value['bin'] = pd.cut(value['index'], bins=cutoffs, labels=False, right=False, include_lowest=True)
        
        num_of_bins[key] = value
    
    df = pd.concat(num_of_bins.values(), keys=num_of_bins.keys()).reset_index()
    df = df[df.bin == bin_num]
    df.rename(columns={'level_0': 'experiment_id'}, inplace=True)
        
    mean_values = df.groupby(['bin', 'experiment_id'])[values_column].mean()
    mean_values = mean_values.reset_index()
    mean_values[values_column] = mean_values[values_column] * 100
    
    mean_values = mean_values[values_column].tolist()

    return mean_values

# =============================================================================
# Plotting a timelapse plot of a specific trait
# =============================================================================

def timeseries_supervised(supervised_annotation, protocol='S1', selected_column_name='sniffing', ax=None, bin_size=10): # in seconds
    
    if ax is None:
        fig, ax = plt.subplots(1, 1)
        
    dict_of_dataframes = copy.deepcopy(supervised_annotation)
    
    num_of_bins = {}
    for key, value in dict_of_dataframes.items():
        if str(key).split(' ')[-2].split('_')[0] == 'Hab':
            factor = int(1200/bin_size)
        if str(key).split(' ')[-2].split('_')[0] == 'S1':
            factor = int(360/bin_size)
        if str(key).split(' ')[-2].split('_')[0] == 'S2':
            factor = int(360/bin_size)
        
        value.reset_index(inplace=True)
        value.drop('index', axis=1, inplace=True)
        value.reset_index(inplace=True)
                
        bin_length = int(len(value) // factor)
        cutoffs = [i * bin_length for i in range(1, factor)]
        
        # Determine the minimum and maximum of the 'index' column
        min_value = value['index'].min()
        max_value = value['index'].max() + 1

        # Add the leftmost and rightmost edges for the first and last bins
        cutoffs = [min_value] + cutoffs + [max_value]
        
        value['bin'] = pd.cut(value['index'], bins=cutoffs, labels=False, right=False, include_lowest=True)
        
        num_of_bins[key] = value
           
    df = pd.concat(num_of_bins.values(), keys=num_of_bins.keys()).reset_index()
    
    mean_values = df.groupby(['bin', 'level_0'])[selected_column_name].mean()
    mean_values = mean_values.reset_index()
    mean_values['bin'] = mean_values['bin'] + 1
    mean_values['group'] = mean_values.level_0.str.split(' ').str[-1].str.split('_').str[0]
    mean_values['protocol'] = mean_values.level_0.str.split(' ').str[-2].str.split('_').str[0]
    
    mean_values = mean_values[mean_values['protocol'] == protocol]
    
    # Define a mapping of strings to replace and their corresponding replacements
    replace_dict = {
        'ab': 'PP',
        'cd': 'UP',
        'ef': 'PU'
    }
    
    # Replace the strings in the 'column_name' column using the replace method
    # mean_values['group'] = mean_values['group'].replace(replace_dict)
    
    # The numbers of the X axis will be expressed in minutes
    mean_values['bin'] = mean_values.bin / 6
    # The numbers of the Y axis will be expressed as %
    mean_values[selected_column_name] = mean_values[selected_column_name] * 100
    
    sns.set_theme(style="whitegrid")
    sns.lineplot(x='bin', y=selected_column_name, data=mean_values, ax=ax,
                  # hue='group',
                 )
        
    plt.ylim(0,100)
    ax.set(xlabel='Time [min]')
    ax.set(ylabel=selected_column_name.capitalize() + ' (% of time)')
    handles, labels = ax.get_legend_handles_labels()
    
    off_list = []
    off1_coords = plt.Rectangle((0, 0), 3, 100) # 3 min OFF
    off_list.append(off1_coords)
    off2_coords = plt.Rectangle((4, 0), 1, 100) # 1 min OFF
    off_list.append(off2_coords)    
    off_coll = PatchCollection(off_list, alpha=0.1, color='yellow')
    ax.add_collection(off_coll)
    off_coll_border = PatchCollection(off_list, facecolor='none', edgecolor='black', alpha=0.5)
    ax.add_collection(off_coll_border)
    off_patch = mpatches.Patch(color='yellow', label='off period', alpha=0.1)
    handles.append(off_patch)
    
    if protocol == 'S2':
        probetest_list = []
        probetest_coords = plt.Rectangle((3, 0), 1, 100) # 5 min probetest
        probetest_list.append(probetest_coords)
        probetest_coords_2 = plt.Rectangle((5, 0), 1, 100) # 1 min probetest
        probetest_list.append(probetest_coords_2)
        probetest_coll = PatchCollection(probetest_list, alpha=0.1, color='blue')
        ax.add_collection(probetest_coll)
        probetest_coll_border = PatchCollection(probetest_list, facecolor='none', edgecolor='black', alpha=0.5)
        ax.add_collection(probetest_coll_border)
        tone_patch = mpatches.Patch(color='blue', label='S2', alpha=0.1)
        handles.append(tone_patch)
    
    else:
        probetest_list = []
        probetest_coords = plt.Rectangle((3, 0), 1, 100) # 1 min probetest
        probetest_list.append(probetest_coords)
        probetest_coords_2 = plt.Rectangle((5, 0), 1, 100) # 1 min probetest
        probetest_list.append(probetest_coords_2)
        probetest_coll = PatchCollection(probetest_list, alpha=0.1, color='red')
        ax.add_collection(probetest_coll)
        probetest_coll_border = PatchCollection(probetest_list, facecolor='none', edgecolor='black', alpha=0.5)
        ax.add_collection(probetest_coll_border)
        tone_patch = mpatches.Patch(color='red', label='S1', alpha=0.1)
        handles.append(tone_patch)
        
    plt.legend(handles=handles)
       
    return ax

# =============================================================================
# Plotting barplots based on the CSV file
# =============================================================================

def barplot_OffOn(ax=None, hue='S2', values_column='huddle', bin_size=60, duration=360, offn=2, onn=3):

    if ax is None:
        fig, ax = plt.subplots()
    
    sns.set_theme(style="whitegrid")
    
    off_position = 0
    on_position = 1
    # bar_width = 0.6
    
    off_list = csv_generator(supervised_annotation, hue, values_column, offn, bin_size, duration)
    on_list = csv_generator(supervised_annotation, hue, values_column, onn, bin_size, duration)
                  
    off_data = np.mean(off_list)
    on_data = np.mean(on_list)

    off_error = np.std(off_list, ddof=1)
    on_error = np.std(on_list, ddof=1)
    
    if hue == 'S1':
        on_color = 'salmon'
        on_edge = 'darkred'
    elif hue == 'S2':
        on_color = 'cornflowerblue'
        on_edge = 'darkblue'
    else:
        on_color = 'moccasin'
        on_edge = 'darkorange'
    
    # ax.bar(off_position, off_data, color='moccasin', edgecolor='black', width=bar_width)
    # ax.bar(on_position, on_data, color=on_color, edgecolor='black', width=bar_width)
    
    ax.hlines(off_data, xmin=-0.25, xmax=0.25, color='black', linewidth=1)
    ax.hlines(on_data, xmin=0.75, xmax=1.25, color='black', linewidth=1)
    
    ax.errorbar(off_position, off_data, yerr=off_error, lolims=False, capsize = 3, ls='None', color='k', zorder=-1)
    ax.errorbar(on_position, on_data, yerr=on_error, lolims=False, capsize = 3, ls='None', color='k', zorder=-1)
    
    ax.set_xticks([0.5])
    ax.set_xticklabels([])
    
    jitter = 0.15 # Dots dispersion
    
    dispersion_values_off = np.random.normal(loc=off_position, scale=jitter, size=len(off_list)).tolist()
    ax.plot(dispersion_values_off, off_list,
            'o',                            
            markerfacecolor='moccasin',    
            markeredgecolor='darkorange',
            markeredgewidth=1,
            markersize=5, 
            label='Absent stimulus')      
    
    dispersion_values_on = np.random.normal(loc=on_position, scale=jitter, size=len(on_list)).tolist()
    ax.plot(dispersion_values_on, on_list,
            'o',                          
            markerfacecolor=on_color,    
            markeredgecolor=on_edge,
            markeredgewidth=1,
            markersize=5, 
            label=hue)               
    
    if len(off_list) == len(on_list):
        for x in range(len(off_list)):
            ax.plot([dispersion_values_off[x], dispersion_values_on[x]], [off_list[x], on_list[x]], color = 'black', linestyle='-', linewidth=0.5)
        
    ax.set_ylim(0,100)
    ax.set_xlabel('')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
    
    pvalue = pg.ttest(off_list, on_list, paired=True)['p-val'][0]
    
    def convert_pvalue_to_asterisks(pvalue):
        ns = "ns (p=" + str(pvalue)[1:4] + ")"
        if pvalue <= 0.0001:
            return "****"
        elif pvalue <= 0.001:
            return "***"
        elif pvalue <= 0.01:
            return "**"
        elif pvalue <= 0.05:
            return "*"
        return ns

    y, h, col = max(max(off_list), max(on_list)) + 5, 2, 'k'
    
    ax.plot([off_position, off_position, on_position, on_position], [y, y+h, y+h, y], lw=1.5, c=col)
    
    if pvalue > 0.05:
        ax.text((off_position+on_position)*.5, y+2*h, convert_pvalue_to_asterisks(pvalue), ha='center', va='bottom', color=col, size=11)
    elif pvalue <= 0.05:    
        ax.text((off_position+on_position)*.5, y, convert_pvalue_to_asterisks(pvalue), ha='center', va='bottom', color=col, size=18)

    return ax


def OffOn_multiplot():
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 4))  # 1 row, 2 columns
    
    # bin_size=10, duration=360, offn=17, onn=18
    # bin_size=60, duration=360, offn=2, onn=3
    
    # Plot data on ax1
    barplot_OffOn(ax=ax1, hue='S1', values_column='lookaround', bin_size=60, duration=360, offn=2, onn=3)
    
    # Plot data on ax2
    barplot_OffOn(ax=ax2, hue='S2', values_column='lookaround', bin_size=60, duration=360, offn=2, onn=3)
    
    # Adjust the spacing between subplots
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # plt.suptitle('Lookaround (%): 170-180 vs 180-190 s')
    plt.suptitle('Lookaround (%): 2-3 vs 3-4 min')
    
    # Show the plots
    plt.show()


def discrimination_index(ax=None, index='di', hue_1='S1', hue_2='S2', values_column='huddle', bin_size=60, duration=360, offn=2, onn=3):

    if ax is None:
        fig, ax = plt.subplots()
    
    sns.set_theme(style="whitegrid")
    
    position_1 = 0
    position_2 = 1
    # bar_width = 0.6
    
    off_list_1 = csv_generator(supervised_annotation, hue_1, values_column, offn, bin_size, duration)
    on_list_1 = csv_generator(supervised_annotation, hue_1, values_column, onn, bin_size, duration)
    
    off_list_2 = csv_generator(supervised_annotation, hue_2, values_column, offn, bin_size, duration)
    on_list_2 = csv_generator(supervised_annotation, hue_2, values_column, onn, bin_size, duration)

    # DISCRIMINATION INDEX
    if index == 'di':
        ax.set_ylim(-1,1)
        ax.axhline(y=0, color='black', linestyle='--')
        if len(off_list_1) == len(on_list_1):
            subtraction_1 = [x - y for x, y in zip(on_list_1, off_list_1)]
            addition_1 = [x + y for x, y in zip(on_list_1, off_list_1)]
            discrimination_1 = [x / y for x, y in zip(subtraction_1, addition_1)]
        if len(off_list_2) == len(on_list_2):
            subtraction_2 = [x - y for x, y in zip(on_list_2, off_list_2)]
            addition_2 = [x + y for x, y in zip(on_list_2, off_list_2)]
            discrimination_2 = [x / y for x, y in zip(subtraction_2, addition_2)]
         
    # GENERALIZATION INDEX
    elif index == 'gi':
        ax.set_ylim(0,1)
        if len(off_list_1) == len(on_list_1):
            addition_1 = [x + y for x, y in zip(on_list_1, off_list_1)]
            discrimination_1 = [x / y for x, y in zip(off_list_1, addition_1)]
        if len(off_list_2) == len(on_list_2):
            addition_2 = [x + y for x, y in zip(on_list_2, off_list_2)]
            discrimination_2 = [x / y for x, y in zip(off_list_2, addition_2)]    
                   
    discrimination_data_1 = np.mean(discrimination_1)
    discrimination_data_2 = np.mean(discrimination_2)

    discrimination_error_1 = np.std(discrimination_1, ddof=1)
    discrimination_error_2 = np.std(discrimination_2, ddof=1)
     
    if hue_1 == 'S1':
        on_color_1 = 'salmon'
        on_edge_1 = 'darkred'
    elif hue_1 == 'S2':
        on_color_1 = 'cornflowerblue'
        on_edge_1 = 'darkblue'
    else:
        on_color_1 = 'moccasin'
        on_edge_1 = 'darkorange'
         
    if hue_2 == 'S1':
        on_color_2 = 'salmon'
        on_edge_2 = 'darkred'
    elif hue_2 == 'S2':
        on_color_2 = 'cornflowerblue'
        on_edge_2 = 'darkblue'
    else:
        on_color_2 = 'moccasin'
        on_edge_2 = 'darkorange'
     
    ax.hlines(discrimination_data_1, xmin=-0.25, xmax=0.25, color='black', linewidth=1)
    ax.hlines(discrimination_data_2, xmin=0.75, xmax=1.25, color='black', linewidth=1)
    
    ax.errorbar(position_1, discrimination_data_1, yerr=discrimination_error_1, lolims=False, capsize = 3, ls='None', color='k', zorder=-1)
    ax.errorbar(position_2, discrimination_data_2, yerr=discrimination_error_2, lolims=False, capsize = 3, ls='None', color='k', zorder=-1)
    
    ax.set_xticks([0.5])
    ax.set_xticklabels([])
    
    jitter = 0.15 # Dots dispersion
    
    dispersion_values_off = np.random.normal(loc=position_1, scale=jitter, size=len(discrimination_1)).tolist()
    ax.plot(dispersion_values_off, discrimination_1,
            'o',                            
            markerfacecolor=on_color_1,    
            markeredgecolor=on_edge_1,
            markeredgewidth=1,
            markersize=5, 
            label=hue_1)      
    
    dispersion_values_on = np.random.normal(loc=position_2, scale=jitter, size=len(discrimination_2)).tolist()
    ax.plot(dispersion_values_on, discrimination_2,
            'o',                          
            markerfacecolor=on_color_2,    
            markeredgecolor=on_edge_2,
            markeredgewidth=1,
            markersize=5, 
            label=hue_2)               
    
    # if len(discrimination_1) == len(discrimination_2):
    #     for x in range(len(discrimination_1)):
    #         ax.plot([dispersion_values_off[x], dispersion_values_on[x]], [discrimination_1[x], discrimination_2[x]], color = 'black', linestyle='-', linewidth=0.5)
            
    pvalue = pg.ttest(discrimination_1, discrimination_2, paired=True)['p-val'][0]
    
    def convert_pvalue_to_asterisks(pvalue):
        ns = "ns (p=" + str(pvalue)[1:4] + ")"
        if pvalue <= 0.0001:
            return "****"
        elif pvalue <= 0.001:
            return "***"
        elif pvalue <= 0.01:
            return "**"
        elif pvalue <= 0.05:
            return "*"
        return ns
    
    y, h, col = max(max(discrimination_1), max(discrimination_2)) + 0.05, 0.02, 'k'
    
    ax.plot([position_1, position_1, position_2, position_2], [y, y+h, y+h, y], lw=1.5, c=col)
    
    if pvalue > 0.05:
        ax.text((position_1+position_2)*.5, y+2*h, convert_pvalue_to_asterisks(pvalue), ha='center', va='bottom', color=col, size=11)
    elif pvalue <= 0.05:    
        ax.text((position_1+position_2)*.5, y, convert_pvalue_to_asterisks(pvalue), ha='center', va='bottom', color=col, size=18)
    
    ax.set_xlabel('')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
    
    return ax
   

def multiplot_index():
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4))  # 1 row, 2 columns
    
    # Plot data on ax1
    discrimination_index(ax=ax1, index='di', hue_1='S1', hue_2='S2', values_column='lookaround', bin_size=60, duration=360, offn=2, onn=3)
    ax1.set_title('Discrimination Index (D.I.)')
    
    # Plot data on ax2
    discrimination_index(ax=ax2, index='gi', hue_1='S1', hue_2='S2', values_column='lookaround', bin_size=60, duration=360, offn=2, onn=3)
    ax2.set_title('Generalization Index (G.I.)')

    # Adjust the spacing between subplots
    plt.tight_layout()  # Increase the pad value to increase the space between plots
    plt.subplots_adjust(top=0.85, bottom=0.15)
    
    plt.suptitle('Lookaround: 2-3 vs 3-4 min')
    # plt.suptitle('PU: 170-180 vs 180-190 s')
    
    # Show the plots
    plt.show()


def cohens_d(supervised_annotation, hue_1='S1', hue_2='S2', values_column='climbing', bin_size=60, duration=360):

    protocols = [hue_1, hue_2]
    cohens_d_list = []

    for hue in protocols:
        data_1 = csv_generator(supervised_annotation, hue, values_column, 2, bin_size, duration)
        data_2 = csv_generator(supervised_annotation, hue, values_column, 3, bin_size, duration)
        data_3 = csv_generator(supervised_annotation, hue, values_column, 4, bin_size, duration)
        data_4 = csv_generator(supervised_annotation, hue, values_column, 5, bin_size, duration)
        
        # arcsine_data_1 = [np.arcsin(np.sqrt(p / 100)) for p in data_1]
        # arcsine_data_2 = [np.arcsin(np.sqrt(p / 100)) for p in data_2]
        # arcsine_data_3 = [np.arcsin(np.sqrt(p / 100)) for p in data_3]
        # arcsine_data_4 = [np.arcsin(np.sqrt(p / 100)) for p in data_4]              
    
        # Calculate the means and standard deviations for the groups
        mean1 = np.mean(data_1)
        mean2 = np.mean(data_2)
        mean3 = np.mean(data_3)
        mean4 = np.mean(data_4)
    
        std1 = np.std(data_1, ddof=1)  # ddof=1 for sample standard deviation
        std2 = np.std(data_2, ddof=1)
        std3 = np.std(data_3, ddof=1)
        std4 = np.std(data_4, ddof=1)
        
        # Calculate Cohen's d
        pooled_std_1 = np.sqrt(((len(data_1) - 1) * std1**2 + (len(data_2) - 1) * std2**2) / (len(data_1) + len(data_2) - 2))
        cohen_d_1 = (mean2 - mean1) / pooled_std_1
        pooled_std_2 = np.sqrt(((len(data_2) - 1) * std2**2 + (len(data_3) - 1) * std3**2) / (len(data_2) + len(data_3) - 2))
        cohen_d_2 = (mean2 - mean3) / pooled_std_2
        pooled_std_3 = np.sqrt(((len(data_3) - 1) * std3**2 + (len(data_4) - 1) * std4**2) / (len(data_3) + len(data_4) - 2))
        cohen_d_3 = (mean4 - mean3) / pooled_std_3
        
        cohens_d = (cohen_d_1 + cohen_d_2 + cohen_d_3) / 3
        cohens_d_list.append(cohens_d)
    
    cohens_d_1 = cohens_d_list[0]
    cohens_d_2 = cohens_d_list[1]

    return cohens_d_1, cohens_d_2


def easy_index(supervised_annotation, ax=None, hue_1='S1', hue_2='S2', values_column='climbing', bin_size=60, duration=360):

    if ax is None:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        
    protocols = [hue_1, hue_2]
    protocols_dict = {}
    
    for hue in protocols:
        data_1 = csv_generator(supervised_annotation, hue, values_column, 2, bin_size, duration)
        data_2 = csv_generator(supervised_annotation, hue, values_column, 3, bin_size, duration)
        data_3 = csv_generator(supervised_annotation, hue, values_column, 4, bin_size, duration)
        data_4 = csv_generator(supervised_annotation, hue, values_column, 5, bin_size, duration)
        
        # arcsine_data_1 = [np.arcsin(np.sqrt(p / 100)) for p in data_1]
        # arcsine_data_2 = [np.arcsin(np.sqrt(p / 100)) for p in data_2]
        # arcsine_data_3 = [np.arcsin(np.sqrt(p / 100)) for p in data_3]
        # arcsine_data_4 = [np.arcsin(np.sqrt(p / 100)) for p in data_4]              
        
        # protocols_dict[protocol] = [arcsine_data_1, arcsine_data_2, arcsine_data_3, arcsine_data_4]
        protocols_dict[hue] = [data_1, data_2, data_3, data_4]
        
    # EASY INDEX
    data_dict = {}
    for key, value in protocols_dict.items():
        rmsd_list = []
        for i in range(len(data_1)):
            # squared_deviations = [(value[1][i] - value[0][i]) / (value[1][i] + value[0][i]),
            #                         (value[1][i] - value[2][i]) / (value[1][i] + value[2][i]),
            #                         (value[3][i] - value[2][i]) / (value[3][i] + value[2][i])
            #                       ]
            
            # For climbing
            squared_deviations = [(value[0][i] - value[1][i]) / (value[1][i] + value[0][i]),
                                    (value[2][i] - value[1][i]) / (value[1][i] + value[2][i]),
                                    (value[2][i] - value[3][i]) / (value[3][i] + value[2][i])
                                  ]
                        
            rmsd = statistics.mean(squared_deviations)
            rmsd_list.append(rmsd)
        data_dict[key] = rmsd_list
        
    tags = ['D.I.>0.4', '0.3<D.I.<0.4', '0.2<D.I.<0.3', '0.1<D.I.<0.2', 'D.I.<0.1']
    colors = ['#64ff5c', '#bad900', '#e8ae00', '#ff7f1d', '#ff4f4f']
    
    def count_function(data_dict, tags, colors, protocol):
        discrimination = data_dict[protocol]
        count_non = sum(1 for num in discrimination if -1 <= num < 0.1)
        count_poor = sum(1 for num in discrimination if 0.1 <= num < 0.2)
        count_average = sum(1 for num in discrimination if 0.2 <= num < 0.3)
        count_good = sum(1 for num in discrimination if 0.3 <= num < 0.4)
        count_excellent = sum(1 for num in discrimination if 0.4 <= num < 1)    
        values = [count_excellent, count_good, count_average, count_poor, count_non]
        colors_dict = dict(zip(tags, colors))
        values_dict = dict(zip(tags, values))
        for key, value in list(values_dict.items()):
            if value == 0:
                del colors_dict[key]
                del values_dict[key]
        return values_dict, colors_dict
                
    values_1_dict, colors_dict_1 = count_function(data_dict, tags, colors, hue_1)                
    values_2_dict, colors_dict_2 = count_function(data_dict, tags, colors, hue_2)                         
            
    ax1.pie(values_1_dict.values(), autopct='%1.1f%%', labels=values_1_dict.keys(), colors=colors_dict_1.values())
    ax1.set_title(hue_1.capitalize())
    
    ax2.pie(values_2_dict.values(), autopct='%1.1f%%', labels=values_2_dict.keys(), colors=colors_dict_2.values())
    ax2.set_title(hue_2.capitalize())

    plt.show()


def easy_index_dots(supervised_annotation, ax=None, hue_1='S1', hue_2='S2', values_column='climbing', bin_size=60, duration=360):
    
    if ax is None:
        fig, ax = plt.subplots()
    
    sns.set_theme(style="whitegrid")
        
    position_1 = 0
    position_2 = 1
        
    protocols = [hue_1, hue_2]
    protocols_dict = {}
    
    for hue in protocols:
        data_1 = csv_generator(supervised_annotation, hue, values_column, 2, bin_size, duration)
        data_2 = csv_generator(supervised_annotation, hue, values_column, 3, bin_size, duration)
        data_3 = csv_generator(supervised_annotation, hue, values_column, 4, bin_size, duration)
        data_4 = csv_generator(supervised_annotation, hue, values_column, 5, bin_size, duration)
        
        # arcsine_data_1 = [np.arcsin(np.sqrt(p / 100)) for p in data_1]
        # arcsine_data_2 = [np.arcsin(np.sqrt(p / 100)) for p in data_2]
        # arcsine_data_3 = [np.arcsin(np.sqrt(p / 100)) for p in data_3]
        # arcsine_data_4 = [np.arcsin(np.sqrt(p / 100)) for p in data_4]      
                
        # protocols_dict[protocol] = [arcsine_data_1, arcsine_data_2, arcsine_data_3, arcsine_data_4]
        protocols_dict[hue] = [data_1, data_2, data_3, data_4]

    # EASY INDEX
    data_dict = {}
    for key, value in protocols_dict.items():
        rmsd_list = []
        for i in range(len(data_1)): 
            # squared_deviations = [(value[1][i] - value[0][i]) / (value[1][i] + value[0][i]),
            #                         (value[1][i] - value[2][i]) / (value[1][i] + value[2][i]),
            #                         (value[3][i] - value[2][i]) / (value[3][i] + value[2][i])
            #                       ]
            
            # For climbing
            squared_deviations = [(value[0][i] - value[1][i]) / (value[1][i] + value[0][i]),
                                    (value[2][i] - value[1][i]) / (value[1][i] + value[2][i]),
                                    (value[2][i] - value[3][i]) / (value[3][i] + value[2][i])
                                  ]
                         
            rmsd = statistics.mean(squared_deviations)
            rmsd_list.append(rmsd)
        data_dict[key] = rmsd_list

    discrimination_data_1 = np.mean(data_dict[hue_1])
    discrimination_data_2 = np.mean(data_dict[hue_2])

    discrimination_error_1 = np.std(data_dict[hue_1], ddof=1)
    discrimination_error_2 = np.std(data_dict[hue_2], ddof=1)
    
    if hue_1 == 'S1':
        on_color_1 = 'salmon'
        on_edge_1 = 'darkred'
    elif hue_1 == 'S2':
        on_color_1 = 'cornflowerblue'
        on_edge_1 = 'darkblue'
    else:
        on_color_1 = 'moccasin'
        on_edge_1 = 'darkorange'
        
    if hue_2 == 'S1':
        on_color_2 = 'salmon'
        on_edge_2 = 'darkred'
    elif hue_2 == 'S2':
        on_color_2 = 'cornflowerblue'
        on_edge_2 = 'darkblue'
    else:
        on_color_2 = 'moccasin'
        on_edge_2 = 'darkorange'
    
    ax.set_ylim(-1,1)
    ax.axhline(y=0, color='black', linestyle='--')
        
    ax.hlines(discrimination_data_1, xmin=-0.25, xmax=0.25, color='black', linewidth=1)
    ax.hlines(discrimination_data_2, xmin=0.75, xmax=1.25, color='black', linewidth=1)
    
    ax.errorbar(position_1, discrimination_data_1, yerr=discrimination_error_1, lolims=False, capsize = 3, ls='None', color='k', zorder=-1)
    ax.errorbar(position_2, discrimination_data_2, yerr=discrimination_error_2, lolims=False, capsize = 3, ls='None', color='k', zorder=-1)
    
    ax.set_xticks([0.5])
    ax.set_xticklabels([])
    
    jitter = 0.15 # Dots dispersion
    
    dispersion_values_off = np.random.normal(loc=position_1, scale=jitter, size=len(data_dict[protocols[0]])).tolist()
    ax.plot(dispersion_values_off, data_dict[hue_1],
            'o',                            
            markerfacecolor=on_color_1,    
            markeredgecolor=on_edge_1,
            markeredgewidth=1,
            markersize=5, 
            label=hue_1)      
    
    dispersion_values_on = np.random.normal(loc=position_2, scale=jitter, size=len(data_dict[protocols[1]])).tolist()
    ax.plot(dispersion_values_on, data_dict[hue_2],
            'o',                          
            markerfacecolor=on_color_2,    
            markeredgecolor=on_edge_2,
            markeredgewidth=1,
            markersize=5, 
            label=hue_2)               
                
    pvalue = pg.ttest(data_dict[hue_1], data_dict[hue_2], paired=True)['p-val'][0]
    
    def convert_pvalue_to_asterisks(pvalue):
        ns = "ns (p=" + str(pvalue)[1:4] + ")"
        if pvalue <= 0.0001:
            return "****"
        elif pvalue <= 0.001:
            return "***"
        elif pvalue <= 0.01:
            return "**"
        elif pvalue <= 0.05:
            return "*"
        return ns

    y, h, col = max(max(data_dict[protocols[0]]), max(data_dict[protocols[1]])) + 0.05, 0.02, 'k'
    
    ax.plot([position_1, position_1, position_2, position_2], [y, y+h, y+h, y], lw=1.5, c=col)
    
    if pvalue > 0.05:
        ax.text((position_1+position_2)*.5, y+2*h, convert_pvalue_to_asterisks(pvalue), ha='center', va='bottom', color=col, size=11)
    elif pvalue <= 0.05:    
        ax.text((position_1+position_2)*.5, y, convert_pvalue_to_asterisks(pvalue), ha='center', va='bottom', color=col, size=18)
    
    ax.set_xlabel('')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)

    return ax



























