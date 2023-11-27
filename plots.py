"""
Version number 1: 02/11/2023
@author: mcanela
DeepOF SUPERVISED ANALYSIS TIMESERIES PLOTS
"""

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import PatchCollection
import matplotlib.ticker as mtick

# =============================================================================
# Defining functions for future plotting
# =============================================================================

def filter_out_other_behavior(supervised_annotation, filter_out):
    '''
    Parameters
    ----------
    supervised_annotation: data.TableDict from DeepOF
    filter_out: str (huddle, lookaround, etc.)
    '''
    
    copy_supervised_annotation = copy.deepcopy(supervised_annotation)

    for key, df in copy_supervised_annotation.items():
        mask = df[filter_out] == 1
        df[mask] = np.nan
        copy_supervised_annotation[key] = df       
        
    return copy_supervised_annotation


def data_to_plot_mask(directory_output, conditions_cols, specific_conditions):
    '''
    Parameters
    ----------
    directory_output: str
    conditions: list (names of the columns from the conditions.csv)
    specific_conditions: list (names of the specific value of the conditions)
    '''
    
    conditions = pd.read_csv(directory_output + "conditions.csv")
    
    conditions_dict = dict(zip(conditions_cols, specific_conditions))
    
    mask_list = []
    for key, value in conditions_dict.items():
        mask = np.array(conditions[key] == value)
        mask_list.append(mask)
    
    global_mask = mask_list[0]
    for mask in mask_list[1:]:
        global_mask &= mask
    
    experiment_ids = conditions[global_mask]['experiment_id'].to_list()  
    
    return experiment_ids


def data_set_to_plot(supervised_annotation, directory_output, conditions_cols, specific_conditions, behavior, length=360, bin_size=10, filter_out=''):
    '''
    Parameters
    ----------
    supervised_annotation: data.TableDict from DeepOF
    directory_output: str
    conditions: list (names of the columns from the conditions.csv)
    specific_conditions: list (names of the specific value of the conditions)
    behavior: str (huddle, lookaround, etc.)
    length: int (expected size of the video in seconds)
    bin_size: int (bin size in seconds)
    filter_out: str (do you want to filter out? yes or no)
    '''
    
    if filter_out != '':
        copy_supervised_annotation = filter_out_other_behavior(supervised_annotation, filter_out)
    else:
        copy_supervised_annotation = copy.deepcopy(supervised_annotation)

    experiment_ids = data_to_plot_mask(directory_output, conditions_cols, specific_conditions)
    dict_of_dataframes = {key: value for key, value in copy_supervised_annotation.items() if key in experiment_ids}   
    
    num_of_bins = {}
    for key, value in dict_of_dataframes.items():
        factor = int(length/bin_size)
        
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
    
    mean_values = df.groupby(['bin', 'level_0'])[behavior].mean()
    mean_values = mean_values.reset_index()
    mean_values['bin'] = mean_values['bin'] + 1
    
    # The numbers of the Y axis will be expressed as %
    mean_values[behavior] = mean_values[behavior] * 100
    
    return mean_values


def calculate_discrimination(data1, data2):
    
    if len(data1) == len(data2):
        subtraction = [x - y for x, y in zip(data2, data1)]
        addition = [x + y for x, y in zip(data2, data1)]
        discrimination = [x / y for x, y in zip(subtraction, addition)]
    
    return discrimination


def calculate_global_discrimination(data1, data2, data3, data4): 
    
    if len(data1) == len(data2):
        subtraction = [x - y for x, y in zip(data2, data1)]
        addition = [x + y for x, y in zip(data2, data1)]
        discrimination_1 = [x / y for x, y in zip(subtraction, addition)]
    
    if len(data2) == len(data3):
        subtraction = [x - y for x, y in zip(data2, data3)]
        addition = [x + y for x, y in zip(data2, data3)]
        discrimination_2 = [x / y for x, y in zip(subtraction, addition)]
    
    if len(data3) == len(data4):
        subtraction = [x - y for x, y in zip(data4, data3)]
        addition = [x + y for x, y in zip(data4, data3)]
        discrimination_3 = [x / y for x, y in zip(subtraction, addition)]

    combined_lists = zip(discrimination_1, discrimination_2, discrimination_3)
    mean_list = [sum(x) / len(x) for x in combined_lists]
    
    return mean_list


def count_function(my_list):
        
    total = sum(1 for num in my_list)
    count_non = sum(1 for num in my_list if -1 <= num < 0.1)
    count_poor = sum(1 for num in my_list if 0.1 <= num < 0.2)
    count_average = sum(1 for num in my_list if 0.2 <= num < 0.3)
    count_good = sum(1 for num in my_list if 0.3 <= num < 0.4)
    count_excellent = sum(1 for num in my_list if 0.4 <= num < 1) 
    # values = [count_non, count_poor, count_average, count_good, count_excellent]
    values = [count_excellent, count_good, count_average, count_poor, count_non]
    values = [x/total for x in values]
    values = [x*100 for x in values]
    
    # values_dict = dict(zip(tags, values))
    # for key, value in list(values_dict.items()):
    #     if value == 0:
    #         del values_dict[key]
            
    return values


# =============================================================================
# Plot functions
# =============================================================================

def timeseries(supervised_annotation, directory_output, column='lookaround', ax=None):
    '''
    Parameters
    ----------
    supervised_annotation: data.TableDict from DeepOF
    directory_output: str
    column: str (huddle, lookaround, etc.)
    '''
    
    # blue color storytelling = #194680
    # red color storytelling = #801946
    # other blues = royalblue
    # grey color storytelling = #636466
    
    color_contrast = '#801946'
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(7,4))
        
    sns.set_theme(style="whitegrid")
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    
    label_offset = 0.2  # Offset for label positioning
    
    data1 = data_set_to_plot(supervised_annotation, directory_output, ['group'], ['s1'], column)
    data1['bin']= data1['bin'] / 6
    sns.lineplot(x=data1['bin'], y=data1[column], label='', legend=None, color=color_contrast)
    ax.text(data1['bin'].iloc[-1] + label_offset, data1[data1.bin == data1.bin.iloc[-1]][column].mean(), 'Direct', fontsize=12, color=color_contrast, weight='bold')

    data2 = data_set_to_plot(supervised_annotation, directory_output, ['group'], ['s2'], column)
    data2['bin']= data2['bin'] / 6
    sns.lineplot(x=data2['bin'], y=data2[column], label='', legend=None, color='grey')    
    ax.text(data2['bin'].iloc[-1] + label_offset, data2[data2.bin == data2.bin.iloc[-1]][column].mean(), 'Mediated', fontsize=12, color='grey', weight='bold')

    plt.ylim(0,100)
    ax.set_xlabel('Minutes', loc='left')
    ax.set_ylabel('Percentage of time doing ' + column, loc='top')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter()) #Add % symbol to the Y axis
    
    plt.title(column.capitalize() + " response in young females", loc = 'left', color='#636466')
    
    # Grey color
    ax.xaxis.label.set_color('#636466')
    ax.yaxis.label.set_color('#636466')
    ax.tick_params(axis='x', colors='#636466')
    ax.tick_params(axis='y', colors='#636466')
    # for spine in plt.gca().spines.values():
    #     spine.set_edgecolor('#636466')
    
    # Shaded area
    probetest_list = []
    probetest_coords = plt.Rectangle((3, 0), 1, 100)
    probetest_list.append(probetest_coords)
    probetest_coords_2 = plt.Rectangle((5, 0), 1, 100)
    probetest_list.append(probetest_coords_2)
    probetest_coll = PatchCollection(probetest_list, alpha=0.1, color='#636466')
    ax.add_collection(probetest_coll)
    probetest_coll_border = PatchCollection(probetest_list, facecolor='none', edgecolor='#636466', alpha=0.5)
    ax.add_collection(probetest_coll_border)
    ax.annotate('Cue', (3.5, 90), ha='center', fontsize=12, color='#636466', weight='bold', alpha=0.5)
    ax.annotate('Cue', (5.5, 90), ha='center', fontsize=12, color='#636466', weight='bold', alpha=0.5)
    
    plt.tight_layout()
    return ax


def boxplot(supervised_annotation, directory_output, column='immobility', ax=None):

    color_contrast = '#801946'    

    if ax is None:
        fig, ax = plt.subplots(figsize=(3.5,4))

    
    sns.set_theme(style="whitegrid")
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    
    data1_position = 0
    data1 = data_set_to_plot(supervised_annotation, directory_output, ['group'], ['s1'], column, bin_size=60)
    data1 = data1[data1.bin == 3] # The bin starts with 1 (i.e., 1, 2, 3, 4, etc.)
    data1 = data1[column].tolist()
    data1_mean = np.mean(data1)
    data1_error = np.std(data1, ddof=1)

    data2_position = 1
    data2 = data_set_to_plot(supervised_annotation, directory_output, ['group'], ['s1'], column, bin_size=60)
    data2 = data2[data2.bin == 4] # The bin starts with 1 (i.e., 1, 2, 3, 4, etc.)
    data2 = data2[column].tolist()
    data2_mean = np.mean(data2)
    data2_error = np.std(data2, ddof=1)

    ax.hlines(data1_mean, xmin=data1_position-0.25, xmax=data1_position+0.25, color='#636466', linewidth=1.5)
    ax.hlines(data2_mean, xmin=data2_position-0.25, xmax=data2_position+0.25, color='#636466', linewidth=1.5)
    
    ax.errorbar(data1_position, data1_mean, yerr=data1_error, lolims=False, capsize = 3, ls='None', color='#636466', zorder=-1)
    ax.errorbar(data2_position, data2_mean, yerr=data2_error, lolims=False, capsize = 3, ls='None', color='#636466', zorder=-1)

    ax.set_xticks([data1_position, data2_position])
    ax.set_xticklabels(['Before the cue', 'During the cue'])
    
    jitter = 0.15 # Dots dispersion
    
    dispersion_values_data1 = np.random.normal(loc=data1_position, scale=jitter, size=len(data1)).tolist()
    ax.plot(dispersion_values_data1, data1,
            'o',                            
            markerfacecolor=color_contrast,    
            markeredgecolor=color_contrast,
            markeredgewidth=1,
            markersize=5, 
            label='Data1')      
    
    dispersion_values_data2 = np.random.normal(loc=data2_position, scale=jitter, size=len(data2)).tolist()
    ax.plot(dispersion_values_data2, data2,
            'o',                          
            markerfacecolor=color_contrast,    
            markeredgecolor=color_contrast,
            markeredgewidth=1,
            markersize=5, 
            label='Data2')               
    
    if len(data1) == len(data2):
        for x in range(len(data1)):
            ax.plot([dispersion_values_data1[x], dispersion_values_data2[x]], [data1[x], data2[x]], color = '#636466', linestyle='--', linewidth=0.5)
        
    
    plt.ylim(0,100)
    ax.set_xlabel('')
    ax.set_ylabel('Percentage of time doing ' + column, loc='top')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter()) #Add % symbol to the Y axis
    
    plt.title(column.capitalize() + " response in young females", loc = 'left', color='#636466')

    
    # Grey color
    ax.xaxis.label.set_color('#636466')
    ax.yaxis.label.set_color('#636466')
    ax.tick_params(axis='x', colors='#636466')
    ax.tick_params(axis='y', colors='#636466')
    
    # pvalue = pg.ttest(off_list, on_list, paired=True)['p-val'][0]
    
    # def convert_pvalue_to_asterisks(pvalue):
    #     ns = "ns (p=" + str(pvalue)[1:4] + ")"
    #     if pvalue <= 0.0001:
    #         return "****"
    #     elif pvalue <= 0.001:
    #         return "***"
    #     elif pvalue <= 0.01:
    #         return "**"
    #     elif pvalue <= 0.05:
    #         return "*"
    #     return ns

    # y, h, col = max(max(off_list), max(on_list)) + 5, 2, 'k'
    
    # ax.plot([off_position, off_position, on_position, on_position], [y, y+h, y+h, y], lw=1.5, c=col)
    
    # if pvalue > 0.05:
    #     ax.text((off_position+on_position)*.5, y+2*h, convert_pvalue_to_asterisks(pvalue), ha='center', va='bottom', color=col, size=11)
    # elif pvalue <= 0.05:    
    #     ax.text((off_position+on_position)*.5, y, convert_pvalue_to_asterisks(pvalue), ha='center', va='bottom', color=col, size=18)
    
    plt.tight_layout()
    return ax


def discrimination_index(supervised_annotation, directory_output, column='huddle', ax=None):
    
    color_contrast = '#801946'

    if ax is None:
        fig, ax = plt.subplots(figsize=(2.5,4))
    
    sns.set_theme(style="whitegrid")
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    
    data_position = 0
    
    data1 = data_set_to_plot(supervised_annotation, directory_output, ['group'], ['s1'], column, bin_size=60)
    data1 = data1[data1.bin == 3] # The bin starts with 1 (i.e., 1, 2, 3, 4, etc.)
    data1 = data1[column].tolist()

    data2 = data_set_to_plot(supervised_annotation, directory_output, ['group'], ['s1'], column, bin_size=60)
    data2 = data2[data2.bin == 4] # The bin starts with 1 (i.e., 1, 2, 3, 4, etc.)
    data2 = data2[column].tolist()
    
    # data3 = data_set_to_plot(supervised_annotation, directory_output, ['protocol','group'], ['s2','shock'], column, bin_size=60)
    # data3 = data3[data3.bin == 5] # The bin starts with 1 (i.e., 1, 2, 3, 4, etc.)
    # data3 = data3[column].tolist()
    
    # data4 = data_set_to_plot(supervised_annotation, directory_output, ['protocol','group'], ['s2','shock'], column, bin_size=60)
    # data4 = data4[data4.bin == 6] # The bin starts with 1 (i.e., 1, 2, 3, 4, etc.)
    # data4 = data4[column].tolist()
    
    # To calculate the discrimination index (2-3 vs 3-4)
    discrimination = calculate_discrimination(data1, data2)
    
    # To calculate the global discrimination index (2-3 vs 3-4 vs 4-5 vs 5-6)
    # discrimination = calculate_global_discrimination(data1, data2, data3, data4)  
    
    data_mean = np.mean(discrimination)
    data_error = np.std(discrimination, ddof=1)

    ax.hlines(data_mean, xmin=data_position-0.1, xmax=data_position+0.1, color='#636466', linewidth=1.5)
    
    ax.errorbar(data_position, data_mean, yerr=data_error, lolims=False, capsize = 3, ls='None', color='#636466', zorder=-1)

    ax.set_xticks([data_position])
    ax.set_xticklabels([])
    
    jitter = 0.05 # Dots dispersion
    
    dispersion_values_data = np.random.normal(loc=data_position, scale=jitter, size=len(discrimination)).tolist()
    ax.plot(dispersion_values_data, discrimination,
            'o',                            
            markerfacecolor=color_contrast,    
            markeredgecolor=color_contrast,
            markeredgewidth=1,
            markersize=5, 
            label='Data1')      
                     
    plt.ylim(-1,1)
    ax.axhline(y=0, color='#636466', linestyle='--')
    ax.set_xlabel('')
    ax.set_ylabel('Discrimination Index ' + column, loc='top')
    
    # plt.title(column.capitalize() + " DI in TRAP2", loc = 'left', color='#636466')

    # Grey color
    ax.xaxis.label.set_color('#636466')
    ax.yaxis.label.set_color('#636466')
    ax.tick_params(axis='x', colors='#636466')
    ax.tick_params(axis='y', colors='#636466')
        
    plt.tight_layout()
    return ax


def discrimination_index_summary(supervised_annotation, directory_output, column='lookaround', ax=None):
    
    color_contrast = '#801946'

    if ax is None:
        fig, ax = plt.subplots(figsize=(7,2))
    
    sns.set_theme(style="whitegrid")
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
        
    data1 = data_set_to_plot(supervised_annotation, directory_output, ['group'], ['s1'], column, bin_size=60)
    data1 = data1[data1.bin == 3] # The bin starts with 1 (i.e., 1, 2, 3, 4, etc.)
    data1 = data1[column].tolist()

    data2 = data_set_to_plot(supervised_annotation, directory_output, ['group'], ['s1'], column, bin_size=60)
    data2 = data2[data2.bin == 4] # The bin starts with 1 (i.e., 1, 2, 3, 4, etc.)
    data2 = data2[column].tolist()
    
    # data3 = data_set_to_plot(supervised_annotation, directory_output, ['protocol','group'], ['s2','shock'], column, bin_size=60)
    # data3 = data3[data3.bin == 5] # The bin starts with 1 (i.e., 1, 2, 3, 4, etc.)
    # data3 = data3[column].tolist()
    
    # data4 = data_set_to_plot(supervised_annotation, directory_output, ['protocol','group'], ['s2','shock'], column, bin_size=60)
    # data4 = data4[data4.bin == 6] # The bin starts with 1 (i.e., 1, 2, 3, 4, etc.)
    # data4 = data4[column].tolist()
    
    # To calculate the discrimination index (2-3 vs 3-4)
    discrimination = calculate_discrimination(data1, data2)
    
    # To calculate the global discrimination index (2-3 vs 3-4 vs 4-5 vs 5-6)
    # discrimination = calculate_global_discrimination(data1, data2, data3, data4)  
 
    values = count_function(discrimination)
    
    bar_height = 0.3
    categories = [0]
    
    bar1 = ax.barh(categories, np.array(values)[0], bar_height, left=0, align='center', color=color_contrast, label='>0.4')
    bar2 = ax.barh(categories, np.array(values)[1], bar_height, left=np.array(values)[0], align='center', color=color_contrast, label='0.3-0.4')    
    bar3 = ax.barh(categories, np.array(values)[2], bar_height, left=np.array(values)[0] + np.array(values)[1], align='center', color=color_contrast, label='0.2-0.3')    
    bar4 = ax.barh(categories, np.array(values)[3], bar_height, left=np.array(values)[0] + np.array(values)[1] + np.array(values)[2], align='center', color='#636466', label='0.1-0.2')    
    bar5 = ax.barh(categories, np.array(values)[4], bar_height, left=np.array(values)[0] + np.array(values)[1] + np.array(values)[2] + np.array(values)[3], align='center', color='#636466', label='<0.1')            
    
    ax.set_yticks([])  # Hide y-axis ticks
    
    ax.set_xlabel('% of animals', loc='left')
    plt.xlim(0,100)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter()) #Add % symbol to the X axis

    # Move x-axis ticks and label to the top
    # ax.xaxis.tick_top()
    # ax.xaxis.set_label_position('top')
    
    # Remove the frame around the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Grey color
    ax.xaxis.label.set_color('#636466')
    ax.yaxis.label.set_color('#636466')
    ax.tick_params(axis='x', which='both', bottom=True, colors='#636466')
    ax.tick_params(axis='y', colors='#636466')
    
    plt.title('Distribution of Discrimination Index (' + column.capitalize() + ") among young females", loc = 'left', color='#636466')
    
    # Add labels inside the bars
    for bars, label in zip([bar1, bar2, bar3, bar4, bar5], ['DI\nMore than 0.4', 'DI\n0.3 to 0.4', 'DI\n0.2 to 0.3', 'DI\n0.1 to 0.2', 'DI\nLess than 0.1']):
        for bar in bars:
            if bar.get_width() != 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                    label, color='white', ha='center', va='center', fontsize=9)
        
    plt.tight_layout()
    return ax
    
    
# =============================================================================
# Statistics functions
# ============================================================================= 
    

    
    
    
    
    
    
    
    
    
    












