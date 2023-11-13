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
import copy
import numpy as np
import pingouin as pg
import pickle
import statistics
import os

directory_output = "//FOLDER/becell/Lab Projects/ERCstG_HighMemory/Data/Marc/1) SOC/2023-10 - TRAP2/Female/DeepOF/"

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
    
# You may also add the immobility information
def update_supervised_annotation_with_immobility(supervised_annotation):

    directory_path = '//FOLDER/becell/Lab Projects/ERCstG_HighMemory/Data/Marc/1) SOC/2023-09 - Young males/DeepOF/Data/pickles/'
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.pck'):
            file_path = os.path.join(directory_path, filename)            
            with open(file_path, 'rb') as file:
                binary_data = [np.nan] + [int(x) for x in pickle.load(file)['freezing'].tolist()]
                tag = filename.split("DLC")[0]
                
                if tag in supervised_annotation:
                    df = supervised_annotation[tag]
                    df['immobility'] = binary_data                
                    supervised_annotation[tag] = df
                                
                
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













def cohens_d(supervised_annotation, conditions_column='protocol', hue_1='s1', hue_2='s2', values_column='climbing', bin_size=60, duration=360, filter_out=''):

    protocols = [hue_1, hue_2]
    cohens_d_list = []

    for hue in protocols:
        data_1 = csv_generator(supervised_annotation, conditions_column, hue, values_column, 2, bin_size, duration, filter_out)
        data_2 = csv_generator(supervised_annotation, conditions_column, hue, values_column, 3, bin_size, duration, filter_out)
        data_3 = csv_generator(supervised_annotation, conditions_column, hue, values_column, 4, bin_size, duration, filter_out)
        data_4 = csv_generator(supervised_annotation, conditions_column, hue, values_column, 5, bin_size, duration, filter_out)
        
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


def easy_index(supervised_annotation, ax=None, conditions_column='protocol', hue_1='s1', hue_2='s2', values_column='lookaround', bin_size=60, duration=360, filter_out=''):

    if ax is None:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        
    protocols = [hue_1, hue_2]
    protocols_dict = {}
    
    for hue in protocols:
        data_1 = csv_generator(supervised_annotation, conditions_column, hue, values_column, 2, bin_size, duration, filter_out)
        data_2 = csv_generator(supervised_annotation, conditions_column, hue, values_column, 3, bin_size, duration, filter_out)
        data_3 = csv_generator(supervised_annotation, conditions_column, hue, values_column, 4, bin_size, duration, filter_out)
        data_4 = csv_generator(supervised_annotation, conditions_column, hue, values_column, 5, bin_size, duration, filter_out)
        
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
            squared_deviations = [(value[1][i] - value[0][i]) / (value[1][i] + value[0][i]),
                                    (value[1][i] - value[2][i]) / (value[1][i] + value[2][i]),
                                    (value[3][i] - value[2][i]) / (value[3][i] + value[2][i])
                                  ]
            
            # For climbing
            # squared_deviations = [(value[0][i] - value[1][i]) / (value[1][i] + value[0][i]),
            #                         (value[2][i] - value[1][i]) / (value[1][i] + value[2][i]),
            #                         (value[2][i] - value[3][i]) / (value[3][i] + value[2][i])
            #                       ]
                        
            rmsd = statistics.mean(squared_deviations)
            rmsd_list.append(rmsd)
        data_dict[key] = rmsd_list
        
    tags = ['>0.4', '0.3-0.4', '0.2-0.3', '0.1-0.2', '<0.1']
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