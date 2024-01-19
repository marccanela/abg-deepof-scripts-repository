"""
Version number 1: 28/07/2023
@author: mcanela
DeepOF ANALYSIS - PART 2
Based on: https://deepof.readthedocs.io/en/latest/tutorial_notebooks/deepof_preprocessing_tutorial.html
"""

# =============================================================================
# Importing packages and directories
# =============================================================================

import os
import pandas as pd
import deepof.data
import numpy as np
import deepof.visuals
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import display
from networkx import Graph, draw

# Modify the following directories as necessary
directory_output = '//FOLDER/becell/Lab Projects/ERCstG_HighMemory/Data/Marc/1) SOC/2023-08 SOC - Females/DeepOF/'
directory_dlc = '//FOLDER/becell/Lab Projects/ERCstG_HighMemory/Data/Marc/1) SOC/2023-08 SOC - Females/DeepOF/Data/dlc/'
directory_videos = '//FOLDER/becell/Lab Projects/ERCstG_HighMemory/Data/Marc/1) SOC/2023-08 SOC - Females/DeepOF/Data/Corrected videos/'

# =============================================================================
# Loading a previously generated project
# =============================================================================

my_deepof_project = deepof.data.load_project(directory_output + "deepof_tutorial_project")
# Uploading the experimental conditions to the DeepOF project
my_deepof_project.load_exp_conditions(directory_output + "conditions.csv")

# =============================================================================
# Creating a heatmap comparing conditions
# =============================================================================

def heatmap(my_deepof_project, specific='s2'):
    sns.set_context("notebook")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    deepof.visuals.plot_heatmaps(
        my_deepof_project,
        ["Center"], # Body part to plot
        center="arena",
        exp_condition="protocol", # Name of the experimental condition
        condition_value=specific, # Specific experimental condition to plot
        ax=ax1,
        bin_index=2, # Bin number to select
        bin_size=60, # Size of the bin in seconds
        show=False,
        display_arena=True,
        experiment_id="average",
    )
    
    deepof.visuals.plot_heatmaps(
        my_deepof_project,
        ["Center"], # Body part to plot
        center="arena",
        exp_condition="protocol", # Name of the experimental condition
        condition_value=specific, # Specific experimental condition to plot
        ax=ax2,
        bin_index=3, # Bin number to select
        bin_size=60, # Size of the bin in seconds
        show=False,
        display_arena=True,
        experiment_id="average",
    )
    
    ax1.title.set_text('OFF period (120-180 s)')
    ax2.title.set_text('ON period (180-240 s)')
    plt.tight_layout()
    
    return plt
    
# =============================================================================
# Creating a video render
# =============================================================================

def video_render(my_deepof_project):
    video = deepof.visuals.animate_skeleton(
        my_deepof_project,
        experiment_id="20230530_Marc_ERC SOC Hab_Males_box bc_01_01_1", # Video to plot
        frame_limit=500, # Number of frames to plot
        dpi=60,
    )

    html = display.HTML(video)
    display.display(html)
    plt.close()
    
    return plt
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    