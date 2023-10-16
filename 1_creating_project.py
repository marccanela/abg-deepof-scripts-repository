"""
@author: mcanela
DeepOF ANALYSIS - PART 1
Based on: https://deepof.readthedocs.io/en/latest/tutorial_notebooks/deepof_preprocessing_tutorial.html
"""

# =============================================================================
# Importing packages and directories
# =============================================================================

import os
import pandas as pd
import deepof.data
import numpy as np

# Modify the following directories as necessary
directory_output = '//FOLDER/becell/Lab Projects/ERCstG_HighMemory/Data/Marc/1) SOC/2023-09 - Young males/DeepOF/'
directory_dlc = '//FOLDER/becell/Lab Projects/ERCstG_HighMemory/Data/Marc/1) SOC/2023-09 - Young males/DeepOF/dlc/'
directory_videos = '//FOLDER/becell/Lab Projects/ERCstG_HighMemory/Data/Marc/1) SOC/2023-09 - Young males/DeepOF/corrected videos/'

# NOTE: Run the corrected videos with the DLC files of the corrected videos

# =============================================================================
# Creating a new DeepOF project
# =============================================================================

def creating_deepof_project(directory_output, directory_dlc, directory_videos):
    my_deepof_project = deepof.data.Project(
                        project_path=os.path.join(directory_output),
                        video_path=os.path.join(directory_videos),
                        table_path=os.path.join(directory_dlc),
                        project_name="deepof_tutorial_project",
                        arena="polygonal-manual",
                        # animal_ids=["Animal_1", 'Animal_2'],
                        video_format=".avi",
                        # exclude_bodyparts=["Tail_1", "Tail_2", "Tail_tip"],
                        video_scale=200,  # in mm
                        enable_iterative_imputation=10,
                        smooth_alpha=1,
                        exp_conditions=None,
                        )
    
    # Creating a DeepOF project:
    my_deepof_project = my_deepof_project.create(force=True)
    
    return my_deepof_project

my_deepof_project = creating_deepof_project(directory_output, directory_dlc, directory_videos)

# OPTIONAL: Edit the arena size of some given videos after the project is already created
my_deepof_project.edit_arenas(videos=['20230728_Marc_ERC SOC S1_Males_box ef_05_01_1',
                                      '20230728_Marc_ERC SOC S2_Males_box ab_04_01_1',
                                      '20230728_Marc_ERC SOC S2_Males_box ab_06_01_1',
                                      '20230728_Marc_ERC SOC S2_Males_box cd_04_01_1'
                                      ])

# =============================================================================
# Uploading conditions to my DeepOF project
# =============================================================================

# Uploading the experimental conditions to the DeepOF project
my_deepof_project.load_exp_conditions(directory_output + "conditions.csv")

# Check if everything is correct
conditions = np.unique(pd.read_csv(directory_output + "conditions.csv", index_col=0)['protocol'].to_list())
coords = my_deepof_project.get_coords()
print("The dataset has {} videos:".format(len(coords)))
for condition in conditions:
    coords_filter = coords.filter_condition({"protocol": condition})
    print("\t{} videos corresponding to ".format(len(coords_filter)) + condition)    



























