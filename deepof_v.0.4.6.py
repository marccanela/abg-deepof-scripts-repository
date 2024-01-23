"""
DeepOF v.0.4.6 (To try also in v.0.5.1)
conda activate deepof2
"""

import os
import pandas as pd
import pickle
import deepof.data

import deepof.visuals
import matplotlib.pyplot as plt
import seaborn as sns

# Convert CSV-multi-animal into CSV-single-animal
directory_csv = '//folder/becell/Lab Projects/ERCstG_HighMemory/Data/Marc/1) SOC/2023-07a08 SOC Controls_males/data_controls_dlc/csv_converted/'
for file in os.listdir(directory_csv):
    if file.endswith('.csv'):
        file_path = os.path.join(directory_csv, file)
        df = pd.read_csv(file_path, header=None)
        df_modified = df.drop(1)
        df_modified.to_csv(file_path[:-7] + '.csv', index=False, header=False)
        
directory_output = '//folder/becell/Lab Projects/ERCstG_HighMemory/Data/Marc/1) SOC/2023-07a08 SOC Controls_males/data_controls_dlc/'
directory_dlc = '//folder/becell/Lab Projects/ERCstG_HighMemory/Data/Marc/1) SOC/2023-07a08 SOC Controls_males/data_controls_dlc/csv_converted/'
directory_videos = '//folder/becell/Lab Projects/ERCstG_HighMemory/Data/Marc/1) SOC/2023-07a08 SOC Controls_males/data_controls_dlc/avi/'

my_deepof_project_raw = deepof.data.Project(
                project_path=os.path.join(directory_output),
                video_path=os.path.join(directory_videos),
                table_path=os.path.join(directory_dlc),
                project_name="deepof_tutorial_project",
                arena="polygonal-manual",
                # animal_ids=['A'],
                table_format=".csv",
                video_format=".avi",
                bodypart_graph='deepof_14',
                # exclude_bodyparts=["Tail_1", "Tail_2", "Tail_tip"],
                video_scale=200,
                smooth_alpha=1,
                exp_conditions=None,
)

my_deepof_project = my_deepof_project_raw.create(force=True)
