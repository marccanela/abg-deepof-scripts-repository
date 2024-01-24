"""
SINGLE ANIMAL
DeepOF v.0.5.1
conda activate deepof2
Analyze with DLC multia_bed_14_keyp-mcanela-2023-11-16
"""

import os
import pandas as pd
import pickle
import deepof.data

import deepof.visuals
import matplotlib.pyplot as plt
import seaborn as sns

# Convert CSV-multi-animal into CSV-single-animal
directory_csv = '/home/sie/Desktop/marc/data_controls_dlc/csv_converted'
for file in os.listdir(directory_csv):
    if file.endswith('.csv'):
        file_path = os.path.join(directory_csv, file)
        df = pd.read_csv(file_path, header=None)
        df_modified = df.drop(1)
        df_modified.to_csv(file_path[:-7] + '.csv', index=False, header=False)

# Define directories
directory_output = '/home/sie/Desktop/marc/data_controls_dlc'
directory_dlc = '/home/sie/Desktop/marc/data_controls_dlc/csv_converted'
directory_videos = '/home/sie/Desktop/marc/data_controls_dlc/avi'

# Prepare the project
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

# Create the project
my_deepof_project = my_deepof_project_raw.create(force=True)

# Edit wrong arenas
my_deepof_project.edit_arenas(
    videos=['20240119_Marc_ERC SOC light_Males_box de_06_01_1'],
    arena_type="polygonal-manual",
)

# Load conditions
my_deepof_project.load_exp_conditions('/home/sie/Desktop/marc/data_controls_dlc/conditions.csv')

# Check conditions
coords = my_deepof_project.get_coords()
print("The original dataset has {} videos".format(len(coords)))
coords = coords.filter_condition({"protocol": "s2"})
print("The filtered dataset has only {} videos".format(len(coords)))

# Load a previously saved project
my_deepof_project = deepof.data.load_project(directory_output + "/tutorial_project")

# Perform a supervised analysis
supervised_annotation = my_deepof_project.supervised_annotation()
with open(directory_output + 'supervised_annotation.pkl', 'wb') as file:
    pickle.dump(supervised_annotation, file)



























