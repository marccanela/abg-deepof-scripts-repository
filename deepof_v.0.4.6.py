"""
DeepOF v.0.4.6
"""

import os
import pandas as pd
import pickle
import deepof.data

import deepof.visuals
import matplotlib.pyplot as plt
import seaborn as sns

directory_output = '//folder/becell/Lab Projects/ERCstG_HighMemory/Data/Marc/1) SOC/2023-07a08 SOC Controls_males/data_controls_dlc/'
directory_dlc = '//folder/becell/Lab Projects/ERCstG_HighMemory/Data/Marc/1) SOC/2023-07a08 SOC Controls_males/data_controls_dlc/h5/'
directory_videos = '//folder/becell/Lab Projects/ERCstG_HighMemory/Data/Marc/1) SOC/2023-07a08 SOC Controls_males/data_controls_dlc/mp4/'

my_deepof_project_raw = deepof.data.Project(
                project_path=os.path.join(directory_output),
                video_path=os.path.join(directory_videos),
                table_path=os.path.join(directory_dlc),
                project_name="deepof_tutorial_project",
                arena="polygonal-manual",
                animal_ids=["individual1"],
                table_format=".h5",
                video_format=".mp4",
                # exclude_bodyparts=["Tail_1", "Tail_2", "Tail_tip"],
                video_scale=200,
                smooth_alpha=1,
                exp_conditions=None,
)

my_deepof_project = my_deepof_project_raw.create(force=True)
