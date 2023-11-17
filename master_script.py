"""
@author: mcanela
Based on: https://deepof.readthedocs.io/en/latest/tutorial_notebooks/deepof_preprocessing_tutorial.html
"""

# =============================================================================
# Importing packages and directories
# =============================================================================

import os
import copy
import pickle
import deepof.data
import pandas as pd
import numpy as np

# =============================================================================
# Creating a new DeepOF project
# =============================================================================

def creating_deepof_project(directory_output, directory_dlc, directory_videos, manual, scale):
    if manual == False:
        arena = 'polygonal-autodetect'
    elif manual == True:
        arena = 'polygonal-manual'
    my_deepof_project = deepof.data.Project(
                        project_path=os.path.join(directory_output),
                        video_path=os.path.join(directory_videos),
                        table_path=os.path.join(directory_dlc),
                        project_name="deepof_tutorial_project",
                        arena=arena,
                        # animal_ids=["Animal_1", 'Animal_2'],
                        video_format=".avi",
                        # exclude_bodyparts=["Tail_1", "Tail_2", "Tail_tip"],
                        video_scale=scale,  # in mm
                        enable_iterative_imputation=10,
                        smooth_alpha=1,
                        exp_conditions=None,
                        )
    
    # Creating a DeepOF project:
    my_deepof_project = my_deepof_project.create(force=True)
    
    return my_deepof_project

# =============================================================================
# Uploading conditions to my DeepOF project
# =============================================================================

# Check if everything is correct
def check_conditions(directory_output, my_deepof_project, col_name):
    conditions = np.unique(pd.read_csv(directory_output + "conditions.csv", index_col=0)[col_name].to_list())
    coords = my_deepof_project.get_coords()
    print("The dataset has {} videos:".format(len(coords)))
    for condition in conditions:
        coords_filter = coords.filter_condition({col_name: condition})
        print("\t{} videos corresponding to ".format(len(coords_filter)) + condition)    


# =============================================================================
# Supervised analysis
# =============================================================================

def update_supervised_annotation_with_immobility(supervised_annotation, directory_path):
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
# Unsupervised analysis
# =============================================================================

# This code will generate a dataset using graph representations, as well a some auxiliary objects
def graph_dataset_function(my_deepof_project):
    graph_preprocessed_coords, adj_matrix, to_preprocess, global_scaler = my_deepof_project.get_graph_dataset(
        # animal_id="S1", # Comment out for multi-animal embeddings
        center="Center",
        align="Spine_1",
        window_size=25,
        window_step=1,
        test_videos=1,
        preprocess=True,
        scale="standard",
    )

    return graph_preprocessed_coords, adj_matrix, to_preprocess, global_scaler


def train_model_function(my_deepof_project, graph_preprocessed_coords, adj_matrix, pre_trained):
    trained_model = my_deepof_project.deep_unsupervised_embedding(
        preprocessed_object=graph_preprocessed_coords, # Change to preprocessed_coords to use non-graph embeddings
        adjacency_matrix=adj_matrix,
        embedding_model="VaDE", # Can also be set to 'VQVAE' and 'Contrastive'
        epochs=10,
        encoder_type="recurrent", # Can also be set to 'TCN' and 'transformer'
        n_components=10,
        latent_dim=4,
        batch_size=1024,
        verbose=False, # Set to True to follow the training loop
        interaction_regularization=0.0, # Set to 0.5 if multi-animal training
        pretrained=pre_trained, # Set to False to train a new model!
    )

    return trained_model

















