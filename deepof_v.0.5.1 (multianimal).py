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

# Define directories
directory_output = '/home/sie/Desktop/marc/dual videos'
directory_dlc = '/home/sie/Desktop/marc/dual videos/h5'
directory_videos = '/home/sie/Desktop/marc/dual videos/mp4'

# Prepare the project
my_deepof_project_raw = deepof.data.Project(
                project_path=os.path.join(directory_output),
                video_path=os.path.join(directory_videos),
                table_path=os.path.join(directory_dlc),
                project_name="deepof_tutorial_project",
                arena="polygonal-manual",
                animal_ids=['colortail','nocolor'],
                table_format=".h5",
                video_format=".mp4",
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
my_deepof_project.load_exp_conditions(directory_output + '/conditions.csv')

# Check conditions
coords = my_deepof_project.get_coords()
print("The original dataset has {} videos".format(len(coords)))
coords = coords.filter_condition({"protocol": "hc_ind"})
print("The filtered dataset has only {} videos".format(len(coords)))

# Perform a supervised analysis
supervised_annotation = my_deepof_project.supervised_annotation()
with open('/home/sie/Desktop/marc/dual videos/supervised_annotation.pkl', 'wb') as file:
    pickle.dump(supervised_annotation, file)
    
# Perform an unsupervised analysis
graph_preprocessed_coords, adj_matrix, to_preprocess, global_scaler = my_deepof_project.get_graph_dataset(
    # animal_id="B", # Comment out for multi-animal embeddings
    center="Center",
    align="Spine_1",
    window_size=25,
    window_step=1,
    test_videos=1,
    preprocess=True,
    scale="standard",
)

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
    interaction_regularization=0.0,
    pretrained=False, # Set to False to train a new model!
)



# =============================================================================
# Load a previously saved project and supervised analysis
my_deepof_project = deepof.data.load_project(directory_output + "/deepof_tutorial_project")
my_deepof_project.load_exp_conditions(directory_output + '/conditions.csv')
with open('/home/sie/Desktop/marc/dual videos/supervised_annotation.pkl', 'rb') as file:
    supervised_annotation = pickle.load(file)

# =============================================================================
# Heatmaps
# =============================================================================

sns.set_context("notebook")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

deepof.visuals.plot_heatmaps(
    my_deepof_project,
    ["colortail_Nose"],
    center="arena",
    exp_condition="protocol",
    condition_value="hc_ee",
    ax=ax1,
    show=False,
    display_arena=True,
    experiment_id="average",
)

deepof.visuals.plot_heatmaps(
    my_deepof_project,
    ["nocolor_Nose"],
    center="arena",
    exp_condition="protocol",
    condition_value="hc_ee",
    ax=ax2,
    show=False,
    display_arena=True,
    experiment_id="average",
)

plt.tight_layout()
plt.show()

# =============================================================================
# Animated skeleton
# =============================================================================

from IPython import display

video = deepof.visuals.animate_skeleton(
    my_deepof_project,
    experiment_id="SOC INT IGM 05092023 HC A1-EE C1",
    frame_limit=500,
    dpi=60,
)

html = display.HTML(video)
display.display(html)
plt.close()

# =============================================================================
# Supervised enrichment
# =============================================================================

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
    bin_index=0,
    bin_size=120,
    ax = fig["A"],
)

deepof.visuals.plot_enrichment(
    my_deepof_project,
    supervised_annotations=supervised_annotation,
    add_stats="Mann-Whitney",
    plot_proportions=False,
    bin_index=0,
    bin_size=120,
    ax = fig["B"],
)

for ax in fig:
    fig[ax].set_xticklabels(fig[ax].get_xticklabels(), rotation=45, ha='right')
    fig[ax].set_title("")
    fig[ax].set_xlabel("")

fig["A"].get_legend().remove()

plt.tight_layout()
plt.show()

# =============================================================================
# PCA embedding
# =============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

deepof.visuals.plot_embeddings(
    my_deepof_project,
    supervised_annotations=supervised_annotation,
    ax=ax1,
)
deepof.visuals.plot_embeddings(
    my_deepof_project,
    supervised_annotations=supervised_annotation,
    bin_size=120,
    bin_index=0,
    ax=ax2,
)

ax1.set_title("supervised embeddings of full videos")
ax2.set_title("supervised embeddings of first two minutes")

plt.tight_layout()
plt.show()







