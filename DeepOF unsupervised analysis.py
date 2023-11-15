"""
Version number 1: 24/07/2023
@author: mcanela
DeepOF UNSUPERVISED ANALYSIS
https://deepof.readthedocs.io/en/latest/tutorial_notebooks/deepof_unsupervised_tutorial.html
"""

import copy
import os
import numpy as np
import pickle
import deepof.data
from IPython import display
from networkx import Graph, draw
import deepof.visuals
import matplotlib.pyplot as plt
import seaborn as sns



# =============================================================================
# Embedding our data with deep clustering models
# =============================================================================


# Get embeddings, soft_counts, and breaks per video
embeddings, soft_counts, breaks = deepof.model_utils.embedding_per_video(
    coordinates=my_deepof_project,
    to_preprocess=to_preprocess,
    model=trained_model,
    # animal_id="S1",
    global_scaler=global_scaler,
)

# =============================================================================
# Visualizing temporal and global embeddings
# =============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

deepof.visuals.plot_embeddings(
    my_deepof_project,
    embeddings,
    soft_counts,
    breaks,
    aggregate_experiments=False,
    samples=100,
    ax=ax1,
    bin_index=0, # Bin number to select
    bin_size=120, # Size of the bin in seconds
    save=False, # Set to True, or give a custom name, to save the plot
)

deepof.visuals.plot_embeddings(
    my_deepof_project,
    embeddings,
    soft_counts,
    breaks,
    aggregate_experiments="time on cluster", # Can also be set to 'mean' and 'median'
    exp_condition="group",
    show_aggregated_density=False,
    ax=ax2,
    bin_index=0, # Bin number to select
    bin_size=120, # Size of the bin in seconds
    save=False, # Set to True, or give a custom name, to save the plot,
)
ax2.legend(
    bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0
)

plt.tight_layout()
plt.show()

'''
The figure on the left now shows time points in a UMAP projection of the latent space,
where colors indicate different clusters. The figure on the right aggregates all time
points in a given animal as a vector of counts per behavior (indicating how much time each
animal spends on each cluster). We may see a clear separation between conditions, 
in a fully unsupervised way!
'''

# =============================================================================
# Generating Gantt charts with all clusters
# =============================================================================

fig = plt.figure(figsize=(12, 6))

deepof.visuals.plot_gantt(
    my_deepof_project,
    soft_counts=soft_counts,
    experiment_id='20230530_Marc_ERC SOC Hab_Males_box ab_01_01_1',
)

# =============================================================================
# Global separation dynamics
# =============================================================================

fig, ax = plt.subplots(1, 1, figsize=(12, 4))

deepof.visuals.plot_distance_between_conditions(
    my_deepof_project,
    embeddings,
    soft_counts,
    breaks,
    "group",
    distance_metric="wasserstein",
    n_jobs=1,
)

plt.show()























