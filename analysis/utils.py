import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans


DATASET_INFO = {
    "mll23": {
        "path": "dataset/mll23",
        "split": "train",
    },
    "matek": {
        "path": "dataset/PKG - AML-Cytomorphology_LMU",
        "split": "train",
    },
    "acevedo": {
        "path": "dataset/acevedo",
        "split": "train",
    },
    "hehr": {
        "path": "dataset/PKG - AML-Cytomorphology_MLL_Helmholtz_v1",
        "split": "train",
    },
    "bmc": {
        "path": "dataset/bmc", 
        "split": "train",
    },
}

def calculate_entropy(top_val, top_label, ignore_label_idx:int = None, eps=1e-9):
    dict_size = top_label.shape[0]
    entropy = torch.zeros(dict_size)

    for i in range(dict_size):
        unique_labels, counts = top_label[i].unique(return_counts=True)
        if ignore_label_idx is not None:
            counts = counts[unique_labels != ignore_label_idx]
            unique_labels = unique_labels[unique_labels != ignore_label_idx]  
        if len(unique_labels) != 0:
            if counts.sum().item() < 10:
                entropy[i] = -1  # discount as too few datapoints!
            else:
                summed_probs = torch.zeros_like(unique_labels, dtype=top_val.dtype)
                for j, label in enumerate(unique_labels):
                    summed_probs[j] = top_val[i][top_label[i] == label].sum().item()
                summed_probs = (summed_probs / summed_probs.sum()) 
                entropy[i] = -torch.sum(summed_probs * torch.log(summed_probs + eps)) 
        else:
            entropy[i] = -1
    return entropy
    
def load_stats(save_directory: str, device: torch.device):
    mean_acts = torch.load(os.path.join(save_directory, "sae_mean_acts.pt"),map_location=device)
    sparsity = torch.load(os.path.join(save_directory, "sae_sparsity.pt"),map_location=device)
    top_val = torch.load(os.path.join(save_directory, "max_activating_image_values.pt"),map_location=device)
    top_idx = torch.load(os.path.join(save_directory, "max_activating_image_indices.pt"),map_location=device)
    top_label = torch.load(os.path.join(save_directory, "max_activating_image_label_indices.pt"),map_location=device)
    try:
        top_entropy = torch.load(os.path.join(save_directory, "top_entropy.pt")) 
    except:
        print("Calculating top entropy")
        top_entropy = calculate_entropy(top_val, top_label)
        torch.save(top_entropy, os.path.join(save_directory, "top_entropy.pt"))

    print(f"Stats loaded from {save_directory}")
    
    stats = {
        "mean_acts": mean_acts.to(device),
        "sparsity": sparsity.to(device),
        "top_val": top_val.to(device),
        "top_idx": top_idx.to(device).to(torch.int64),
        "top_label": top_label.to(device).to(torch.int64),
        "top_entropy": top_entropy.to(device)
    }
    return stats

def get_stats_scatter_plot(stats, mask=None, save_directory:str=None, eps=1e-9, prefix = ""):

    if mask is None:
        mask = torch.ones_like(stats["sparsity"], dtype=torch.bool, device=stats["sparsity"].device)

    indices = torch.where(mask)[0]
    plotting_data = torch.stack(
        [
            torch.log10(stats["sparsity"][mask] + eps),
            torch.log10(stats["mean_acts"][mask] + eps),
            stats['top_entropy'][mask],
            indices
        ],
        dim=0
    )
    plotting_data = plotting_data.transpose(0, 1)

    x_label = "log10(sparsity)"
    y_label = "log10(mean_acts)"
    color_label = "entropy"
    hover_label = "index"

    df = pd.DataFrame(plotting_data.numpy(), columns=[x_label, y_label, color_label, hover_label])
    fig = px.scatter(df, x=x_label, y=y_label, color=color_label, marginal_x="histogram", marginal_y="histogram", opacity=0.5, hover_data=[hover_label])

    if save_directory is not None:
        fig.write_image(os.path.join(save_directory, f"{prefix}_scatter_plot.png"))
    fig.show()

def random_select_from_cluster(stats, n_clusters = 10, n_each_sample = 5, save_directory:str=None, eps=1e-9):
    data = np.stack([stats['mean_acts'], stats['sparsity']], axis=1)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42) 
    labels = kmeans.fit_predict(data)

    selected_indices = []
    for cluster_id in np.unique(labels):
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) >= n_each_sample:
            selected = np.random.choice(cluster_indices, n_each_sample, replace=False)
        else:
            selected = cluster_indices 
        selected_indices.extend(selected)

    selected_mask = torch.zeros_like(stats['mean_acts'], dtype=torch.bool, device=stats["sparsity"].device)
    selected_mask[selected_indices] = True
    get_stats_scatter_plot(stats, mask=selected_mask, save_directory=save_directory, prefix='selected')

    return selected_indices

import numpy as np
import torch
from sklearn.cluster import KMeans

def random_select_threshold_from_cluster(
    stats, 
    n_clusters=10, 
    n_each_sample=5, 
    seed=42,
    mean_acts_log_threshold=None, 
    sparsity_log_threshold=None, 
    save_directory: str = None, 
    eps=1e-9  # Small value to prevent log(0) errors
):
    # Extract data
    mean_acts = stats['mean_acts']
    sparsity = stats['sparsity']

    # Convert log10 thresholds to linear scale if provided
    if mean_acts_log_threshold is not None:
        mean_acts_threshold = 10 ** mean_acts_log_threshold
    else:
        mean_acts_threshold = None

    if sparsity_log_threshold is not None:
        sparsity_threshold = 10 ** sparsity_log_threshold
    else:
        sparsity_threshold = None

    # Apply thresholds if provided
    if mean_acts_threshold is not None:
        mean_mask = mean_acts >= mean_acts_threshold
    else:
        mean_mask = torch.ones_like(mean_acts, dtype=torch.bool)

    if sparsity_threshold is not None:
        sparsity_mask = sparsity <= sparsity_threshold  # Assuming lower sparsity is better
    else:
        sparsity_mask = torch.ones_like(sparsity, dtype=torch.bool)

    # Combine masks
    final_mask = mean_mask & sparsity_mask

    # Filter data based on mask
    selected_indices = torch.where(final_mask)[0].cpu().numpy()
    
    if len(selected_indices) == 0:
        raise ValueError("No data points meet the threshold criteria. Try adjusting thresholds.")

    filtered_data = np.stack([mean_acts[selected_indices].cpu().numpy(), 
                              sparsity[selected_indices].cpu().numpy()], axis=1)

    # Apply KMeans clustering on filtered data
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    labels = kmeans.fit_predict(filtered_data)

    # Select samples from each cluster
    final_selected_indices = []
    for cluster_id in np.unique(labels):
        cluster_indices = np.where(labels == cluster_id)[0]
        actual_indices = selected_indices[cluster_indices]  # Convert to original indices

        if len(actual_indices) >= n_each_sample:
            selected = np.random.choice(actual_indices, n_each_sample, replace=False)
        else:
            selected = actual_indices  # Take all if not enough samples

        final_selected_indices.extend(selected)

    # Create mask for selected indices
    selected_mask = torch.zeros_like(mean_acts, dtype=torch.bool, device=mean_acts.device)
    selected_mask[final_selected_indices] = True

    # Generate scatter plot
    get_stats_scatter_plot(stats, mask=selected_mask, save_directory=save_directory, prefix='selected')

    return final_selected_indices



def plot_ref_images(stats, dataset, latent_idx: int, save_directory:str=None, plot_top_k: int=10, eps=1e-9):
    resize_size = 224
    num_cols = 5
    num_rows = plot_top_k // num_cols

    images = []
    labels = []

    for i, idx in enumerate(stats['top_idx'][latent_idx][:plot_top_k]):
        img = dataset[idx.item()]['image']
        images.append(img.resize((resize_size, resize_size)))
        assert dataset[idx.item()]['label'] == stats['top_label'][latent_idx][i], "label mismatch, try matching dataset shuffle seed"
        labels.append(dataset[idx.item()]['label'])
        
    _, axes = plt.subplots(num_rows, num_cols, figsize=(4.5 * num_cols, 6 * num_rows))

    for i, (image, label) in enumerate(zip(images, labels)):
        ax = axes[i//num_cols, i%num_cols]
        ax.imshow(image)
        top_val_i = torch.log10(stats['top_val'][latent_idx][i] + eps).item()
        ax.set_title(f"{label}(a: {top_val_i:.2f})",fontsize=35)
        ax.axis('off')

    mean_acts = torch.log10(stats["mean_acts"][latent_idx] + eps).item()
    sparsity = torch.log10(stats["sparsity"][latent_idx] + eps).item()
    plt.suptitle(
        f"Index {latent_idx} (f: {sparsity:.2f}, a: {mean_acts:.2f}, e: {stats['top_entropy'][latent_idx]:.2f})\n",
        fontsize=35)

    plt.tight_layout()
    if save_directory is not None:
        plt.savefig(os.path.join(save_directory, f"ref_images{latent_idx}.png"))
    plt.show()
    plt.close()