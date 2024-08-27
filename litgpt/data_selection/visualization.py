import matplotlib.pyplot as plt
from litgpt.utils import sample_from_simplex
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.patches import Polygon
import os
import torch
import numpy as np

def visualize_predicted_rewards(gp_model, n_points: int = 100, output_dir: str = "./reward_outputs", iters_passed: int = 0, dim=2, device="cuda") -> None:
    """Visualize the predicted rewards for a given GP model.

    Args:
        gp_model: The GP model to visualize.
        n_points: The number of points to sample.
        output_dir: The directory to save the plots.
    """
    # Sample points from the GP model
    plt.gcf().clear()
    samples = sample_from_simplex(n_points, dim).to(device=device)

    with torch.no_grad():
        predictions = gp_model(samples)
        mean = predictions.mean.cpu().numpy().reshape(samples.shape[0])
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    
    # TODO: support higher dimensions, we need to make a polygon instead
    coords = samples.cpu().numpy()
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=mean, cmap=cm.viridis)
    
    # Customize the plot
    ax.set_xlabel('Fineweb Weight')
    ax.set_ylabel('SFT Weight')
    ax.set_title(f'Predicted Reward Landscape (Iteration {iters_passed})')
    
    # Add a color bar
    cbar = fig.colorbar(scatter, shrink=0.5, aspect=5)
    cbar.set_label("Predicted reward")
    #mkdir -p
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/reward_iters_{iters_passed}.png")
    print("Saved plot to", f"{output_dir}/reward_iters_{iters_passed}.png")

def visualize_predicted_rewards_polygon(gp_model, n_points: int = 100, output_dir: str = "./reward_outputs", iters_passed: int = 0, dim=2, device="cuda") -> None:
    """
    Visualize the predicted rewards for a given GP model on a simplex.
    Args:
    gp_model: The GP model to visualize.
    n_points: The number of points to sample.
    output_dir: The directory to save the plots.
    dim: Number of dimensions (datasets)
    """
    plt.gcf().clear()
    samples = sample_from_simplex(n_points, dim).to(device=device)
    
    with torch.no_grad():
        predictions = gp_model(samples)
        mean = predictions.mean.cpu().numpy().reshape(samples.shape[0])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    coords = samples.cpu().numpy()
    
    if dim == 2:
        # For 2D, plot on a line
        ax.scatter(coords[:, 0], np.zeros_like(coords[:, 0]), c=mean, cmap=cm.viridis)
        ax.set_ylim(-0.1, 0.1)
        ax.set_xlabel('Dataset 1 Weight')
        ax.set_yticks([])
        ax.annotate('Dataset 1 (1,0)', xy=(1, 0), xytext=(1.01, 0.01))
        ax.annotate('Dataset 2 (0,1)', xy=(0, 0), xytext=(-0.01, 0.01))
    else:
        # For higher dimensions, plot on an n-gon
        theta = np.linspace(0, 2*np.pi, dim, endpoint=False)
        corners = np.column_stack([np.cos(theta), np.sin(theta)])
        
        # Plot the n-gon
        polygon = Polygon(corners, fill=False, edgecolor='black')
        ax.add_patch(polygon)
        
        # Project points onto 2D
        projected_coords = coords.dot(corners)
        
        scatter = ax.scatter(projected_coords[:, 0], projected_coords[:, 1], c=mean, cmap=cm.viridis)
        
        # Label the corners
        for i, corner in enumerate(corners):
            ax.annotate(f'Dataset {i+1}', xy=corner, xytext=corner*1.1)
    
    ax.set_aspect('equal')
    ax.set_title(f'Predicted Reward Landscape (Iteration {iters_passed})')
    
    # Add a color bar
    cbar = fig.colorbar(scatter if dim > 2 else ax.collections[0], shrink=0.5, aspect=5)
    cbar.set_label("Predicted reward")
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/reward_iters_{iters_passed}.png")
    print("Saved plot to", f"{output_dir}/reward_iters_{iters_passed}.png")
    plt.close(fig)
