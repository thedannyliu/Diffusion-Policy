"""
PushT Visualization Utilities for Trajectory Plotting.

This module provides functions to create publication-quality trajectory plots
similar to Figure 5 in the Diffusion Policy paper, showing multiple rollout
trajectories overlaid on the PushT environment with color-coded time steps.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon, Circle
from typing import List, Optional, Tuple
import os


def create_pusht_background(ax, goal_pose: np.ndarray = None, 
                            show_agent_start: bool = False,
                            window_size: int = 512):
    """
    Draw the PushT environment background with the T-shaped goal region.
    
    Args:
        ax: Matplotlib axis
        goal_pose: [x, y, theta] of the goal T position. If None, use default.
        show_agent_start: Whether to show the default agent start position
        window_size: Size of the environment (default 512)
    """
    # Default goal pose (center of workspace, no rotation)
    if goal_pose is None:
        goal_pose = np.array([256, 256, 0])
    
    # Create T-shape vertices (relative to center, then transform)
    # The T-block dimensions: 50x100 total, 50x50 top bar, 25x50 bottom bar
    # Top bar: width=100, height=25
    # Stem: width=25, height=50 (extends from bottom of top bar)
    t_vertices_local = np.array([
        [-50, -25],   # top-left of top bar
        [50, -25],    # top-right of top bar
        [50, 0],      # bottom-right of top bar
        [12.5, 0],    # right edge of stem top
        [12.5, 50],   # bottom-right of stem
        [-12.5, 50],  # bottom-left of stem
        [-12.5, 0],   # left edge of stem top
        [-50, 0],     # bottom-left of top bar
    ])
    
    # Apply rotation and translation
    theta = goal_pose[2]
    rot_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    t_vertices = t_vertices_local @ rot_matrix.T + goal_pose[:2]
    
    # Draw goal region (light green, semi-transparent)
    goal_polygon = Polygon(t_vertices, 
                          facecolor='lightgreen', 
                          edgecolor='green',
                          alpha=0.5, 
                          linewidth=2,
                          zorder=1)
    ax.add_patch(goal_polygon)
    
    # Draw walls
    ax.plot([5, 5], [5, 507], 'k-', linewidth=2, zorder=0)
    ax.plot([5, 507], [5, 5], 'k-', linewidth=2, zorder=0)
    ax.plot([507, 507], [5, 507], 'k-', linewidth=2, zorder=0)
    ax.plot([5, 507], [507, 507], 'k-', linewidth=2, zorder=0)
    
    # Set axis limits and aspect
    ax.set_xlim(0, window_size)
    ax.set_ylim(0, window_size)
    ax.set_aspect('equal')
    ax.set_facecolor('white')
    
    # Show agent start position marker
    if show_agent_start:
        agent_start = Circle((256, 400), radius=15, 
                            facecolor='lightblue', 
                            edgecolor='blue',
                            alpha=0.5,
                            zorder=1)
        ax.add_patch(agent_start)


def plot_trajectory_heatmap(ax, trajectories: List[np.ndarray], 
                           cmap_name: str = 'plasma',
                           linewidth: float = 1.5,
                           alpha: float = 0.7,
                           label: str = None):
    """
    Plot multiple trajectories with time-based color gradient (heatmap style).
    
    Args:
        ax: Matplotlib axis
        trajectories: List of trajectory arrays, each [T, 2] for (x, y) positions
        cmap_name: Colormap name for time progression
        linewidth: Width of trajectory lines
        alpha: Transparency of lines
        label: Label for legend
    """
    cmap = cm.get_cmap(cmap_name)
    
    for traj in trajectories:
        if traj is None or not isinstance(traj, np.ndarray):
            continue
        if traj.ndim < 2 or traj.shape[0] < 2:
            continue
            
        # Extract x, y (handle both [T, 2] and [T, 3] shapes)
        x = traj[:, 0]
        y = traj[:, 1]
        
        # Create line segments for colored line
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Color by normalized time
        t = np.linspace(0, 1, len(segments))
        
        # Create LineCollection with color gradient
        lc = LineCollection(segments, cmap=cmap, alpha=alpha, linewidth=linewidth)
        lc.set_array(t)
        ax.add_collection(lc)
    
    # Add a single point for legend if label provided
    if label and len(trajectories) > 0:
        ax.plot([], [], '-', color=cmap(0.5), linewidth=linewidth, label=label)


def plot_agent_positions(ax, trajectories: List[np.ndarray],
                        marker_color: str = 'blue',
                        marker_size: float = 3,
                        alpha: float = 0.3):
    """
    Plot agent positions as scatter points with time-based color gradient.
    
    Args:
        ax: Matplotlib axis  
        trajectories: List of trajectory arrays, each [T, 2] for (x, y)
        marker_color: Color for markers
        marker_size: Size of markers
        alpha: Transparency
    """
    cmap = cm.get_cmap('YlOrRd')  # Yellow to Red for agent
    
    for traj in trajectories:
        if traj is None or not isinstance(traj, np.ndarray):
            continue
        if traj.ndim < 2 or traj.shape[0] < 2:
            continue
            
        x = traj[:, 0]
        y = traj[:, 1]
        t = np.linspace(0, 1, len(x))
        
        ax.scatter(x, y, c=t, cmap=cmap, s=marker_size, alpha=alpha, zorder=3)


def create_pusht_trajectory_plot(
    block_trajectories: List[np.ndarray],
    agent_trajectories: List[np.ndarray] = None,
    goal_pose: np.ndarray = None,
    title: str = 'PushT Trajectories',
    coverage_values: List[float] = None,
    figsize: Tuple[int, int] = (6, 6),
    save_path: str = None,
    show_colorbar: bool = True,
    dpi: int = 150
) -> plt.Figure:
    """
    Create a publication-quality trajectory visualization for PushT.
    
    This creates a plot similar to Figure 5 in the Diffusion Policy paper,
    showing multiple rollout trajectories overlaid with color gradients
    indicating time progression.
    
    Args:
        block_trajectories: List of block trajectory arrays, each [T, 2 or 3]
        agent_trajectories: Optional list of agent trajectory arrays, each [T, 2]
        goal_pose: [x, y, theta] of goal position
        title: Plot title
        coverage_values: List of final coverage values for each trajectory
        figsize: Figure size
        save_path: Path to save the figure
        show_colorbar: Whether to show time colorbar
        dpi: DPI for saved figure
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw background with T-shaped goal
    create_pusht_background(ax, goal_pose=goal_pose, show_agent_start=True)
    
    # Plot block trajectories with time-based coloring
    plot_trajectory_heatmap(ax, block_trajectories, 
                           cmap_name='plasma',
                           linewidth=2.0,
                           alpha=0.6,
                           label='Block')
    
    # Plot agent trajectories with different colormap
    if agent_trajectories is not None:
        plot_trajectory_heatmap(ax, agent_trajectories,
                               cmap_name='viridis', 
                               linewidth=1.0,
                               alpha=0.4,
                               label='Agent')
    
    # Add title with stats
    if coverage_values is not None and len(coverage_values) > 0:
        mean_cov = np.mean(coverage_values)
        success_rate = np.mean([c >= 0.95 for c in coverage_values])
        title = f'{title}\nMean Coverage: {mean_cov:.1%}, Success Rate: {success_rate:.1%}'
    
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    
    # Add colorbar for time
    if show_colorbar:
        sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('Time (normalized)', fontsize=9)
    
    # Add legend
    ax.legend(loc='upper left', fontsize=9)
    
    # Remove axis spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    
    return fig


def create_multi_method_comparison_plot(
    method_trajectories: dict,
    goal_pose: np.ndarray = None,
    figsize: Tuple[int, int] = (16, 4),
    save_path: str = None,
    dpi: int = 150
) -> plt.Figure:
    """
    Create a side-by-side comparison plot of multiple methods (like Figure 5 in the paper).
    
    Args:
        method_trajectories: Dict mapping method name to list of block trajectories
        goal_pose: Goal pose for the T
        figsize: Figure size
        save_path: Path to save figure
        dpi: DPI for saved figure
        
    Returns:
        matplotlib Figure object
    """
    n_methods = len(method_trajectories)
    fig, axes = plt.subplots(1, n_methods, figsize=figsize)
    
    if n_methods == 1:
        axes = [axes]
    
    for ax, (method_name, trajectories) in zip(axes, method_trajectories.items()):
        create_pusht_background(ax, goal_pose=goal_pose)
        plot_trajectory_heatmap(ax, trajectories, cmap_name='plasma', linewidth=2.0, alpha=0.6)
        ax.set_title(method_name, fontsize=12, fontweight='bold')
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('y', fontsize=10)
    
    plt.tight_layout()
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    
    return fig
