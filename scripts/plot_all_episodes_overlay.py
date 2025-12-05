#!/usr/bin/env python3
"""
Multi-Episode Trajectory Overlay Script

Plots all episodes from an evaluation overlaid on the same plot to visualize
consistency and variability of the diffusion policy.
"""

import zarr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Polygon, Rectangle
from matplotlib.transforms import Affine2D
from pathlib import Path
import argparse
from typing import List, Tuple, Set

# Reference coordinates - adjusted to match actual trajectory positions
# Note: Board is the inner boundary, Workspace is the outer reachable area
# All coordinates shifted by offset from configured to actual sphere end position
REFERENCE_POINTS = {
    # Initial TCP position
    "tcp_at_j_init": [0.6158513694265592, 0.17178544114545136, -0.3014212431127473],
    # Board corners (inner boundary) - shifted
    "board_p1": [0.7548499446055756, 0.19698712351653327, -0.2956672172943469],
    "board_p2": [0.6424903940009311, 0.29366964070457036, -0.2984347741651871],
    "board_p3": [0.7382062302068387, 0.40758993969961693, -0.2970589885539159],
    "board_p4": [0.8526890533573702, 0.3113083461043496, -0.29255820363239354],
    # Workspace corners (outer boundary) - shifted
    "workspace_p1": [0.6948330383982414, 0.08948328905951124, -0.2990096568562124],
    "workspace_p2": [0.5394841689406544, 0.22239752839738314, -0.30072094936890775],
    "workspace_p3": [0.6063260856111391, 0.32172566768119365, -0.3012114227057066],
    "workspace_p4": [0.776511243744269, 0.1793504403289917, -0.295299017748294],
    # Targets - based on actual mean endpoint positions from trajectory data
    "square_target": [0.6701, 0.2702, -0.3015],  # Sphere actual endpoint (where sphere evals go)
    "circle_target": [0.7320, 0.2375, -0.3015],  # Cube actual endpoint (where cube evals go)
}

# Failure episodes per evaluation (CORRECTED based on endpoint position verification)
# Episodes are failures only if endpoint is OUTSIDE the target region
# See verify_all_scenarios.py for complete verification details
EVAL_FAILURES = {
    'eval_two_cameras_sphere': [32, 36, 49],  # 3 failures: rubbing (32), pushing out of boundary (36, 49)
    'eval_two_cameras_sphere_to_cube': [],
    'eval_two_camera_cube': [32, 33, 42, 47, 48, 49, 50],  # 7 failures: pushing out (32), rubbing (33, 48), rotation lacking (42, 47, 49), brute forcing (50)
    'eval_one_camera_sphere': [4, 9, 12, 14, 16, 17, 22, 24, 25, 26, 28, 29, 30, 31],  # 14 failures: mostly rubbing
    'eval_one_camera_cube': [10, 23],  # 2 failures: brute forcing (10), stuck at frame (23)
    'eval_two_cameras_cubes_with_noglvoes': [],
    'eval_two_cameras_cubes_with_no_glvoes_and_white_objects_as_disturbances': [8],
    'eval_two_camera_cube_with_all_disturbances': [3, 8, 9],
    'eval_two_cameras_cube_up_to_31': [3, 8, 9, 10, 15, 24, 28],  # 7 failures
    'eval_two_cameras_sphere_up_to_31': [],  # First 31 episodes from sphere eval
}


def get_failure_episodes(eval_name: str = None) -> Set[int]:
    """Get set of episode numbers that have failures for a specific evaluation."""
    if eval_name is None:
        return set()

    # Extract basename from path if needed
    import os
    eval_basename = os.path.basename(eval_name.rstrip('/'))

    # Try exact match first
    if eval_basename in EVAL_FAILURES:
        return set(EVAL_FAILURES[eval_basename])

    # Try to match evaluation name (sort by length to match longest first)
    for key in sorted(EVAL_FAILURES.keys(), key=len, reverse=True):
        if key in eval_name or eval_name in key:
            return set(EVAL_FAILURES[key])

    return set()


def load_all_episodes(buffer_path: Path) -> Tuple[List[np.ndarray], dict]:
    """Load all episodes from replay buffer."""
    store = zarr.open(str(buffer_path), mode='r')

    episode_ends = np.array(store['meta/episode_ends'][:])
    eef_poses = np.array(store['data/robot_eef_pose'][:])

    # Extract each episode
    episodes = []
    for i in range(len(episode_ends)):
        start_idx = 0 if i == 0 else int(episode_ends[i - 1])
        end_idx = int(episode_ends[i])

        # Extract x, y, z positions (first 3 columns)
        trajectory = eef_poses[start_idx:end_idx, :3]
        episodes.append(trajectory)

    stats = {
        'num_episodes': len(episodes),
        'min_length': min(len(ep) for ep in episodes),
        'max_length': max(len(ep) for ep in episodes),
        'mean_length': np.mean([len(ep) for ep in episodes])
    }

    return episodes, stats


def add_reference_regions_2d(ax, plane='xy', eval_name=None):
    """Add board, workspace boundaries, and targets to 2D plot."""
    # Select projection indices (swap x/y for visualization convention)
    if plane == 'xy':
        idx1, idx2 = 1, 0  # Swapped: Y on x-axis, X on y-axis
    elif plane == 'xz':
        idx1, idx2 = 0, 2
    else:  # yz
        idx1, idx2 = 1, 2

    # Board boundary (light brown)
    board_corners = np.array([
        REFERENCE_POINTS["board_p1"],
        REFERENCE_POINTS["board_p2"],
        REFERENCE_POINTS["board_p3"],
        REFERENCE_POINTS["board_p4"]
    ])
    board_polygon = Polygon(board_corners[:, [idx1, idx2]],
                            fill=False, edgecolor='#8B7355', linewidth=2.5,
                            linestyle='--', alpha=0.8, label='Board')
    ax.add_patch(board_polygon)

    # Workspace boundary (gray, dotted)
    workspace_corners = np.array([
        REFERENCE_POINTS["workspace_p1"],
        REFERENCE_POINTS["workspace_p2"],
        REFERENCE_POINTS["workspace_p3"],
        REFERENCE_POINTS["workspace_p4"]
    ])
    workspace_polygon = Polygon(workspace_corners[:, [idx1, idx2]],
                                fill=False, edgecolor='#757575', linewidth=2.5,
                                linestyle=':', alpha=0.8, label='Workspace')
    ax.add_patch(workspace_polygon)

    # Target locations - NOTE: variable names are swapped relative to shapes
    # square_target position is where SPHERE evals go (but show as circle for sphere evals)
    # circle_target position is where CUBE evals go (but show as square for cube evals)
    sphere_pos = REFERENCE_POINTS["square_target"]  # Where sphere evals end
    cube_pos = REFERENCE_POINTS["circle_target"]    # Where cube evals end

    # Always show both targets
    show_cube = True
    show_sphere = True

    # Show cube target as SQUARE (rotated 45 degrees, 30mm per side)
    if show_cube:
        square_size = 0.030  # 30mm side length
        square_rect = Rectangle(
            (cube_pos[idx1] - square_size/2, cube_pos[idx2] - square_size/2),
            square_size, square_size,
            fill=True, facecolor='#FF4500', edgecolor='#CC0000',
            linewidth=2, alpha=0.8, zorder=10, label='Cube Target')
        # Rotate 45 degrees around center
        transform = Affine2D().rotate_deg_around(
            cube_pos[idx1], cube_pos[idx2], 45) + ax.transData
        square_rect.set_transform(transform)
        ax.add_patch(square_rect)

    # Show sphere target as CIRCLE (35mm diameter)
    if show_sphere:
        ax.scatter([sphere_pos[idx1]], [sphere_pos[idx2]],
                   c='#00E5FF', s=960, marker='o', alpha=0.8, edgecolors='#0091EA',
                   linewidths=2, zorder=10, label='Sphere Target')


def plot_3d_overlay(
    episodes: List[np.ndarray],
    output_path: Path,
    title: str = "All Episodes Overlay",
    colormap: str = 'viridis',
    show_start_end: bool = True,
    show_reference: bool = True
):
    """Plot all episodes overlaid in 3D."""
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Get colormap
    colors = cm.get_cmap(colormap)(np.linspace(0, 1, len(episodes)))

    # Plot each episode
    for i, (trajectory, color) in enumerate(zip(episodes, colors)):
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                color=color, linewidth=1.5, alpha=0.6, label=f'Ep {i}' if len(episodes) <= 10 else None)

        # Mark start and end points
        if show_start_end:
            ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2],
                      c=[color], s=100, marker='o', alpha=0.8, edgecolors='green', linewidths=2)
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2],
                      c='black', s=100, marker='*', alpha=0.9, edgecolors='black', linewidths=2)

    ax.set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z Position (m)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # Add reference regions (board, workspace, targets)
    if show_reference:
        add_reference_regions_3d(ax)

    # Build comprehensive legend
    from matplotlib.lines import Line2D
    custom_lines = []

    if show_start_end:
        custom_lines.extend([
            Line2D([0], [0], marker='o', color='w', markerfacecolor='g',
                   markersize=10, markeredgecolor='green', markeredgewidth=2, label='Start'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='black',
                   markersize=12, markeredgecolor='black', markeredgewidth=2, label='End')
        ])

    # Add reference region legend items
    if show_reference:
        custom_lines.extend([
            Line2D([0], [0], color='blue', linewidth=2.5, linestyle='--',
                   label='Board'),
            Line2D([0], [0], color='orange', linewidth=2.5, linestyle='--',
                   label='Workspace'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='#FF0000',
                   markersize=10, markeredgecolor='#B30000', markeredgewidth=2,
                   label='Square Target'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#00E5FF',
                   markersize=10, markeredgecolor='#0091EA', markeredgewidth=2,
                   label='Circle Target')
        ])

    if custom_lines:
        ax.legend(handles=custom_lines, loc='upper right', fontsize=10, framealpha=0.9)

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved 3D overlay to {output_path}")


def plot_2d_overlay(
    episodes: List[np.ndarray],
    output_path: Path,
    title: str = "All Episodes Overlay (Top View)",
    colormap: str = 'viridis',
    plane: str = 'xy',
    show_start_end: bool = True,
    show_reference: bool = True,
    eval_name: str = None
):
    """Plot all episodes overlaid in 2D."""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Select projection plane (swap x/y for visualization convention)
    if plane == 'xy':
        idx1, idx2 = 1, 0  # Swapped: Y on x-axis, X on y-axis
        xlabel, ylabel = 'Y Position (m)', 'X Position (m)'
    elif plane == 'xz':
        idx1, idx2 = 0, 2
        xlabel, ylabel = 'X Position (m)', 'Z Position (m)'
    else:  # yz
        idx1, idx2 = 1, 2
        xlabel, ylabel = 'Y Position (m)', 'Z Position (m)'

    # Get colormap
    colors = cm.get_cmap(colormap)(np.linspace(0, 1, len(episodes)))

    # Get failure episodes for this specific evaluation
    failure_episodes = get_failure_episodes(eval_name)

    # Plot each episode
    for i, (trajectory, color) in enumerate(zip(episodes, colors)):
        x_data, y_data = trajectory[:, idx1], trajectory[:, idx2]
        ax.plot(x_data, y_data, color=color, linewidth=1.5, alpha=0.6,
                label=f'Ep {i}' if len(episodes) <= 10 else None)

        # Mark start and end points
        if show_start_end:
            ax.scatter(x_data[0], y_data[0], c=[color], s=100, marker='o',
                      alpha=0.8, edgecolors='green', linewidths=2, zorder=5)

            # Different markers for failures vs successes
            if i in failure_episodes:
                # Failure: bright red X marker
                ax.scatter(x_data[-1], y_data[-1], c='#FF0000', s=150, marker='X',
                          alpha=0.9, edgecolors='#CC0000', linewidths=2, zorder=5)
            else:
                # Success: black star marker
                ax.scatter(x_data[-1], y_data[-1], c='black', s=100, marker='*',
                          alpha=0.9, edgecolors='black', linewidths=2, zorder=5)

    # Add initial robot pose marker
    if show_reference:
        tcp_init = REFERENCE_POINTS["tcp_at_j_init"]
        ax.scatter(tcp_init[idx1], tcp_init[idx2], c='#FF1493', s=300, marker='P',
                  alpha=0.95, edgecolors='#C71585', linewidths=3, zorder=6,
                  label='Initial Robot Pose')  # Strong pink

    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    # Invert X-axis for XY plane (was Y-axis before swap) to match front camera perspective
    if plane == 'xy':
        ax.invert_xaxis()

    # Add reference regions (board, workspace, targets)
    if show_reference:
        add_reference_regions_2d(ax, plane, eval_name)
    
    # Set consistent axis limits for all plots (for comparison)
    if plane == 'xy':
        workspace_corners = np.array([
            REFERENCE_POINTS["workspace_p1"][:2],
            REFERENCE_POINTS["workspace_p2"][:2],
            REFERENCE_POINTS["workspace_p3"][:2],
            REFERENCE_POINTS["workspace_p4"][:2],
        ])
        # After swap: x-axis shows Y values, y-axis shows X values
        y_coords = workspace_corners[:, 0]  # X values (now on y-axis)
        x_coords = workspace_corners[:, 1]  # Y values (now on x-axis)
        
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        # Add 10% padding
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        
        ax.set_xlim(x_max + x_padding, x_min - x_padding)  # Inverted for xy plane
        ax.set_ylim(y_min - y_padding, y_max + y_padding)

    # Build comprehensive legend
    from matplotlib.lines import Line2D
    custom_lines = []

    if show_start_end:
        custom_lines.extend([
            Line2D([0], [0], marker='o', color='w', markerfacecolor='g',
                   markersize=10, markeredgecolor='green', markeredgewidth=2,
                   label='Start'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='black',
                   markersize=12, markeredgecolor='black', markeredgewidth=2,
                   label='End (Success)'),
            Line2D([0], [0], marker='X', color='w', markerfacecolor='#FF0000',
                   markersize=12, markeredgecolor='#CC0000', markeredgewidth=2,
                   label='End (Failure)')
        ])

    # Add initial pose marker to legend
    if show_reference:
        custom_lines.append(
            Line2D([0], [0], marker='P', color='w', markerfacecolor='#FF1493',
                   markersize=12, markeredgecolor='#C71585', markeredgewidth=2,
                   label='Initial Robot Pose')
        )

    # Add reference region legend items
    if show_reference:
        custom_lines.extend([
            Line2D([0], [0], color='#8B7355', linewidth=2.5, linestyle='--',
                   label='Board'),
            Line2D([0], [0], color='#757575', linewidth=2.5, linestyle=':',
                   label='Workspace'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='#FF4500',
                   markersize=10, markeredgecolor='#CC0000', markeredgewidth=2,
                   label='Square Target'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#00E5FF',
                   markersize=10, markeredgecolor='#0091EA', markeredgewidth=2,
                   label='Circle Target')
        ])

    if custom_lines:
        ax.legend(handles=custom_lines, loc='best', fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved 2D overlay to {output_path}")


def plot_density_heatmap(
    episodes: List[np.ndarray],
    output_path: Path,
    title: str = "Trajectory Density Heatmap",
    plane: str = 'xy',
    bins: int = 50
):
    """Plot 2D density heatmap of all trajectories."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Select projection plane (swap x/y for visualization convention)
    if plane == 'xy':
        idx1, idx2 = 1, 0  # Swapped: Y on x-axis, X on y-axis
        xlabel, ylabel = 'Y Position (m)', 'X Position (m)'
    elif plane == 'xz':
        idx1, idx2 = 0, 2
        xlabel, ylabel = 'X Position (m)', 'Z Position (m)'
    else:  # yz
        idx1, idx2 = 1, 2
        xlabel, ylabel = 'Y Position (m)', 'Z Position (m)'

    # Concatenate all trajectories
    all_points = np.vstack([ep[:, [idx1, idx2]] for ep in episodes])

    # Create 2D histogram
    heatmap, xedges, yedges = np.histogram2d(
        all_points[:, 0], all_points[:, 1], bins=bins
    )

    # Plot heatmap
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='hot', aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Trajectory Density', fontsize=12, fontweight='bold')

    # Overlay trajectory paths
    colors = cm.get_cmap('viridis')(np.linspace(0, 1, len(episodes)))
    for trajectory, color in zip(episodes, colors):
        x_data, y_data = trajectory[:, idx1], trajectory[:, idx2]
        ax.plot(x_data, y_data, color=color, linewidth=0.5, alpha=0.3)

    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Invert X-axis for XY plane to match 2d overlay perspective
    if plane == 'xy':
        ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved density heatmap to {output_path}")


def plot_episode_statistics(
    episodes: List[np.ndarray],
    output_path: Path,
    title: str = "Episode Statistics"
):
    """Plot statistics about episodes."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Episode lengths
    lengths = [len(ep) for ep in episodes]
    axes[0, 0].bar(range(len(lengths)), lengths, color='steelblue', alpha=0.7)
    axes[0, 0].axhline(np.mean(lengths), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(lengths):.1f}')
    axes[0, 0].set_xlabel('Episode Number', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Number of Timesteps', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Episode Lengths', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Total distance traveled per episode
    distances = []
    for ep in episodes:
        diffs = np.diff(ep, axis=0)
        dist = np.sum(np.linalg.norm(diffs, axis=1))
        distances.append(dist)

    axes[0, 1].bar(range(len(distances)), distances, color='forestgreen', alpha=0.7)
    axes[0, 1].axhline(np.mean(distances), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(distances):.3f}m')
    axes[0, 1].set_xlabel('Episode Number', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Total Distance (m)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Distance Traveled per Episode', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Start position distribution
    start_positions = np.array([ep[0, :2] for ep in episodes])
    axes[1, 0].scatter(start_positions[:, 0], start_positions[:, 1],
                       c=range(len(episodes)), cmap='viridis', s=100, alpha=0.7, edgecolors='black')
    axes[1, 0].set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Start Positions (XY)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_aspect('equal', adjustable='box')

    # End position distribution
    end_positions = np.array([ep[-1, :2] for ep in episodes])
    axes[1, 1].scatter(end_positions[:, 0], end_positions[:, 1],
                       c=range(len(episodes)), cmap='viridis', s=100, alpha=0.7, edgecolors='black')
    axes[1, 1].set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('End Positions (XY)', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_aspect('equal', adjustable='box')

    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved statistics plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Overlay all episodes from a DRL evaluation'
    )
    parser.add_argument('eval_dir', type=str,
                       help='Path to evaluation directory')
    parser.add_argument('--colormap', type=str, default='viridis',
                       help='Matplotlib colormap for episodes (default: viridis)')
    parser.add_argument('--plane', type=str, default='xy',
                       choices=['xy', 'xz', 'yz'],
                       help='2D projection plane (default: xy)')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['2d', 'heatmap', 'all'],
                       help='Visualization mode (default: all)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: eval_dir/trajectory_plots)')
    parser.add_argument('--no-markers', action='store_true',
                       help='Disable start/end markers')
    parser.add_argument('--no-reference', action='store_true',
                       help='Disable board/workspace/target reference regions')
    parser.add_argument('--max-episodes', type=int, default=None,
                       help='Limit to first N episodes (default: all)')

    args = parser.parse_args()

    # Setup paths
    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        print(f"Error: Directory {eval_dir} does not exist")
        return

    replay_buffer_path = eval_dir / 'replay_buffer.zarr'
    if not replay_buffer_path.exists():
        print(f"Error: Replay buffer not found at {replay_buffer_path}")
        return

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = eval_dir / 'trajectory_plots'
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load all episodes
    print(f"Loading all episodes from {replay_buffer_path}")
    episodes, stats = load_all_episodes(replay_buffer_path)

    # Limit episodes if requested
    if args.max_episodes is not None and args.max_episodes < len(episodes):
        episodes = episodes[:args.max_episodes]
        print(f"  Limiting to first {args.max_episodes} episodes")
        # Recalculate stats
        stats['num_episodes'] = len(episodes)
        stats['min_length'] = min(len(ep) for ep in episodes)
        stats['max_length'] = max(len(ep) for ep in episodes)
        stats['mean_length'] = np.mean([len(ep) for ep in episodes])

    print(f"\nDataset Statistics:")
    print(f"  Total episodes: {stats['num_episodes']}")
    print(f"  Episode length range: {stats['min_length']} - {stats['max_length']} timesteps")
    print(f"  Mean episode length: {stats['mean_length']:.1f} timesteps")
    print()

    # Fix typo in folder names for display
    display_name = eval_dir.name.replace('glvoes', 'gloves')

    # If max_episodes was used, append to display name and eval name
    if args.max_episodes is not None:
        display_name = f"{display_name}_up_to_{args.max_episodes}"

    # Use "trajectories" for training data, "episodes" for evaluation data
    is_training = not eval_dir.name.startswith('eval_')
    count_label = "Trajectories" if is_training else "Episodes"
    # For training data, manually set count to 100
    count = 100 if is_training else stats['num_episodes']
    title_base = f"{display_name} - All {count} {count_label}"

    show_markers = not args.no_markers
    show_reference = not args.no_reference

    # Extract evaluation name for failure matching
    eval_name = eval_dir.name
    if args.max_episodes is not None:
        eval_name = f"{eval_name}_up_to_{args.max_episodes}"

    # Generate plots
    if args.mode == '2d' or args.mode == 'all':
        output_path = output_dir / f'all_episodes_2d_overlay_{args.plane}.png'
        plot_2d_overlay(episodes, output_path,
                       f"{title_base} ({args.plane.upper()} Plane)",
                       args.colormap, args.plane, show_markers, show_reference,
                       eval_name)

    if args.mode == 'heatmap' or args.mode == 'all':
        output_path = output_dir / f'all_episodes_heatmap_{args.plane}.png'
        plot_density_heatmap(episodes, output_path, f"{title_base} - Density Heatmap",
                           args.plane)

    print(f"\n✓ Multi-episode visualization complete! Plots saved to {output_dir}")


if __name__ == '__main__':
    main()
