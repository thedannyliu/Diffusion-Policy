#!/usr/bin/env python3
"""
Training Data Density Analysis Script

Analyzes spatial density of training data, comparing "easy spots" (central regions)
vs "hard spots" (edge positions).
"""

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import zarr
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Polygon


REFERENCE_POINTS = {
    "tcp_at_j_init": [0.6158513694265592, 0.17178544114545136, -0.3014212431127473],
    "board_p1": [0.7548499446055756, 0.19698712351653327, -0.2956672172943469],
    "board_p2": [0.6424903940009311, 0.29366964070457036, -0.2984347741651871],
    "board_p3": [0.7382062302068387, 0.40758993969961693, -0.2970589885539159],
    "board_p4": [0.8526890533573702, 0.3113083461043496, -0.29255820363239354],
    "workspace_p1": [0.6948330383982414, 0.08948328905951124, -0.2990096568562124],
    "workspace_p2": [0.5394841689406544, 0.22239752839738314, -0.30072094936890775],
    "workspace_p3": [0.6063260856111391, 0.32172566768119365, -0.3012114227057066],
    "workspace_p4": [0.776511243744269, 0.1793504403289917, -0.295299017748294],
    "square_target": [0.6701, 0.2702, -0.3015],
    "circle_target": [0.7320, 0.2375, -0.3015],
}


def load_training_data(zarr_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load trajectory positions and episode start positions from replay buffer."""
    store = zarr.open(str(zarr_path), mode='r')
    eef_poses = np.array(store['data/robot_eef_pose'][:])
    episode_ends = np.array(store['meta/episode_ends'][:])

    all_positions = eef_poses[:, :2]

    start_positions = []
    for i in range(len(episode_ends)):
        start_idx = 0 if i == 0 else int(episode_ends[i - 1])
        start_positions.append(eef_poses[start_idx, :2])

    return all_positions, np.array(start_positions)


def define_workspace_grid(num_rows: int = 5, num_cols: int = 10) -> Dict:
    """Define 5×10 grid across workspace with easy/hard spot regions."""
    workspace_corners = np.array([
        REFERENCE_POINTS["workspace_p1"][:2],
        REFERENCE_POINTS["workspace_p2"][:2],
        REFERENCE_POINTS["workspace_p3"][:2],
        REFERENCE_POINTS["workspace_p4"][:2],
    ])

    width_vector = workspace_corners[1] - workspace_corners[0]
    height_vector = workspace_corners[3] - workspace_corners[0]
    margin = 0.08

    grid_positions = []
    for i in range(num_rows):
        height_param = margin + (i / (num_rows - 1)) * (1 - 2 * margin)
        for j in range(num_cols):
            width_param = margin + (j / (num_cols - 1)) * (1 - 2 * margin)
            pos = workspace_corners[0] + width_param * width_vector + height_param * height_vector
            grid_positions.append(pos)
    
    grid_positions = np.array(grid_positions)
    x_grid = grid_positions[:, 0].reshape(num_rows, num_cols)
    y_grid = grid_positions[:, 1].reshape(num_rows, num_cols)
    
    x_min, x_max = workspace_corners[:, 0].min(), workspace_corners[:, 0].max()
    y_min, y_max = workspace_corners[:, 1].min(), workspace_corners[:, 1].max()

    easy_x_range = (x_grid[1, 1], x_grid[1, 8])
    easy_y_range = (y_grid[1, 1], y_grid[4, 1])

    return {
        'x_grid': x_grid,
        'y_grid': y_grid,
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max,
        'num_rows': num_rows,
        'num_cols': num_cols,
        'total_spots': num_rows * num_cols,
        'easy_spots_count': 32,
        'hard_spots_count': 18,
        'easy_x_range': easy_x_range,
        'easy_y_range': easy_y_range,
    }


def calculate_density(positions: np.ndarray, bins: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate 2D density histogram of positions."""
    heatmap, xedges, yedges = np.histogram2d(
        positions[:, 0], positions[:, 1], bins=bins
    )
    return heatmap, xedges, yedges


def calculate_region_density(positions: np.ndarray, grid_info: Dict) -> Dict:
    """Calculate density statistics for easy vs hard spot regions."""
    x_grid = grid_info['x_grid']
    y_grid = grid_info['y_grid']
    num_rows = grid_info['num_rows']
    num_cols = grid_info['num_cols']

    easy_positions = []
    hard_positions = []

    for i in range(num_rows):
        for j in range(num_cols):
            x, y = x_grid[i, j], y_grid[i, j]
            is_easy = (1 <= i <= 4) and (1 <= j <= 8)

            if is_easy:
                easy_positions.append([x, y])
            else:
                hard_positions.append([x, y])

    easy_positions = np.array(easy_positions)
    hard_positions = np.array(hard_positions)

    easy_count = 0
    hard_count = 0
    radius = 0.02

    for pos in positions:
        if len(easy_positions) > 0:
            easy_dists = np.linalg.norm(easy_positions - pos, axis=1)
            if np.min(easy_dists) < radius:
                easy_count += 1
                continue

        if len(hard_positions) > 0:
            hard_dists = np.linalg.norm(hard_positions - pos, axis=1)
            if np.min(hard_dists) < radius:
                hard_count += 1

    total_count = easy_count + hard_count
    total_area = (grid_info['x_max'] - grid_info['x_min']) * (grid_info['y_max'] - grid_info['y_min'])
    spot_area = total_area / (num_rows * num_cols)
    easy_area = 32 * spot_area
    hard_area = 18 * spot_area

    easy_density = easy_count / easy_area if easy_area > 0 else 0
    hard_density = hard_count / hard_area if hard_area > 0 else 0

    return {
        'easy_count': easy_count,
        'hard_count': hard_count,
        'total_count': total_count,
        'easy_percentage': (easy_count / total_count * 100) if total_count > 0 else 0,
        'hard_percentage': (hard_count / total_count * 100) if total_count > 0 else 0,
        'easy_area': easy_area,
        'hard_area': hard_area,
        'total_area': total_area,
        'easy_density': easy_density,
        'hard_density': hard_density,
        'density_ratio': easy_density / hard_density if hard_density > 0 else float('inf'),
    }


def plot_density_with_regions(
    positions: np.ndarray,
    grid_info: Dict,
    density_stats: Dict,
    output_path: Path,
    title: str = "Training Data Spatial Density",
    bins: int = 50
):
    """Plot density heatmap with grid overlay showing easy/hard spots."""
    fig, ax = plt.subplots(figsize=(16, 12))

    heatmap, xedges, yedges = calculate_density(positions[:, [1, 0]], bins)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(heatmap.T, extent=extent, origin='lower',
                   cmap='hot', aspect='auto', alpha=0.8)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Visit Count (Training Timesteps)', fontsize=14, fontweight='bold')

    x_grid = grid_info['x_grid']
    y_grid = grid_info['y_grid']

    for i in range(grid_info['num_rows']):
        for j in range(grid_info['num_cols']):
            x, y = x_grid[i, j], y_grid[i, j]
            is_easy = (1 <= i <= 4) and (1 <= j <= 8)

            if is_easy:
                ax.scatter([y], [x], s=150, c='#00FF00', marker='o',
                          edgecolors='#00FF00', linewidths=2, alpha=0.7, zorder=5)
            else:
                ax.scatter([y], [x], s=100, c='#FF0000', marker='o',
                          edgecolors='#FF0000', linewidths=2, alpha=0.7, zorder=5)


    workspace_corners = np.array([
        REFERENCE_POINTS["workspace_p1"][:2],
        REFERENCE_POINTS["workspace_p2"][:2],
        REFERENCE_POINTS["workspace_p3"][:2],
        REFERENCE_POINTS["workspace_p4"][:2],
    ])
    workspace_corners_swapped = workspace_corners[:, [1, 0]]
    workspace_polygon = Polygon(workspace_corners_swapped, fill=False, edgecolor='#FFA500',
                                linewidth=3, linestyle=':', label='Workspace Boundary', alpha=0.8)
    ax.add_patch(workspace_polygon)

    board_corners = np.array([
        REFERENCE_POINTS["board_p1"][:2],
        REFERENCE_POINTS["board_p2"][:2],
        REFERENCE_POINTS["board_p3"][:2],
        REFERENCE_POINTS["board_p4"][:2],
    ])
    board_corners_swapped = board_corners[:, [1, 0]]
    board_polygon = Polygon(board_corners_swapped, fill=False, edgecolor='#FFFFFF',
                            linewidth=3, linestyle='-', alpha=0.9, label='Board Boundary')
    ax.add_patch(board_polygon)

    sphere_target = REFERENCE_POINTS["square_target"][:2]
    cube_target = REFERENCE_POINTS["circle_target"][:2]

    ax.scatter([sphere_target[1]], [sphere_target[0]], c='#00E5FF', s=500, marker='o',
               alpha=0.95, edgecolors='#0091EA', linewidths=3, label='Sphere Target', zorder=10)
    ax.scatter([cube_target[1]], [cube_target[0]], c='#FF4500', s=500, marker='s',
               alpha=0.95, edgecolors='#CC0000', linewidths=3, label='Cube Target', zorder=10)

    ax.set_xlabel('Y Position (m)', fontsize=14, fontweight='bold')
    ax.set_ylabel('X Position (m)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Set consistent axis limits (matching plot_all_episodes_overlay.py)
    # After swapping X and Y, limits need to be swapped too
    workspace_corners_for_limits = np.array([
        REFERENCE_POINTS["workspace_p1"][:2],
        REFERENCE_POINTS["workspace_p2"][:2],
        REFERENCE_POINTS["workspace_p3"][:2],
        REFERENCE_POINTS["workspace_p4"][:2],
    ])
    x_min_lim = workspace_corners_for_limits[:, 0].min()
    x_max_lim = workspace_corners_for_limits[:, 0].max()
    y_min_lim = workspace_corners_for_limits[:, 1].min()
    y_max_lim = workspace_corners_for_limits[:, 1].max()

    # Add 10% padding
    x_padding = (x_max_lim - x_min_lim) * 0.1
    y_padding = (y_max_lim - y_min_lim) * 0.1

    # Swap limits since we swapped axes
    ax.set_xlim(y_min_lim - y_padding, y_max_lim + y_padding)
    ax.set_ylim(x_min_lim - x_padding, x_max_lim + x_padding)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved density plot to {output_path}")


def plot_grid_spots(
    positions: np.ndarray,
    grid_info: Dict,
    output_path: Path,
    title: str = "50-Spot Grid Coverage"
):
    """Plot 5×10 grid with spot positions and visit counts."""
    fig, ax = plt.subplots(figsize=(18, 10))

    x_grid = grid_info['x_grid']
    y_grid = grid_info['y_grid']
    spot_counts = np.zeros((grid_info['num_rows'], grid_info['num_cols']))
    cell_radius = 0.015

    for i in range(grid_info['num_rows']):
        for j in range(grid_info['num_cols']):
            x, y = x_grid[i, j], y_grid[i, j]
            distances = np.sqrt((positions[:, 0] - x)**2 + (positions[:, 1] - y)**2)
            spot_counts[i, j] = np.sum(distances < cell_radius)

    max_count = spot_counts.max()

    for i in range(grid_info['num_rows']):
        for j in range(grid_info['num_cols']):
            x, y = x_grid[i, j], y_grid[i, j]
            count = spot_counts[i, j]
            is_easy = (1 <= i <= 4) and (1 <= j <= 8)

            color_intensity = count / max_count if max_count > 0 else 0
            color = cm.get_cmap('YlOrRd')(color_intensity)
            border_color = '#00FF00' if is_easy else '#FF0000'
            border_width = 3 if is_easy else 2

            ax.scatter([y], [x], s=800, c=[color], marker='o',
                      edgecolors=border_color, linewidths=border_width,
                      alpha=0.8, zorder=5)

            ax.text(y, x, f'{int(count)}', ha='center', va='center',
                   fontsize=8, fontweight='bold', color='black', zorder=6)

    workspace_corners = np.array([
        REFERENCE_POINTS["workspace_p1"][:2],
        REFERENCE_POINTS["workspace_p2"][:2],
        REFERENCE_POINTS["workspace_p3"][:2],
        REFERENCE_POINTS["workspace_p4"][:2],
    ])
    workspace_corners_swapped = workspace_corners[:, [1, 0]]
    workspace_polygon = Polygon(workspace_corners_swapped, fill=False, edgecolor='#757575',
                                linewidth=2.5, linestyle=':', alpha=0.8, label='Workspace')
    ax.add_patch(workspace_polygon)

    board_corners = np.array([
        REFERENCE_POINTS["board_p1"][:2],
        REFERENCE_POINTS["board_p2"][:2],
        REFERENCE_POINTS["board_p3"][:2],
        REFERENCE_POINTS["board_p4"][:2],
    ])
    board_corners_swapped = board_corners[:, [1, 0]]
    board_polygon = Polygon(board_corners_swapped, fill=False, edgecolor='#8B7355',
                           linewidth=2.5, linestyle='--', alpha=0.8, label='Board')
    ax.add_patch(board_polygon)

    sphere_target = REFERENCE_POINTS["square_target"][:2]
    cube_target = REFERENCE_POINTS["circle_target"][:2]

    ax.scatter([sphere_target[1]], [sphere_target[0]], c='#00E5FF', s=600, marker='o',
               alpha=0.95, edgecolors='#0091EA', linewidths=3, label='Sphere Target', zorder=10)
    ax.scatter([cube_target[1]], [cube_target[0]], c='#FF0000', s=600, marker='s',
               alpha=0.95, edgecolors='#B30000', linewidths=3, label='Cube Target', zorder=10)

    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='w', markersize=15,
               markeredgecolor='#00FF00', markeredgewidth=3, label='Easy Spots (32 spots)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='w', markersize=15,
               markeredgecolor='#FF0000', markeredgewidth=2, label='Hard Spots (18 spots)'),
    ]

    legend1 = ax.legend(handles=custom_lines, loc='upper left', fontsize=12,
                       framealpha=0.9, title='Grid Spots')
    ax.add_artist(legend1)

    ax.set_xlabel('Y Position (m)', fontsize=14, fontweight='bold')
    ax.set_ylabel('X Position (m)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)

    workspace_corners_for_limits = np.array([
        REFERENCE_POINTS["workspace_p1"][:2],
        REFERENCE_POINTS["workspace_p2"][:2],
        REFERENCE_POINTS["workspace_p3"][:2],
        REFERENCE_POINTS["workspace_p4"][:2],
    ])
    x_min_lim = workspace_corners_for_limits[:, 0].min()
    x_max_lim = workspace_corners_for_limits[:, 0].max()
    y_min_lim = workspace_corners_for_limits[:, 1].min()
    y_max_lim = workspace_corners_for_limits[:, 1].max()

    x_padding = (x_max_lim - x_min_lim) * 0.1
    y_padding = (y_max_lim - y_min_lim) * 0.1

    ax.set_xlim(y_min_lim - y_padding, y_max_lim + y_padding)
    ax.set_ylim(x_min_lim - x_padding, x_max_lim + x_padding)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved grid spots plot to {output_path}")


def main():
    """Analyze training data density."""
    print("=" * 80)
    print("Training Data Density Analysis")
    print("=" * 80)

    # Get repo root directory (parent of scripts/)
    repo_root = Path(__file__).parent.parent
    training_dir = repo_root / "data" / "training"
    output_dir = repo_root / "plots" / "training_density"
    output_dir.mkdir(exist_ok=True, parents=True)

    datasets = {
        'two_cameras_sphere': training_dir / 'two_cameras_sphere' / 'replay_buffer.zarr',
        'two_cameras_cube': training_dir / 'two_cameras_cube' / 'replay_buffer.zarr',
    }

    grid_info = define_workspace_grid(num_rows=5, num_cols=10)

    print(f"\nWorkspace Grid Configuration:")
    print(f"  Total: {grid_info['total_spots']} spots (5×10 grid)")
    print(f"  Easy: {grid_info['easy_spots_count']} spots (central 4×8)")
    print(f"  Hard: {grid_info['hard_spots_count']} spots (edges)")
    print()

    all_results = {}

    for dataset_name, zarr_path in datasets.items():
        print(f"\nAnalyzing {dataset_name}...")
        print("-" * 80)

        if not zarr_path.exists():
            print(f"  ⚠ Replay buffer not found")
            continue

        all_positions, start_positions = load_training_data(zarr_path)
        print(f"  Loaded {len(all_positions):,} timesteps from {len(start_positions)} episodes")

        density_stats = calculate_region_density(all_positions, grid_info)
        start_density_stats = calculate_region_density(start_positions, grid_info)

        all_results[dataset_name] = {
            'all_timesteps': density_stats,
            'episode_starts': start_density_stats,
            'num_episodes': len(start_positions),
            'num_timesteps': len(all_positions)
        }

        print(f"\n  All Timesteps:")
        print(f"    Easy: {density_stats['easy_count']:,} ({density_stats['easy_percentage']:.1f}%), {density_stats['easy_density']:.1f} visits/m²")
        print(f"    Hard: {density_stats['hard_count']:,} ({density_stats['hard_percentage']:.1f}%), {density_stats['hard_density']:.1f} visits/m²")
        print(f"    Ratio: {density_stats['density_ratio']:.2f}×")

        print(f"\n  Episode Start Positions:")
        print(f"    Easy: {start_density_stats['easy_count']} ({start_density_stats['easy_percentage']:.1f}%)")
        print(f"    Hard: {start_density_stats['hard_count']} ({start_density_stats['hard_percentage']:.1f}%)")
        print(f"    Ratio: {start_density_stats['density_ratio']:.2f}×")
        print()

        plot_density_with_regions(
            all_positions, grid_info, density_stats,
            output_dir / f'{dataset_name}_density_heatmap.png',
            title=f'{dataset_name.replace("_", " ").title()} - Training Data Density (All Timesteps)'
        )

        plot_grid_spots(
            all_positions, grid_info,
            output_dir / f'{dataset_name}_grid_coverage.png',
            title=f'{dataset_name.replace("_", " ").title()} - 50-Spot Grid Coverage'
        )

        # Plot start positions
        plot_grid_spots(
            start_positions, grid_info,
            output_dir / f'{dataset_name}_start_positions.png',
            title=f'{dataset_name.replace("_", " ").title()} - Episode Start Positions'
        )

    def convert_to_serializable(obj):
        """Convert numpy types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_file = output_dir / 'density_analysis_results.json'
    with open(results_file, 'w') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    print(f"\n✓ Saved results to {results_file}")

    print("\n" + "=" * 80)
    print("✓ Analysis complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
