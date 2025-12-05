#!/usr/bin/env python3
"""
Training vs Evaluation Comparison Plots

Overlays training demonstrations with evaluation rollouts for performance comparison.
"""

import sys
from pathlib import Path
from typing import List, Set

import numpy as np
import zarr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
from plot_all_episodes_overlay import REFERENCE_POINTS, get_failure_episodes, add_reference_regions_2d


def load_all_episodes(buffer_path: Path) -> List[np.ndarray]:
    """Load all episodes from replay buffer."""
    store = zarr.open(str(buffer_path), mode='r')
    episode_ends = np.array(store['meta/episode_ends'][:])
    eef_poses = np.array(store['data/robot_eef_pose'][:])

    episodes = []
    for i in range(len(episode_ends)):
        start_idx = 0 if i == 0 else int(episode_ends[i - 1])
        end_idx = int(episode_ends[i])
        trajectory = eef_poses[start_idx:end_idx, :3]
        episodes.append(trajectory)

    return episodes


def plot_training_vs_eval_comparison(
    training_episodes: List[np.ndarray],
    eval_episodes: List[np.ndarray],
    eval_failures: Set[int],
    output_path: Path,
    title: str,
    plane: str = 'xy'
):
    """Plot training demos vs evaluation rollouts comparison."""
    fig, ax = plt.subplots(figsize=(16, 12))

    for trajectory in training_episodes:
        if plane == 'xy':
            x, y = trajectory[:, 1], trajectory[:, 0]
        elif plane == 'xz':
            x, y = trajectory[:, 0], trajectory[:, 2]
        else:
            x, y = trajectory[:, 1], trajectory[:, 2]
        ax.plot(x, y, color='gray', alpha=0.25, linewidth=1.5, zorder=1)

    for trajectory in eval_episodes:
        if plane == 'xy':
            x, y = trajectory[:, 1], trajectory[:, 0]
        elif plane == 'xz':
            x, y = trajectory[:, 0], trajectory[:, 2]
        else:
            x, y = trajectory[:, 1], trajectory[:, 2]
        ax.plot(x, y, color='#0066CC', alpha=0.6, linewidth=1.8, zorder=3)

    if plane == 'xy':
        add_reference_regions_2d(ax)

    num_successes = len(eval_episodes) - len(eval_failures)
    num_failures = len(eval_failures)
    legend_elements = [
        Line2D([0], [0], color='gray', linewidth=2.5, alpha=0.4,
               label=f'Training Demos (n={len(training_episodes)})'),
        Line2D([0], [0], color='#0066CC', linewidth=2.5, alpha=0.6,
               label=f'Evaluation Rollouts (n={len(eval_episodes)}, {num_successes} success, {num_failures} failure)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=0.95)

    # Labels and title
    if plane == 'xy':
        ax.set_xlabel('Y Position (m)', fontsize=14, fontweight='bold')
        ax.set_ylabel('X Position (m)', fontsize=14, fontweight='bold')
    elif plane == 'xz':
        ax.set_xlabel('X Position (m)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Z Position (m)', fontsize=14, fontweight='bold')
    else:
        ax.set_xlabel('Y Position (m)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Z Position (m)', fontsize=14, fontweight='bold')

    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    if plane == 'xy':
        workspace_corners = np.array([
            REFERENCE_POINTS["workspace_p1"][:2],
            REFERENCE_POINTS["workspace_p2"][:2],
            REFERENCE_POINTS["workspace_p3"][:2],
            REFERENCE_POINTS["workspace_p4"][:2],
        ])
        y_coords = workspace_corners[:, 0]
        x_coords = workspace_corners[:, 1]

        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1

        ax.set_xlim(x_max + x_padding, x_min - x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved training vs eval comparison to {output_path}")


def main():
    """Generate training vs evaluation comparison plots."""
    print("=" * 80)
    print("Training vs Evaluation Comparison")
    print("=" * 80)

    # Get repo root directory (parent of scripts/)
    repo_root = Path(__file__).parent.parent
    output_dir = repo_root / "plots"
    training_dir = repo_root / "data" / "training"
    eval_dir = repo_root / "data" / "evaluations"

    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True, parents=True)

    comparisons = [
        ("two_cameras_cube", "eval_two_cameras_cube_up_to_31", "Two Cameras - Cube"),
        ("two_cameras_sphere", "eval_two_cameras_sphere", "Two Cameras - Sphere"),
    ]

    for training_name, eval_name, display_name in comparisons:
        print(f"\n{display_name}")
        print("-" * 80)

        training_path = training_dir / training_name / "replay_buffer.zarr"
        eval_path = eval_dir / eval_name / "replay_buffer.zarr"

        if not training_path.exists():
            print(f"  ⚠ Training data not found")
            continue
        if not eval_path.exists():
            print(f"  ⚠ Eval data not found")
            continue

        print(f"  Loading training data...")
        training_episodes = load_all_episodes(training_path)

        print(f"  Loading evaluation data...")
        eval_episodes = load_all_episodes(eval_path)

        eval_failures = get_failure_episodes(eval_name)

        print(f"  Training: {len(training_episodes)} demos")
        print(f"  Evaluation: {len(eval_episodes)} rollouts")

        output_path = output_dir / f"comparison_train_vs_{eval_name}.png"
        plot_training_vs_eval_comparison(training_episodes, eval_episodes, eval_failures,
                                        output_path, f"{display_name}: Training vs Evaluation", 'xy')

    print("\n" + "=" * 80)
    print("✓ Comparison complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
