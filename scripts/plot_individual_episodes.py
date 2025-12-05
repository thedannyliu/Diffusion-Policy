#!/usr/bin/env python3
"""
Individual Episode Visualization Script

Generates visualizations showing trajectory plot and key camera frames.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import zarr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from plot_all_episodes_overlay import REFERENCE_POINTS, EVAL_FAILURES, add_reference_regions_2d


def load_episode_trajectory(buffer_path: Path, episode_idx: int) -> np.ndarray:
    """Load trajectory for a specific episode."""
    store = zarr.open(str(buffer_path), mode='r')

    episode_ends = np.array(store['meta/episode_ends'][:])
    eef_poses = np.array(store['data/robot_eef_pose'][:])

    # Get start and end indices for this episode
    start_idx = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1])
    end_idx = int(episode_ends[episode_idx])

    # Extract x, y, z positions (first 3 columns)
    trajectory = eef_poses[start_idx:end_idx, :3]

    return trajectory


def extract_video_frames(video_path: Path, num_frames: int = 8,
                         fail_frame_ratio: Optional[float] = None) -> List[np.ndarray]:
    """Extract evenly spaced frames from video, optionally focusing on failure moments."""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Warning: Could not open video {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        print(f"Warning: Video {video_path} has 0 frames")
        cap.release()
        return []

    if fail_frame_ratio is not None:
        fail_frame_idx = int(fail_frame_ratio * (total_frames - 1))
        indices = []
        frames_before = num_frames // 2
        frames_after = num_frames - frames_before - 1

        if frames_before > 0:
            indices.extend(np.linspace(0, fail_frame_idx, frames_before, dtype=int))
        indices.append(fail_frame_idx)
        if frames_after > 0:
            indices.extend(np.linspace(fail_frame_idx + 1, total_frames - 1, frames_after, dtype=int))

        indices = sorted(set(indices))
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    return frames


def plot_episode_with_frames(
    trajectory: np.ndarray,
    camera_frames: List[List[np.ndarray]],
    output_path: Path,
    episode_idx: int,
    eval_name: str,
    is_failure: bool = False,
    failure_reason: str = None,
    num_frames_to_show: int = 8
):
    """Create visualization with trajectory plot and camera frames."""
    num_cameras = len(camera_frames)
    fig = plt.figure(figsize=(20, 8 + 4 * num_cameras))
    gs = GridSpec(1 + num_cameras, 1, height_ratios=[8] + [4] * num_cameras,
                  hspace=0.3, figure=fig)

    ax_traj = fig.add_subplot(gs[0])
    idx1, idx2 = 1, 0
    x_data, y_data = trajectory[:, idx1], trajectory[:, idx2]

    ax_traj.plot(x_data, y_data, 'b-', linewidth=2.5, alpha=0.7, label='Trajectory')
    ax_traj.scatter(x_data[0], y_data[0], c='green', s=200, marker='o',
                   alpha=0.9, edgecolors='darkgreen', linewidths=3, zorder=5, label='Start')

    if is_failure:
        ax_traj.scatter(x_data[-1], y_data[-1], c='red', s=250, marker='X',
                       alpha=0.9, edgecolors='darkred', linewidths=3, zorder=5, label='End (Failure)')
    else:
        ax_traj.scatter(x_data[-1], y_data[-1], c='black', s=200, marker='*',
                       alpha=0.9, edgecolors='black', linewidths=3, zorder=5, label='End')

    add_reference_regions_2d(ax_traj, plane='xy', eval_name=eval_name)

    ax_traj.set_xlabel('X Position (m)', fontsize=14, fontweight='bold')
    ax_traj.set_ylabel('Y Position (m)', fontsize=14, fontweight='bold')

    title = f"{eval_name} - Episode {episode_idx} (XY Plane)"
    if is_failure and failure_reason:
        title += f"\nFailure Mode: {failure_reason}"
    ax_traj.set_title(title, fontsize=16, fontweight='bold', pad=15)

    ax_traj.grid(True, alpha=0.3)
    ax_traj.set_aspect('equal', adjustable='box')
    ax_traj.invert_xaxis()
    ax_traj.legend(loc='best', fontsize=11, framealpha=0.9)

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

    ax_traj.set_xlim(x_max + x_padding, x_min - x_padding)
    ax_traj.set_ylim(y_min - y_padding, y_max + y_padding)

    for cam_idx, frames in enumerate(camera_frames):
        ax_frames = fig.add_subplot(gs[1 + cam_idx])
        ax_frames.axis('off')

        frames_to_display = frames[:num_frames_to_show]
        if len(frames_to_display) == 0:
            continue

        composite_height = frames_to_display[0].shape[0]
        composite_width = sum(f.shape[1] for f in frames_to_display)
        composite = np.zeros((composite_height, composite_width, 3), dtype=np.uint8)

        x_offset = 0
        for i, frame in enumerate(frames_to_display):
            frame_width = frame.shape[1]
            composite[:, x_offset:x_offset + frame_width] = frame

            text = str(i + 1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, 1.5, 3)[0]
            text_x, text_y = x_offset + 10, 40

            cv2.rectangle(composite,
                         (text_x - 5, text_y - text_size[1] - 5),
                         (text_x + text_size[0] + 5, text_y + 5),
                         (0, 0, 0), -1)
            cv2.putText(composite, text, (text_x, text_y), font, 1.5, (255, 255, 255), 3)

            x_offset += frame_width

        ax_frames.imshow(composite)
        ax_frames.set_title(f"Camera {cam_idx} Frames", fontsize=14, fontweight='bold', pad=10)

    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved episode visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate individual episode visualizations')
    parser.add_argument('eval_dir', type=str, help='Path to evaluation directory')
    parser.add_argument('--episodes', type=str, required=True,
                       help='Comma-separated episode indices (e.g., "0,5,10" or "all")')
    parser.add_argument('--num-frames', type=int, default=8, help='Frames per camera (default: 8)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: eval_dir/episode_visualizations)')
    parser.add_argument('--failure-moments', action='store_true',
                       help='Focus on failure moments for failed episodes')
    parser.add_argument('--failure-frame-ratio', type=float, default=0.8,
                       help='Failure focus position 0.0-1.0 (default: 0.8)')

    args = parser.parse_args()
    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        print(f"Error: Directory {eval_dir} does not exist")
        return 1

    replay_buffer_path = eval_dir / 'replay_buffer.zarr'
    videos_dir = eval_dir / 'videos'

    if not replay_buffer_path.exists():
        print(f"Error: Replay buffer not found at {replay_buffer_path}")
        return 1

    if not videos_dir.exists():
        print(f"Error: Videos directory not found")
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else eval_dir / 'episode_visualizations'
    output_dir.mkdir(exist_ok=True, parents=True)

    if args.episodes.lower() == 'all':
        store = zarr.open(str(replay_buffer_path), mode='r')
        episode_indices = list(range(len(store['meta/episode_ends'][:])))
    else:
        episode_indices = [int(x.strip()) for x in args.episodes.split(',')]

    eval_name = eval_dir.name
    failure_episodes = set()
    for key, failures in EVAL_FAILURES.items():
        if key in eval_name or eval_name in key:
            failure_episodes = set(failures)
            break

    failure_reasons = {
        'eval_two_cameras_sphere': {
            32: 'rubbing',
            36: 'pushing out of boundary',
            49: 'pushing out of boundary',
        },
        'eval_two_camera_cube': {
            32: 'pushing out of boundary',
            33: 'rubbing',
            42: 'last rotation lacking',
            50: 'brute forcing',
        },
        'eval_one_camera_sphere': {},
        'eval_one_camera_cube': {
            10: 'brute forcing',
            23: 'stuck at frame',
        },
    }

    failure_reason_map = {}
    for key, reasons in failure_reasons.items():
        if key in eval_name or eval_name in key:
            failure_reason_map = reasons
            break

    print(f"\nProcessing {len(episode_indices)} episodes from {eval_name}")
    print(f"Output: {output_dir}\n")

    for ep_idx in episode_indices:
        print(f"Processing episode {ep_idx}...")

        try:
            trajectory = load_episode_trajectory(replay_buffer_path, ep_idx)
        except Exception as e:
            print(f"  Error: {e}")
            continue

        is_failure = ep_idx in failure_episodes
        failure_reason = failure_reason_map.get(ep_idx, "unknown") if is_failure else None

        episode_video_dir = videos_dir / str(ep_idx)
        if not episode_video_dir.exists():
            print(f"  Warning: Video directory not found")
            continue

        video_files = sorted(episode_video_dir.glob('*.mp4'))
        if len(video_files) == 0:
            print(f"  Warning: No video files found")
            continue

        all_camera_frames = []
        for video_file in video_files:
            if is_failure and args.failure_moments:
                frames = extract_video_frames(video_file, args.num_frames, args.failure_frame_ratio)
            else:
                frames = extract_video_frames(video_file, args.num_frames)
            all_camera_frames.append(frames)

        output_path = output_dir / f"episode_{ep_idx}_visualization.png"
        plot_episode_with_frames(trajectory, all_camera_frames, output_path, ep_idx, eval_name,
                                is_failure, failure_reason, args.num_frames)

    print(f"\n✓ Visualization complete! Saved {len(episode_indices)} visualizations")
    return 0


if __name__ == '__main__':
    sys.exit(main())
