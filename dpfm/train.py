"""
Training script for Flow Matching Diffusion Policy.

Usage:
    # Image-only policy (legacy):
    python -m dpfm.train --config-name=train_fm_unet_image_workspace
    
    # Hybrid policy (paper-aligned, recommended):
    python -m dpfm.train --config-name=train_fm_unet_hybrid_image_workspace
    
    # With overrides:
    python -m dpfm.train --config-name=train_fm_unet_hybrid_image_workspace \
        policy.num_inference_steps=4 \
        training.seed=42 \
        optimizer.lr=1e-4
"""

import sys
import os
import pathlib

# Add project root to path
ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.insert(0, ROOT_DIR)

# Add diffusion_policy to path
DP_DIR = os.path.join(ROOT_DIR, 'diffusion_policy')
sys.path.insert(0, DP_DIR)

import hydra
from omegaconf import OmegaConf

# Import workspaces
from dpfm.workspace.train_fm_unet_image_workspace import TrainFlowMatchingUnetImageWorkspace
from dpfm.workspace.train_fm_unet_hybrid_image_workspace import TrainFlowMatchingUnetHybridImageWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="train_fm_unet_hybrid_image_workspace"  # Default to hybrid (paper-aligned)
)
def main(cfg):
    # Change to diffusion_policy directory for relative data paths AFTER hydra init
    os.chdir(DP_DIR)
    
    # Get the workspace class from config target
    cls = hydra.utils.get_class(cfg._target_)
    # Instantiate workspace with config
    workspace = cls(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
