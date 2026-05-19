# Triangle Force Dataset Pipeline

This repository contains the clean dataset-generation pipeline for the triangle force project.

## Installation

Create and activate a virtual environment:

python -m venv .venv
.\.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

## Generate dataset

Generate accepted samples:

python .\main_generate_dataset_loop.py

Build the PyTorch dataset:

python .\main_build_pytorch_dataset.py

Check dataset integrity:

python .\main_check_dataset_integrity.py

## Final output

The final PyTorch dataset is saved as:

outputs/pytorch_dataset/triangle_force_dataset.npz

It contains:

X     input features: x_ref_px, y_ref_px, dx_px, dy_px, rotation_deg
y     target forces: normal_force, shear_force_x, shear_force_y
mask  valid triangle mask

## Important

Do not commit generated files from the outputs folder.
The outputs folder is ignored by Git.
