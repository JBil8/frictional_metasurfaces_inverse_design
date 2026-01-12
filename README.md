# Differentiable Physics for Surface Inverse Design

This project implements a **Scientific Machine Learning (SciML)** pipeline to solve the inverse contact mechanics problem. It uses a **differentiable physics engine** and a Convolutional Neural Network (CNN) to recover microscopic surface topography (asperity heights and shapes) purely from macroscopic contact measurements (Load-Area curves).

## Key Features

* **Differentiable Physics Layer:** A custom PyTorch module (`AxisymmetricContactLayer`) that analytically computes contact response, allowing backpropagation through the physical laws.
* **Hybrid Loss Function:** Trains on both parameter accuracy (MSE) and physical curve reconstruction (Log-Likelihood + Derivative/Stiffness matching).
* **Feature Engineering:** Inputs include **Stiffness** (Load derivative) to explicitly guide the learning of curvature exponents.
* **MLflow Integration:** Full experiment tracking, parameter logging, and real-time visualization of validation curves.

## Installation

1.  **Clone the repository** and navigate to the folder.
2.  **Install dependencies** (Recommend using a virtual environment):
    ```bash
    pip install torch numpy scipy pyyaml matplotlib mlflow
    ```
    *(Note: If you are on a laptop without a GPU, install the CPU-only version of PyTorch to save space.)*

## Quick Start

### 1. Generate Synthetic Data
Run the physics-based generator to create the dataset. This uses Latin Hypercube Sampling (LHS) to ensure good coverage of the parameter space.

```bash
# Must be run as a module to handle imports correctly
python -m data.surface_generator
```
- output: `data/dataset_<n_asperities>_sp.pt`

### 2. train the Inverse Model

Train the neural network. This script automatically handles data loading, training, validation, and testing

```bash
python main_inverse_design.py
```

### 3. Monitor Results

Launch the MLflow dashboard to view loss curves and validation plots in real-time.

```bash
mlflow ui
```


## Project Structure

```plaintext
├── config.yaml              # Global configuration (Physics, Data, Training)
├── main_inverse_design.py   # Main training entry point
├── data/
│   ├── surface_generator.py # Generates Load/Area/Stiffness curves from Physics
│   └── dataset_16_asp.pt    # Generated dataset (ignored by git)
├── physics/
│   └── differentiable.py    # The Differentiable Physics Engine (PyTorch Layer)
├── ml_models/
│   └── model_mlp.py         # CNN Encoder + MLP Decoder architecture
└── utils/
    └── plotting.py          # Visualization utilities
```

## Configuration 

You can tweak the experiment settings in `config.yaml`:

- `physics`: Change `n_asperities` or `E_star` (equivalent Young's Modulus).

    - Note: If you change physics parameters, you MUST regenerate the dataset.

- `data`: Adjust `n_samples` (e.g., 50k for high accuracy) or `n_steps`, wich correpsonds to the indentation steps, to increase the resolution.

- `training`: Modify `batch_size`, `learning_rate`, or `epochs`.