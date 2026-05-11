# Differentiable Physics for Surface Inverse Design

This project implements a **Scientific Machine Learning (SciML)** pipeline to solve the highly non-convex inverse contact mechanics problem. It combines a **fully differentiable physics engine** (Sneddon mechanics) with a Deep Neural Network surrogate and L-BFGS optimization to recover microscopic surface topographies (asperity heights and variable-shape exponents) purely from macroscopic contact measurements (Load-Area curves). 

The resulting framework enables the automated design of scale-invariant tribological unit cells (metainterfaces) with programmable static contact area-load curves.

## Key Features

* **Differentiable Physics Layer:** A custom PyTorch module (`AxisymmetricContactLayer`) that analytically computes multi-asperity contact response, allowing the backpropagation of exact physical gradients.
* **Decoupled MLP Architecture:** Separates scale from shape. The network processes normalized Area/Stiffness arrays and logarithmic scale multipliers independently for highly robust zero-shot topological predictions.
* **Curriculum & Contact Regularization:** Utilizes a contact regularization schedule (kappa) and supervised parameter anchoring (lambda)

## Installation

1.  **Clone the repository** and navigate to the folder.
2.  **Install dependencies** (Recommend using a virtual environment):
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: For High-Performance Computing (HPC) environments, ensure OpenMP and MKL thread limits are configured for Tamaas).*

## Quick Start

### 1. Generate the Partitioned Dataset

Run the physics-based generator to create the dataset. This generates mathematically challenging topological sub-domains (LHS, Bimodal, Stratified, Coplanar, Truncated, Mixed) to force the network to learn extreme stiffness transitions.

```bash
python -m data.surface_generator
```

Output: `data/dataset_9_asp.pt`

### 2. Train the Inverse Model

```bash
python main_inverse_design.py
```

(You can monitor the loss curves with `mlfow ui`)

## Project structure 

```text
├── config.yaml                     # Global configuration
├── data/                           # Dataset generation
│   ├── dataset_asp_unitcell.pt
│   └── __init__.py
├── kuma_launcher.sh                # Shell script for submitting jobs to the HPC cluster
├── main_inverse_design.py          # training the neural surrogate
├── ml_models/                      # Neural network architecture and objectives
│   ├── __init__.py
│   ├── loss.py                     # Custom loss functions
│   └── model_mlp.py                # Deep MLP surrogate architecture
├── physics/                        # Contact mechanics engines
│   ├── analytical.py               # Standard analytical Sneddon solutions
│   ├── differentiable.py           # PyTorch differentiable physics
│   ├── __init__.py
│   └── tamas_solution.py           # Tamaas BEM integration
├── README.md                       # Project documentation
├── requirements.txt                # dependencies
├── utils/                          # Core helper functions
│   ├── config.py                   # YAML configuration loader/parser
│   ├── early_stopping.py           # Training logic to halt
│   ├── __init__.py
│   ├── interpolation.py            # Batched 1D interpolation
│   ├── normalization.py            # Local scaling and normalization │   ├── optimizer.py                # Multi-stage L-BFGS topographic refinement
│   └── seeding.py                  # Random seed enforcement 
└── validation/                     # Testing, benchmarking, and plotting scripts
    ├── evaluate_model_perf.py      # General test-set evaluation and error metrics
    ├── __init__.py
    ├── targets.py                  # Generates out-of-distribution targets
    ├── validate_tamaas.py          # Full pipeline: Zero-shot prediction -> L-BFGS -> Tamaas BEM solver
    └── validator.py                # Unified validation class to manage test subsets
```

## Configuration (`config.yaml`)

You can tweak the experiment settings in `config.yaml`:

physics: Define the design space bounds (`n_asperities`, `E_star`, `gamma_min`, `gamma_max`, `delta_max`).
Note: Changing these requires regenerating the dataset.

data: Adjust `n_samples` and interpolation resolution (`n_steps`).

training: Modify batch_size, learning rates, curriculum epochs (lambda), and contact regularization boundaries (kappa_start to kappa_end).

### Acknowledgements 

This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No 945363. J.G.S. and G.C. gratefully acknowledge financial support from the Swiss National Science Foundation, via Ambizione Grant PZ00P2_216341 ``Data-Driven Computational Friction''.
