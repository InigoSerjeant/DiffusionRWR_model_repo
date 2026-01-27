# DiffusionRWR Model

A multi-layer graph random walk model for analyzing biological data trajectories across different histone modifications and RNA expression.

## Overview

This package implements a Random Walk with Restart (RWR) model on multi-layer graphs to analyze possible gene causation trajectories that causes a change of cell state.


## Features

- **Multi-layer graph construction** with intra-layer and inter-layer connections
- **Fast RWR implementation** using Numba JIT compilation (20-50x speedup)
- **3D PCA visualization** of visit frequencies and trajectories
- **Flexible edge weight functions** (correlation-based with power transformation)
- **Trajectory analysis** with random sampling and visualization

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd DiffusionRWR_model_repo

# Install dependencies
pip install numpy pandas scikit-learn plotly numba matplotlib
```

## Usage

Run the complete pipeline:

```bash
python -m DiffusionRWR_model_package.main
```

### Configuration

Edit parameters in `main.py`:

- `USE_FAST_RWR`: Toggle between fast (Numba) or slow (detailed) RWR
- `edge_fn_intra`: Edge weight function for intra-layer connections
- `edge_fn_inter`: Edge weight function for inter-layer connections
- `start` / `end`: Basis vectors for start/target nodes
- `n_simulations`: Number of random walk simulations
- `n_trajectories`: Number of trajectories to visualize

## Project Structure

```
DiffusionRWR_model_package/
├── graph_generation/
│   ├── preprocess_data.py          # Data loading and preprocessing
│   ├── generate_graph_internal.py  # Intra-layer graph generation
│   ├── generate_multi_graph.py     # Multi-layer graph assembly
│   └── edge_weight_functions.py    # Edge weight computation
├── run_RWR/
│   ├── slow_RWR.py                 # Standard RWR implementation
│   └── numba_RWR.py                # Fast Numba-optimized RWR
├── model_analysis/
│   └── PCA_frequency_plot.py       # Visualization functions
├── data/
│   └── Modelled/                   # Data files (CSV format)
└── main.py                          # Main pipeline script
```

## Data Format

Input data should be CSV files with:
- Gene names as row indices
- Time points as columns (e.g., '0.0', '1.0', '2.0', '3.0', '4.0')
- Row-wise standardized expression values

## Output

The pipeline generates an interactive 3D Plotly visualization showing:
- Gene nodes colored by dataset (k9me2, k20me3, RNA)
- Node sizes proportional to visit frequency
- Basis vectors (start/target points)
- Sampled random walk trajectories

## Author

Inigo Serjeant
