# Syndrome-Learning-QEC

Learn logical noise channels from syndrome data using circuit-level quantum error correction simulation.

Based on the algorithm described in:

> [Efficient Learning of Logical Error Channels from Syndrome Data](https://arxiv.org/abs/2601.22286)

## Setup

Create a Python 3.10 environment (choose one method), then install dependencies.

### Option A: conda

```bash
conda create -n qec python=3.10 -y
conda activate qec
```

### Option B: built-in venv

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

Install dependencies (same for either environment):

```bash
pip install numpy scipy matplotlib numba networkx galois
pip install stim ldpc bposd beliefmatching pymatching

# Optional (only needed for MLE decoder / ReplayBP decoder)
pip install gurobipy relay_bp graph_tools

# Install this package in development mode
pip install -e .
```

## Project Structure

Files marked with `*` are used by the demo notebook.

```
Syndrome-Learning-QEC/
│
├── demo/
│   ├── qec_learn_syndrome.ipynb *         # Jupyter notebook demo (with benchmark plots)
│   ├── learn_circuit_lep_fromsyndrome.py  # Script version of the demo
│   └── scaling_records_qec.json           # Cached scaling sweep results
│
├── sim_qec/
│   ├── __init__.py                        # bposd sparse-matrix patch
│   ├── pipeline.py *                      # Pipeline API (run_syndrome_extraction, benchmark_lep)
│   │
│   ├── codes_family/
│   │   ├── hpc_lp.py *                    # Rotated surface code, HGP, Lifted Product codes
│   │   ├── classical_codes.py *           # Classical LDPC code generators
│   │   └── est_distance.py               # Code distance estimation
│   │
│   ├── detector_error_models/
│   │   ├── dem_sim.py *                   # Stim circuit builder (DEMSyndromeExtraction, CircuitErrorParams)
│   │   ├── circuit_scheduling.py *        # CX gate scheduling via graph coloring
│   │   ├── circuit_lep_prediction.py *    # Learn fault priors from syndrome expectations (PredictPriors)
│   │   ├── circuit_decoders.py *          # Decoder implementations (BPLSD, BPOSD, MLE, ReplayBP)
│   │   └── noise_model.py                # Noise injection helpers (depolarizing, measurement, idling)
│   │
│   └── legacy/                            # Archived modules (not used by the demo pipeline)
│       ├── analytic_log_channel.py
│       ├── circuit_sim.py
│       ├── decoders.py
│       ├── compute_equiclass.py
│       ├── pauli_character_basis.py
│       ├── utils.py
│       └── walsh_hadamard.py
│
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

## Notebook Structure

| Section | Description |
|---------|-------------|
| **Step 1** | Build a CSS code (rotated surface code) |
| **Step 2** | Syndrome extraction — build noisy stim circuit, sample, extract DEM |
| **Plot 2** | LEP vs physical error rate with error bars (sweep over *p*) |
| **Step 4** | Sample complexity — minimum shots for ≤10% relative precision |
| **Step 5** | Reproduce Plot (c) — fast working example (surface + color codes) |
| **Step 6** | Scalable data pipeline with JSON checkpoints |

## Running the Demo

```bash
# Activate your environment first:
# conda activate qec
# or
# source .venv/bin/activate

# Jupyter notebook
jupyter notebook demo/qec_learn_syndrome.ipynb

# Or as a script
python demo/learn_circuit_lep_fromsyndrome.py
```

The demo builds a distance-3 rotated surface code, samples 5M syndrome shots from a circuit-level noise model, learns fault priors from the syndrome statistics, and compares the predicted logical error probability against direct Monte Carlo sampling.

## Authors

Developed by **Han Zheng** and **Chia-Tung (Andy) Chu**.

## License

See [LICENSE](LICENSE).
