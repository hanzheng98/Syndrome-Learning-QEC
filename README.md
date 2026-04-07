# Syndrome-Learning-QEC

Efficient estimation of logical noise channels from syndrome measurement data under circuit-level noise models for quantum error correction.

Reference:

> [Efficient Learning of Logical Error Channels from Syndrome Data](https://arxiv.org/abs/2601.22286)

## Setup

**Option A: conda**

```bash
conda create -n qec python=3.10 -y
conda activate qec
pip install -r requirements.txt
pip install -e .
```

**Option B: pip (venv)**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

**Optional** (required only for the MLE and ReplayBP decoders):

```bash
pip install gurobipy relay_bp graph_tools
```

## Project Structure

Files marked with `*` are referenced by the demonstration notebook.

```
Syndrome-Learning-QEC/
│
├── demo/
│   ├── qec_learn_syndrome.ipynb *         # Demonstration notebook with benchmark plots
│   ├── learn_circuit_lep_fromsyndrome.py  # Standalone script version of the demonstration
│   └── scaling_records_qec.json           # Precomputed scaling sweep records
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
│   │   ├── dem_sim.py *                   # Circuit construction and detector error model extraction
│   │   ├── circuit_scheduling.py *        # CNOT scheduling via bipartite graph coloring
│   │   ├── circuit_lep_prediction.py *    # Fault prior estimation from syndrome expectation values
│   │   ├── circuit_decoders.py *          # Decoder implementations (BP+LSD, BP+OSD, MLE, ReplayBP)
│   │   └── noise_model.py                # Noise channel injection (depolarizing, measurement, idling)
│   │
│   └── legacy/                            # Archived modules (not used by the current pipeline)
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
| **Step 1** | Construct a CSS code (rotated surface code) |
| **Step 2** | Syndrome extraction under circuit-level noise and detector error model construction |
| **Step 3** | Logical error probability as a function of varying physical error probabilities |
| **Step 4** | Sample complexity analysis: minimum syndrome measurements for ≤10% relative precision |
| **Step 5** | Reproduction of Figure (c): comparison across surface and color code families |
| **Step 6** | Scalable data pipeline with persistent JSON checkpoints |

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

The demonstration constructs a distance-3 rotated surface code, extracts syndrome measurements from a circuit-level noise model, estimates fault priors from the observed syndrome statistics, and compares the predicted logical error probability against direct Monte Carlo estimation.

## Authors

Developed by **Han Zheng** and **Chia-Tung (Andy) Chu**.

## License

See [LICENSE](LICENSE).
