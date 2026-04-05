# Syndrome-Learning-QEC

Learn logical noise channels from syndrome data using circuit-level quantum error correction simulation.

## Setup

Create a conda environment and install dependencies:

```bash
conda create -n qec python=3.10 -y
conda activate qec

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
│   ├── qec_learn_syndrome.ipynb *        # Jupyter notebook demo
│   └── learn_circuit_lep_fromsyndrome.py # Script version of the demo
│
├── sim_qec/
│   ├── __init__.py                       # bposd sparse-matrix patch
│   ├── pipeline.py *                     # Pipeline API (CSSCode, run_syndrome_extraction, benchmark_lep)
│   │
│   ├── codes_family/
│   │   ├── hpc_lp.py *                   # Rotated surface code, HGP, Lifted Product codes
│   │   ├── classical_codes.py *          # Classical LDPC code generators
│   │   └── est_distance.py              # Code distance estimation
│   │
│   ├── detector_error_models/
│   │   ├── dem_sim.py *                  # Stim circuit builder (DEMSyndromeExtraction, CircuitErrorParams)
│   │   ├── circuit_scheduling.py *       # CX gate scheduling via graph coloring
│   │   ├── circuit_lep_prediction.py *   # Learn fault priors from syndrome expectations (PredictPriors)
│   │   ├── circuit_decoders.py *         # Decoder implementations (BPLSD, BPOSD, MLE, ReplayBP)
│   │   └── noise_model.py               # Noise injection helpers (depolarizing, measurement, idling)
│   │
│   └── legacy/                           # Archived modules (not used by the demo pipeline)
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

## Notebook Call Graph

The notebook uses a 3-stage pipeline. Here is the dependency flow:

```
demo/qec_learn_syndrome.ipynb
  │
  └─ sim_qec/pipeline.py
       │
       │  Stage 1: Build code
       ├─ CSSCode.from_rotated_surface_code(d)
       │    └─ codes_family/hpc_lp.py ── rotated_surface_code_checks()
       │
       │  Stage 2: Circuit + sampling
       ├─ run_syndrome_extraction(code, config) → SyndromeExtractionResult
       │    ├─ detector_error_models/dem_sim.py ── DEMSyndromeExtraction
       │    │    └─ detector_error_models/circuit_scheduling.py ── ColorationCircuit
       │    └─ beliefmatching ── detector_error_model_to_check_matrices()
       │
       │  Stage 3: Benchmark
       └─ benchmark_lep(result) → BenchmarkResult
            ├─ detector_error_models/circuit_lep_prediction.py ── PredictPriors
            └─ detector_error_models/circuit_decoders.py ── BPLSD_Decoder
```

## Running the Demo

```bash
conda activate qec

# Jupyter notebook
jupyter notebook demo/qec_learn_syndrome.ipynb

# Or as a script
python demo/learn_circuit_lep_fromsyndrome.py
```

The demo builds a distance-3 rotated surface code, samples 5M syndrome shots from a circuit-level noise model, learns fault priors from the syndrome statistics, and compares the predicted logical error probability against direct Monte Carlo sampling.

## License

See [LICENSE](LICENSE).
