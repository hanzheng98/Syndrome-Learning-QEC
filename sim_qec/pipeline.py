"""
Pipeline module for syndrome-learning QEC experiments.

Provides a clean 2-stage pipeline:
    1. run_syndrome_extraction() — builds circuit, samples, returns SyndromeExtractionResult
    2. benchmark_lep() — compares sampled vs predicted logical error probability

Example usage::

    from bposd.css import css_code
    from sim_qec.codes_family.hpc_lp import rotated_surface_code_checks
    from sim_qec.pipeline import run_syndrome_extraction, benchmark_lep

    Hx, Hz = rotated_surface_code_checks(3)
    code = css_code(Hx, Hz)
    result = run_syndrome_extraction(code)
    bench = benchmark_lep(result, max_order=4)
    print(f"Sampled LEP: {bench.lep_sampled}, Predicted LEP: {bench.lep_predicted}")
    
    See Algorithm 1 in the paper https://arxiv.org/abs/2601.22286 for more details on the pipeline and its stages.
"""

from __future__ import annotations

import time
import numpy as np
import stim
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from bposd.css import css_code
from beliefmatching import detector_error_model_to_check_matrices

from sim_qec.detector_error_models.dem_sim import (
    DEMSyndromeExtraction,
    CircuitErrorParams,
    QubitLayout,
)
from sim_qec.detector_error_models.circuit_lep_prediction import PredictPriors
from sim_qec.detector_error_models.circuit_decoders import BPLSD_Decoder


# ---------------------------------------------------------------------------
# Stage 1: Syndrome extraction (Round 1 of Algorithm 1)
# ---------------------------------------------------------------------------

@dataclass
class SyndromeExtractionConfig:
    """Configuration for a syndrome extraction experiment.

    Attributes:
        num_cycles:           Number of noisy syndrome extraction rounds.
        physical_error_rate:  Global noise scale multiplying all error params.
        shots:                    Number of Monte Carlo samples for the experiment.
        num_samples_true_lep:     Number of shots for ground-truth LEP computation.
        fault_type:               Stim noise instruction (e.g. 'DEPOLARIZE1').
        circuit_error_params:     Per-location error rate weights.
    """
    num_cycles: int = 1
    physical_error_rate: float = 5e-4
    shots: int = 5_000_000
    num_samples_true_lep: int = 100_000_000
    fault_type: str = "DEPOLARIZE1"
    circuit_error_params: CircuitErrorParams = field(
        default_factory=lambda: CircuitErrorParams(
            p_i=1.0, p_state_p=0.8, p_m=0.9, p_CX=0.5, p_idling_gate=0.0,
        )
    )


@dataclass
class SyndromeExtractionResult:
    """Complete output of a syndrome extraction experiment.

    Contains everything needed for downstream prior-learning and decoding,
    plus full stim objects for inspection and reproducibility.
    """
    # Code and config
    code: css_code
    config: SyndromeExtractionConfig

    # Samples
    dem_vals: np.ndarray            # (shots, num_detectors) int
    log_vals: np.ndarray            # (shots, num_observables) int

    # DEM-derived matrices
    check_matrix: np.ndarray        # (num_detectors, num_faults)
    observables_matrix: np.ndarray  # (num_observables, num_faults)
    true_priors: np.ndarray       # (num_faults,) true priors from DEM

    # Syndrome statistics
    syndrome_expectations: np.ndarray  # (num_detectors,) = 1 - 2*mean(dem_vals)

    # Stim objects
    circuit: stim.Circuit
    detector_error_model: object    # stim.DetectorErrorModel
    sampler: object                 # stim.CompiledDetectorSampler

    # Layout and scheduling
    qubit_layout: QubitLayout
    scheduling: Dict[str, List[dict]]


def run_syndrome_extraction(
    code: css_code,
    config: Optional[SyndromeExtractionConfig] = None,
) -> SyndromeExtractionResult:
    """Build a syndrome extraction circuit, sample, and extract the DEM.

    Args:
        code:   A bposd css_code object (constructed from Hx, Hz).
        config: Experiment configuration. Uses defaults if None.

    Returns:
        A SyndromeExtractionResult containing samples, DEM matrices,
        and all stim objects for further analysis.
    """
    if config is None:
        config = SyndromeExtractionConfig()

    # Build circuit
    dem_builder = DEMSyndromeExtraction(
        code=code,
        num_cycles=config.num_cycles,
        circuit_error_params=config.circuit_error_params,
        physical_error_rate=config.physical_error_rate,
    )
    circ = dem_builder.build_circuit(fault_type=config.fault_type)

    # Extract DEM and sampler
    det_model = circ.detector_error_model(flatten_loops=True)
    sampler = circ.compile_detector_sampler()

    # Sample
    dem_vals, log_vals = sampler.sample(
        shots=int(config.shots), separate_observables=True,
    )
    dem_vals = dem_vals.astype(int)
    log_vals = log_vals.astype(int)

    # DEM check matrix and priors
    dem_matrix = detector_error_model_to_check_matrices(
        det_model, allow_undecomposed_hyperedges=True,
    )
    
    h = dem_matrix.check_matrix.toarray()
    l = dem_matrix.observables_matrix.toarray()
    true_priors = dem_matrix.priors

    # Syndrome expectations
    q = dem_vals.mean(axis=0)
    syndrome_expectations = 1.0 - 2.0 * q

    return SyndromeExtractionResult(
        code=code,
        config=config,
        dem_vals=dem_vals,
        log_vals=log_vals,
        check_matrix=h,
        observables_matrix=l,
        true_priors=true_priors,
        syndrome_expectations=syndrome_expectations,
        circuit=circ,
        detector_error_model=det_model,
        sampler=sampler,
        qubit_layout=dem_builder.layout,
        scheduling=dem_builder.scheduling,
    )


# ---------------------------------------------------------------------------
# Stage 2: Benchmark (Round 2 of Algorithm 1)
# ---------------------------------------------------------------------------

DEFAULT_BPLSD_PARAMS = {
    'max_iter': 5,
    'bp_method': 'min_sum',
    'ms_scaling_factor': 0.5,
    'schedule': 'parallel',
    'lsd_method': 'lsd_e',
    'lsd_order': 3,
}


@dataclass
class BenchmarkResult:
    """Comparison of sampled vs predicted logical error probability.

    Attributes:
        lep_sampled:            LEP from direct Monte Carlo decoding.
        lep_sampled_runtime:    Wall-clock time for sampled LEP (seconds).
        lep_predicted:          LEP from syndrome-learned priors.
        lep_predicted_runtime:  Wall-clock time for predicted LEP (seconds).
        predicted_priors:       Learned fault probabilities.
        true_priors:            True fault probabilities from the DEM.
        true_lep:               LEP computed from a large number of samples (ground truth).
        num_samples_true_lep:   Number of shots used to compute true_lep.
    """
    lep_sampled: float
    lep_sampled_runtime: float
    lep_predicted: float
    lep_predicted_runtime: float
    predicted_priors: np.ndarray
    true_priors: np.ndarray
    true_lep: float
    num_samples_true_lep: int


def benchmark_lep(
    result: SyndromeExtractionResult,
    decoder_params: Optional[dict] = None,
    predict_mode: str = 'rip',
    max_order: int = 4,
    subsample_factor: int = 2,
) -> BenchmarkResult:
    """Compare sampled LEP vs predicted-prior LEP.

    Args:
        result:                 Output of run_syndrome_extraction().
        decoder_params:         BPLSD decoder parameters (defaults to DEFAULT_BPLSD_PARAMS).
        predict_mode:           Prior prediction mode ('rip' or 'direct').
        max_order:              Maximum fault weight for predicted LEP enumeration.
        subsample_factor:       Number of stabilizer products per fault for prior learning.
                                Higher values give smoother predicted LEP at the cost of
                                more computation per call. Default 2.

    Returns:
        BenchmarkResult with both LEP values and diagnostics.
    """
    if decoder_params is None:
        decoder_params = DEFAULT_BPLSD_PARAMS

    h = result.check_matrix
    l = result.observables_matrix
    true_priors = result.true_priors
    dem_vals = result.dem_vals
    log_vals = result.log_vals

    # --- Learn priors from syndromes ---
    predictor = PredictPriors(
        dectector_samples=dem_vals,
        check_matrix=h,
        subsample=True,
        subsample_factor=subsample_factor,
    )
    A_syndrome, sample_stabs = predictor._build_A_matrix_syndromes()
    sample_stab_eigs = predictor._get_syndrome_expectations(sample_stabs=sample_stabs)
    predicted_priors = predictor.predict_priors(
        A_syndrome, sample_stab_eigs, mode=predict_mode,
    )


    # Learning the logical error probability (LEP) with the predicted priors, and comparing to the sampled LEP. (Round 3 of Algorithm 1)
    # --- Set up decoder with true priors ---
    decoder = BPLSD_Decoder(BPLSD_params=decoder_params)
    decoder.set_decoder({'H': h, 'L': l, 'channel_probs': true_priors})

    # --- Sampled LEP ---
    t0 = time.perf_counter()
    corrections = decoder.decode(dem_vals)
    les = ((log_vals + (corrections @ l.T) % 2) % 2).any(axis=1).astype(int)
    lep_sampled = float(np.average(les))
    lep_sampled_runtime = time.perf_counter() - t0

    # --- Predicted LEP ---
    t0 = time.perf_counter()
    lep_predicted = predictor.predict_logical_error_efficient(
        decoder=decoder,
        observables_matrix=l,
        priors=predicted_priors,
        max_order=max_order,
    )
    lep_predicted_runtime = time.perf_counter() - t0



    # --- True LEP (ground truth from large sample) ---
    num_samples_true_lep = result.config.num_samples_true_lep
    dem_vals_true, log_vals_true = result.sampler.sample(
        shots=int(num_samples_true_lep), separate_observables=True,
    )
    dem_vals_true = dem_vals_true.astype(int)
    log_vals_true = log_vals_true.astype(int)
    corrections_true = decoder.decode(dem_vals_true)
    les_true = ((log_vals_true + (corrections_true @ l.T) % 2) % 2).any(axis=1).astype(int)
    true_lep = float(np.average(les_true))

    return BenchmarkResult(
        lep_sampled=lep_sampled,
        lep_sampled_runtime=lep_sampled_runtime,
        lep_predicted=lep_predicted,
        lep_predicted_runtime=lep_predicted_runtime,
        predicted_priors=predicted_priors,
        true_priors=true_priors,
        true_lep=true_lep,
        num_samples_true_lep=num_samples_true_lep,
    )
