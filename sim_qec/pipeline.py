"""
Pipeline module for syndrome-learning QEC experiments.

Provides a clean 3-stage pipeline:
    1. CSSCode — wraps (Hx, Hz) parity check matrices with code metadata
    2. run_syndrome_extraction() — builds circuit, samples, returns SyndromeExtractionResult
    3. benchmark_lep() — compares sampled vs predicted logical error probability

Example usage::

    from sim_qec.pipeline import CSSCode, run_syndrome_extraction, benchmark_lep

    code = CSSCode.from_rotated_surface_code(3)
    result = run_syndrome_extraction(code)
    bench = benchmark_lep(result, max_order=4)
    print(f"Sampled LEP: {bench.lep_sampled}, Predicted LEP: {bench.lep_predicted}")
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
# Stage 1: Code family
# ---------------------------------------------------------------------------

@dataclass
class CSSCode:
    """A CSS quantum error-correcting code defined by (Hx, Hz).

    Validates the CSS orthogonality condition Hx @ Hz.T == 0 (mod 2)
    and computes code parameters [[n, k, d]].

    Attributes:
        Hx:             X-type parity check matrix.
        Hz:             Z-type parity check matrix.
        distance:       Code distance (user-supplied; None if unknown).
        n:              Number of physical qubits.
        k:              Number of logical qubits (via matrix rank).
        num_x_checks:   Number of X stabilizer generators (rows of Hx).
        num_z_checks:   Number of Z stabilizer generators (rows of Hz).
    """
    Hx: np.ndarray
    Hz: np.ndarray
    distance: Optional[int] = None

    # Computed fields
    n: int = field(init=False)
    k: int = field(init=False)
    num_x_checks: int = field(init=False)
    num_z_checks: int = field(init=False)

    # Lazy cache (not shown in repr)
    _css_code_obj: Optional[css_code] = field(init=False, default=None, repr=False)

    def __post_init__(self):
        self.Hx = np.asarray(self.Hx, dtype=np.uint8)
        self.Hz = np.asarray(self.Hz, dtype=np.uint8)
        if self.Hx.shape[1] != self.Hz.shape[1]:
            raise ValueError(
                f"Hx and Hz must have the same number of columns, "
                f"got {self.Hx.shape[1]} and {self.Hz.shape[1]}"
            )
        if not np.all((self.Hx @ self.Hz.T) % 2 == 0):
            raise ValueError("CSS condition violated: Hx @ Hz^T != 0 mod 2")
        self.n = self.Hx.shape[1]
        self.num_x_checks = self.Hx.shape[0]
        self.num_z_checks = self.Hz.shape[0]
        rank_hx = int(np.linalg.matrix_rank(self.Hx))
        rank_hz = int(np.linalg.matrix_rank(self.Hz))
        self.k = self.n - rank_hx - rank_hz

    @property
    def bposd_code(self) -> css_code:
        """Lazily build and cache the bposd css_code object."""
        if self._css_code_obj is None:
            self._css_code_obj = css_code(self.Hx, self.Hz)
        return self._css_code_obj

    @classmethod
    def from_rotated_surface_code(cls, d: int) -> CSSCode:
        """Build a rotated surface code of odd distance d >= 3."""
        from sim_qec.codes_family.hpc_lp import rotated_surface_code_checks
        Hx, Hz = rotated_surface_code_checks(d)
        return cls(Hx=Hx, Hz=Hz, distance=d)

    @classmethod
    def from_matrices(cls, Hx: np.ndarray, Hz: np.ndarray,
                      distance: Optional[int] = None) -> CSSCode:
        """Build from raw parity-check matrices."""
        return cls(Hx=Hx, Hz=Hz, distance=distance)


# ---------------------------------------------------------------------------
# Stage 2: Syndrome extraction
# ---------------------------------------------------------------------------

@dataclass
class SyndromeExtractionConfig:
    """Configuration for a syndrome extraction experiment.

    Attributes:
        num_cycles:           Number of noisy syndrome extraction rounds.
        physical_error_rate:  Global noise scale multiplying all error params.
        shots:                Number of Monte Carlo samples.
        fault_type:           Stim noise instruction (e.g. 'DEPOLARIZE1').
        circuit_error_params: Per-location error rate weights.
    """
    num_cycles: int = 1
    physical_error_rate: float = 5e-4
    shots: int = 5_000_000
    fault_type: str = "DEPOLARIZE1"
    circuit_error_params: CircuitErrorParams = field(
        default_factory=lambda: CircuitErrorParams(
            p_i=1.0, p_state_p=0.8, p_m=0.9, p_CX=1.0, p_idling_gate=0.0,
        )
    )


@dataclass
class SyndromeExtractionResult:
    """Complete output of a syndrome extraction experiment.

    Contains everything needed for downstream prior-learning and decoding,
    plus full stim objects for inspection and reproducibility.
    """
    # Code and config
    code: CSSCode
    config: SyndromeExtractionConfig

    # Samples
    det_vals: np.ndarray            # (shots, num_detectors) int
    log_vals: np.ndarray            # (shots, num_observables) int

    # DEM-derived matrices
    check_matrix: np.ndarray        # (num_detectors, num_faults)
    observables_matrix: np.ndarray  # (num_observables, num_faults)
    channel_probs: np.ndarray       # (num_faults,) true priors from DEM

    # Syndrome statistics
    syndrome_expectations: np.ndarray  # (num_detectors,) = 1 - 2*mean(det_vals)

    # Stim objects
    circuit: stim.Circuit
    detector_error_model: object    # stim.DetectorErrorModel
    sampler: object                 # stim.CompiledDetectorSampler

    # Layout and scheduling
    qubit_layout: QubitLayout
    scheduling: Dict[str, List[dict]]


def run_syndrome_extraction(
    code: CSSCode,
    config: Optional[SyndromeExtractionConfig] = None,
) -> SyndromeExtractionResult:
    """Build a syndrome extraction circuit, sample, and extract the DEM.

    Args:
        code:   The CSS code to simulate.
        config: Experiment configuration. Uses defaults if None.

    Returns:
        A SyndromeExtractionResult containing samples, DEM matrices,
        and all stim objects for further analysis.
    """
    if config is None:
        config = SyndromeExtractionConfig()

    # Build circuit
    dem_builder = DEMSyndromeExtraction(
        code=code.bposd_code,
        num_cycles=config.num_cycles,
        circuit_error_params=config.circuit_error_params,
        physical_error_rate=config.physical_error_rate,
    )
    circ = dem_builder.build_circuit(fault_type=config.fault_type)

    # Extract DEM and sampler
    det_model = circ.detector_error_model(flatten_loops=True)
    sampler = circ.compile_detector_sampler()

    # Sample
    det_vals, log_vals = sampler.sample(
        shots=config.shots, separate_observables=True,
    )
    det_vals = det_vals.astype(int)
    log_vals = log_vals.astype(int)

    # DEM check matrix and priors
    dem_matrix = detector_error_model_to_check_matrices(
        det_model, allow_undecomposed_hyperedges=True,
    )
    h = dem_matrix.check_matrix.toarray()
    l = dem_matrix.observables_matrix.toarray()
    channel_probs = dem_matrix.priors

    # Syndrome expectations
    q = det_vals.mean(axis=0)
    syndrome_expectations = 1.0 - 2.0 * q

    return SyndromeExtractionResult(
        code=code,
        config=config,
        det_vals=det_vals,
        log_vals=log_vals,
        check_matrix=h,
        observables_matrix=l,
        channel_probs=channel_probs,
        syndrome_expectations=syndrome_expectations,
        circuit=circ,
        detector_error_model=det_model,
        sampler=sampler,
        qubit_layout=dem_builder.layout,
        scheduling=dem_builder.scheduling,
    )


# ---------------------------------------------------------------------------
# Stage 3: Benchmark
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
        prior_l2_error:         L2 norm between predicted and true priors.
    """
    lep_sampled: float
    lep_sampled_runtime: float
    lep_predicted: float
    lep_predicted_runtime: float
    predicted_priors: np.ndarray
    prior_l2_error: float


def benchmark_lep(
    result: SyndromeExtractionResult,
    decoder_params: Optional[dict] = None,
    predict_mode: str = 'rip',
    max_order: int = 4,
) -> BenchmarkResult:
    """Compare sampled LEP vs predicted-prior LEP.

    Args:
        result:         Output of run_syndrome_extraction().
        decoder_params: BPLSD decoder parameters (defaults to DEFAULT_BPLSD_PARAMS).
        predict_mode:   Prior prediction mode ('rip' or 'direct').
        max_order:      Maximum fault weight for predicted LEP enumeration.

    Returns:
        BenchmarkResult with both LEP values and diagnostics.
    """
    if decoder_params is None:
        decoder_params = DEFAULT_BPLSD_PARAMS

    h = result.check_matrix
    l = result.observables_matrix
    channel_probs = result.channel_probs
    det_vals = result.det_vals
    log_vals = result.log_vals

    # --- Learn priors from syndromes ---
    predictor = PredictPriors(
        dectector_samples=det_vals,
        check_matrix=h,
        subsample=True,
    )
    A_syndrome, sample_stabs = predictor._build_A_matrix_syndromes()
    sample_stab_eigs = predictor._get_syndrome_expectations(sample_stabs=sample_stabs)
    predicted_priors = predictor.predict_priors(
        A_syndrome, sample_stab_eigs, mode=predict_mode,
    )

    # --- Set up decoder with true priors ---
    decoder = BPLSD_Decoder(BPLSD_params=decoder_params)
    decoder.set_decoder({'H': h, 'L': l, 'channel_probs': channel_probs})

    # --- Sampled LEP ---
    t0 = time.perf_counter()
    corrections = decoder.decode(det_vals)
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

    # --- Diagnostics ---
    prior_l2_error = float(np.linalg.norm(
        np.asarray(predicted_priors) - np.asarray(channel_probs)
    ))

    return BenchmarkResult(
        lep_sampled=lep_sampled,
        lep_sampled_runtime=lep_sampled_runtime,
        lep_predicted=lep_predicted,
        lep_predicted_runtime=lep_predicted_runtime,
        predicted_priors=predicted_priors,
        prior_l2_error=prior_l2_error,
    )
