"""
Learn the logical channel from syndrome information — circuit-level demonstration.

Pipeline:
    1. Build a CSS code (rotated surface code)
    2. Run syndrome extraction circuit and sample
    3. Benchmark sampled LEP vs predicted-prior LEP

Written by Han Zheng
"""

import numpy as np
from sim_qec.pipeline import (
    CSSCode,
    SyndromeExtractionConfig,
    run_syndrome_extraction,
    benchmark_lep,
)
from sim_qec.detector_error_models.dem_sim import CircuitErrorParams


def main():
    # ==================================================================
    # Step 1: Build code
    # ==================================================================
    d = 3
    code = CSSCode.from_rotated_surface_code(d)
    print(f"Code: [[{code.n}, {code.k}, {code.distance}]]")
    print(f"Hx shape: {code.Hx.shape}, Hz shape: {code.Hz.shape}")
    print(f"CSS condition Hx @ Hz^T = 0: {np.all((code.Hx @ code.Hz.T) % 2 == 0)}")

    # ==================================================================
    # Step 2: Build circuit, sample, extract DEM
    # ==================================================================
    config = SyndromeExtractionConfig(
        num_cycles=1,
        physical_error_rate=5e-4,
        shots=5_000_000,
        fault_type="DEPOLARIZE1",
        circuit_error_params=CircuitErrorParams(
            p_i=1.0,
            p_state_p=0.8,
            p_m=0.9,
            p_CX=1.0,
            p_idling_gate=0.0,
        ),
    )

    result = run_syndrome_extraction(code, config)

    print(f"Detectors: {result.detector_error_model.num_detectors}")
    print(f"Observables: {result.detector_error_model.num_observables}")
    print(f"Detector array shape: {result.det_vals.shape}")
    print(f"Observable array shape: {result.log_vals.shape}")
    print(f"Syndrome expectations: {result.syndrome_expectations}")
    print(f"Check matrix shape: {result.check_matrix.shape}, "
          f"num faults: {len(result.channel_probs)}")

    # ==================================================================
    # Step 3: Benchmark — sampled LEP vs predicted-prior LEP
    # ==================================================================
    bench = benchmark_lep(result, max_order=4)

    print("-" * 40)
    print("Prior prediction diagnostics:")
    print(f"  L2 error:             {bench.prior_l2_error:.6e}")
    print(f"  Predicted (first 10): {bench.predicted_priors[:10]}")
    print(f"  True      (first 10): {result.channel_probs[:10]}")
    print("-" * 40)
    print(f"Sampled LEP:   {bench.lep_sampled:.6e}  "
          f"({bench.lep_sampled_runtime:.3f}s)")
    print(f"Predicted LEP: {bench.lep_predicted:.6e}  "
          f"({bench.lep_predicted_runtime:.3f}s)")


if __name__ == "__main__":
    main()
