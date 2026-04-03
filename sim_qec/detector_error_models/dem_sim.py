"""
Detector-error-model (DEM) based syndrome extraction simulation.

This module builds stim circuits for memory experiments on CSS codes,
samples detector/observable outcomes, and provides the interface for
computing syndrome expectation values and logical error rates.

Circuit structure (memory experiment):
    1. Initialization — reset data + ancilla qubits.
    2. Repeated noisy syndrome extraction rounds (num_cycles times).
    3. Terminal perfect stabilizer measurement (via MPP) or transversal
       data-qubit measurement, producing detectors and observables.

Noise is injected at five configurable locations per round; see
CircuitErrorParams for details.
"""

import stim
import numpy as np
import copy
from dataclasses import dataclass, field
from bposd.css import css_code
from typing import Dict, List, Optional, Union, Tuple
from sim_qec.detector_error_models.circuit_scheduling import ColorationCircuit


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CircuitErrorParams:
    """Per-location error probabilities for the syndrome extraction circuit.

    Each field is a bare rate in [0, 1].  When a ``physical_error_rate`` *p*
    is supplied to :class:`BaseDEMSim`, every field is multiplied by *p* so
    that the relative weights are preserved while the overall noise scale is
    controlled by a single knob.

    Attributes:
        p_i:            Single-qubit idling error probability.
                        Applied to every qubit that is idle during a time step
                        (i.e. not involved in a gate in that step).
        p_state_p:      State-preparation error probability.
                        Applied to data qubits immediately after the initial
                        reset gate (R / RX).
        p_m:            Measurement error probability.
                        Applied to ancilla qubits just before each MR
                        instruction, or to data qubits before the final
                        transversal measurement.
        p_CX:           Two-qubit gate (CX) depolarizing error probability.
                        Applied as DEPOLARIZE2 on each (control, target) pair
                        after every CX gate in each scheduling time step.
        p_idling_gate:  Idling error during two-qubit gate time steps.
                        Applied to data qubits that are *not* involved in a CX
                        gate during a given scheduling time step (CSS mode only).
    """
    p_i: float = 0.0
    p_state_p: float = 0.0
    p_m: float = 0.0
    p_CX: float = 0.0
    p_idling_gate: float = 0.0

    def scaled(self, 
               physical_error_rate: float) -> 'CircuitErrorParams':
        """Return a new instance with all rates multiplied by *physical_error_rate*."""
        return CircuitErrorParams(
            p_i=self.p_i * physical_error_rate,
            p_state_p=self.p_state_p * physical_error_rate,
            p_m=self.p_m * physical_error_rate,
            p_CX=self.p_CX * physical_error_rate,
            p_idling_gate=self.p_idling_gate * physical_error_rate,
        )


@dataclass
class QubitLayout:
    """Qubit-index assignments for the syndrome extraction circuit.

    Stim circuits use a flat integer index space.  This layout assigns
    three contiguous blocks::

        [0, n)                                      -> data qubits
        [n, n + n_Z_ancilla)                        -> Z-type ancilla qubits
        [n + n_Z_ancilla, n + n_Z_ancilla + n_X_ancilla) -> X-type ancilla qubits

    Attributes:
        n:                  Number of data (physical) qubits  (= columns of Hx / Hz).
        n_Z_ancilla:        Number of Z-stabilizer ancilla qubits (= rows of Hz).
        n_X_ancilla:        Number of X-stabilizer ancilla qubits (= rows of Hx).
        data_indices:       ``list(range(n))``.
        Z_ancilla_indices:  Index list for Z ancillas.
        X_ancilla_indices:  Index list for X ancillas.
    """
    n: int
    n_Z_ancilla: int
    n_X_ancilla: int
    data_indices: List[int] = field(init=False, repr=False)
    Z_ancilla_indices: List[int] = field(init=False, repr=False)
    X_ancilla_indices: List[int] = field(init=False, repr=False)

    def __post_init__(self):
        self.data_indices = list(range(self.n))
        self.Z_ancilla_indices = list(range(self.n, self.n + self.n_Z_ancilla))
        self.X_ancilla_indices = list(range(
            self.n + self.n_Z_ancilla,
            self.n + self.n_Z_ancilla + self.n_X_ancilla,
        ))


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseDEMSim:
    """Base class for DEM-based simulation of CSS quantum error-correcting codes.

    This class owns the code object, the noise parameters, and the qubit
    layout.  It provides reusable building blocks shared by every concrete
    circuit builder (e.g. full CSS syndrome extraction, repetition-code
    demo, etc.):

    * :meth:`_initialize_circuit` — reset layer.
    * :meth:`_ideal_sec_round` — noiseless stabilizer measurement via MPP.
    * :meth:`_transversal_measurement` — final data-qubit measurement with
      detector and observable wiring.
    * :meth:`_add_one_qubit_fault` — inject single-qubit noise at a given
      circuit location.
    """

    def __init__(self,
                 code: css_code,
                 num_cycles: int,
                 circuit_error_params: Union[dict, CircuitErrorParams],
                 physical_error_rate: Optional[float] = None,
                 ):
        self.eval_code = code
        self.num_cycles = num_cycles

        # Accept both a plain dict (backward compat) and the dataclass.
        if isinstance(circuit_error_params, dict):
            circuit_error_params = CircuitErrorParams(**circuit_error_params)

        # Scale all rates by the global physical error rate if provided.
        if physical_error_rate is not None:
            circuit_error_params = circuit_error_params.scaled(physical_error_rate)

        self.error_params = circuit_error_params

        # Build the qubit layout from the code dimensions.
        self.layout = QubitLayout(
            n=code.hx.shape[1],
            n_Z_ancilla=code.hz.shape[0],
            n_X_ancilla=code.hx.shape[0],
        )

        # Convenience aliases so that existing code referencing
        # self.data_indices / self.Z_ancilla_indices / etc. still works.
        self.data_indices = self.layout.data_indices
        self.n = self.layout.n
        self.Z_ancilla_indices = self.layout.Z_ancilla_indices
        self.X_ancilla_indices = self.layout.X_ancilla_indices

        # Backward-compat alias: some call sites use self.circuit_error_params['key'].
        # We expose a dict view so that dict-style access keeps working.
        self.circuit_error_params = self.error_params.__dict__

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize_circuit(self,
                            circuit: stim.Circuit,
                            reset: Union[str, None],
                            do_syndrome: bool = False,
                            mode: str = 'repetition',
                            ) -> stim.Circuit:
        """Append the initialization layer to *circuit*.

        Args:
            circuit:      The stim circuit being built (modified in-place).
            reset:        Reset instruction to apply to data qubits, e.g.
                          ``"R"`` (Z-basis) or ``"RX"`` (X-basis).
                          If ``None``, no reset is performed and a noiseless
                          syndrome round is used instead (requires
                          ``do_syndrome=True``).
            do_syndrome:  If ``True`` *and* ``reset is None``, perform an
                          ideal syndrome extraction round (MPP-based) to
                          establish the initial stabilizer eigenvalues.
            mode:         ``'repetition'`` — only Z ancillas are initialized.
                          ``'CSS'`` — both X and Z ancillas are initialized.

        Returns:
            The same *circuit* object (for chaining convenience).
        """
        data_indices = self.data_indices
        Z_ancilla_indices = self.Z_ancilla_indices
        X_ancilla_indices = self.X_ancilla_indices

        if reset is not None:
            # Reset data qubits in the requested basis.
            circuit.append(reset, data_indices)
            # Ancilla qubits are always reset in Z basis.
            if mode == 'CSS':
                circuit.append("R", X_ancilla_indices + Z_ancilla_indices)
            elif mode == 'repetition':
                circuit.append("R", Z_ancilla_indices)
        else:
            if not do_syndrome:
                raise ValueError("If no reset is specified, do_syndrome must be True")
            # Use a perfect MPP round to establish initial eigenvalues.
            self._ideal_sec_round(circuit, data_indices, pairwise_diff=False)

        return circuit

    # ------------------------------------------------------------------
    # Ideal (noiseless) syndrome extraction via MPP
    # ------------------------------------------------------------------

    def _ideal_sec_round(self,
                         circuit: stim.Circuit,
                         data_indices: List[int],
                         pairwise_diff: bool = True,
                         ) -> stim.Circuit:
        """Append a *perfect* stabilizer measurement round using MPP.

        MPP (multi-Pauli product measurement) measures each stabilizer
        generator in a single instruction without ancilla qubits, so it
        is noiseless by construction.  This is used for:

        * The boundary/initial round when no explicit reset is given.
        * The terminal perfect round that anchors the last noisy round's
          detectors.

        Measurement record layout after this method:
            The MPP block appends ``num_z_checks`` Z-stabilizer results
            followed by ``num_x_checks`` X-stabilizer results to the
            measurement record.

        Detector wiring:
            ``pairwise_diff=True`` — each detector compares the terminal
            MPP result with the corresponding ancilla MR result from the
            *previous* noisy round.  This is the normal case for rounds
            after the first.

            ``pairwise_diff=False`` — each detector references only the
            MPP result itself (boundary anchoring for the very first
            round or when no prior round exists).

        Args:
            circuit:        The stim circuit (modified in-place).
            data_indices:   Indices of data qubits to measure.
            pairwise_diff:  Whether to wire detectors as pairwise XOR with
                            the previous round's measurements.

        Returns:
            The same *circuit* object.
        """
        hx, hz = self.eval_code.hx, self.eval_code.hz
        num_x_checks = hx.shape[0]
        num_z_checks = hz.shape[0]

        # --- Z-stabilizer MPP products ---
        z_terms = []
        for r in range(num_z_checks):
            cols = np.flatnonzero(hz[r])
            prod = None
            for c in cols:
                t = stim.target_z(int(data_indices[c]))
                prod = t if prod is None else (prod * t)
            z_terms.append(prod)
        if z_terms:
            circuit.append("MPP", z_terms)

        # --- X-stabilizer MPP products ---
        x_terms = []
        for r in range(num_x_checks):
            cols = np.flatnonzero(hx[r])
            prod = None
            for c in cols:
                t = stim.target_x(int(data_indices[c]))
                prod = t if prod is None else (prod * t)
            x_terms.append(prod)
        if x_terms:
            circuit.append("MPP", x_terms)

        # --- Detector wiring ---
        # After the two MPP blocks the measurement record has gained
        # num_z_checks + num_x_checks new results (Z first, then X).
        total_new = num_x_checks + num_z_checks

        if pairwise_diff:
            # XOR each terminal MPP result with the matching ancilla MR
            # from the previous noisy round (which sits 2*total_new back
            # in the measurement record).
            for j in range(num_z_checks):
                circuit.append(
                    "DETECTOR",
                    [
                        stim.target_rec(-total_new + j),              # terminal Z_j
                        stim.target_rec(-2 * total_new + j),          # previous round Z_j
                    ],
                    0,
                )
            for j in range(num_x_checks):
                circuit.append(
                    "DETECTOR",
                    [
                        stim.target_rec(-num_x_checks + j),           # terminal X_j
                        stim.target_rec(-2 * total_new + num_z_checks + j),  # previous round X_j
                    ],
                    0,
                )
        else:
            # Boundary anchoring: detector references only the MPP result.
            for j in range(num_z_checks):
                circuit.append("DETECTOR", [stim.target_rec(-total_new + j)])
            for j in range(num_x_checks):
                circuit.append("DETECTOR", [stim.target_rec(-num_x_checks + j)])
            circuit.append("TICK")

        return circuit

    # ------------------------------------------------------------------
    # Final transversal data-qubit measurement
    # ------------------------------------------------------------------

    def _transversal_measurement(self,
                                 circuit: stim.Circuit,
                                 basis: str = 'Z',
                                 add_faults: bool = False,
                                 pairwise_diff: bool = True,
                                 ) -> stim.Circuit:
        """Append the final transversal measurement of all data qubits.

        This is the last layer of the memory experiment.  Each data qubit
        is measured individually (``M`` for Z-basis, ``MX`` for X-basis),
        and detectors are wired by computing the parity of the data-qubit
        outcomes that form each stabilizer, optionally XOR-ed with the
        matching ancilla result from the last noisy round.

        Logical observables (``OBSERVABLE_INCLUDE``) are emitted for each
        row of Lz (Z-basis) or Lx (X-basis).

        Args:
            circuit:        The stim circuit (modified in-place).
            basis:          ``'Z'`` or ``'X'``.
            add_faults:     If ``True``, inject noise on data qubits before
                            the measurement (using ``p_m``).
            pairwise_diff:  If ``True``, each detector also references the
                            corresponding ancilla measurement from the last
                            noisy round.

        Returns:
            The same *circuit* object.
        """
        hx = self.eval_code.hx
        hz = self.eval_code.hz
        lz = self.eval_code.lz
        lx = self.eval_code.lx

        data_indices = list(range(hx.shape[1]))

        if basis == 'Z':
            if add_faults:
                self._add_one_qubit_fault(circuit, data_indices, fault_location='p_m', fault_type='X_ERROR')
            circuit.append("M", data_indices, float(self.circuit_error_params['p_m']))
            circuit.append("SHIFT_COORDS", [], (1))

            # Wire one detector per Z-stabilizer row.
            # Each detector XORs the data-qubit parities from this round
            # with the ancilla MR result from the previous round (if pairwise_diff).
            for i in range(hz.shape[0]):
                supported_data_indices = list(np.where(hz[i, :] == 1)[0])
                rec_indices = [- len(data_indices) + j for j in supported_data_indices]
                if pairwise_diff:
                    # Previous ancilla MR result sits hz.shape[0] + len(data_indices) back.
                    rec_indices.append(- hz.shape[0] + i - len(data_indices))
                circuit.append("Detector", [stim.target_rec(r) for r in rec_indices], (0))

            # Logical observable declarations (Lz rows).
            for i in range(lz.shape[0]):
                supported = list(np.where(lz[i, :] == 1)[0])
                rec_indices = [- len(data_indices) + j for j in supported]
                circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(r) for r in rec_indices], (i))

        elif basis == 'X':
            if add_faults:
                self._add_one_qubit_fault(circuit, data_indices, fault_location='p_m', fault_type='DEPOLARIZE1')
            circuit.append("MX", data_indices, float(self.circuit_error_params['p_m']))
            circuit.append("SHIFT_COORDS", [], (1))

            # Wire one detector per X-stabilizer row.
            for i in range(hx.shape[0]):
                supported_data_indices = list(np.where(hx[i, :] == 1)[0])
                rec_indices = [- len(data_indices) + j for j in supported_data_indices]
                if pairwise_diff:
                    rec_indices.append(- hx.shape[0] + i - len(data_indices))
                circuit.append("Detector", [stim.target_rec(r) for r in rec_indices], (0))

            # Logical observable declarations (Lx rows).
            for i in range(lx.shape[0]):
                supported = list(np.where(lx[i, :] == 1)[0])
                rec_indices = [- len(data_indices) + j for j in supported]
                circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(r) for r in rec_indices], (i))

    # ------------------------------------------------------------------
    # Noise injection helpers
    # ------------------------------------------------------------------

    def _add_one_qubit_fault(self,
                           circuit: stim.Circuit,
                           indices: List[int],
                           fault_location: str,
                           fault_type: Optional[str] = "X_ERROR",
                           ) -> None:
        """Inject single-qubit noise into the circuit.

        Appends a single stim noise instruction (e.g. ``X_ERROR`` or
        ``DEPOLARIZE1``) on every qubit in *indices* with probability
        taken from ``self.circuit_error_params[fault_location]``.

        Args:
            circuit:        The stim circuit (modified in-place).
            indices:        Qubit indices to apply the noise channel to.
            fault_location: Key into :attr:`circuit_error_params` that
                            determines the probability (e.g. ``'p_i'``,
                            ``'p_m'``, ``'p_state_p'``).
            fault_type:     Stim noise instruction name.  Supported:
                            ``'X_ERROR'``, ``'DEPOLARIZE1'``.
        """
        circuit.append(fault_type, indices, float(self.circuit_error_params[fault_location]))

    def _add_two_qubit_fault(self,
                             circuit: stim.Circuit,
                             qubit_pairs: List[Tuple[int, int]],
                             fault_location: str = 'p_CX',
                             ) -> None:
        """Inject two-qubit depolarizing noise on CX gate pairs.

        Appends a DEPOLARIZE2 instruction covering all (control, target)
        pairs from the most recent CX layer.  Skipped when the probability
        is zero, so existing behavior is unchanged for p_CX=0.

        Args:
            circuit:        The stim circuit (modified in-place).
            qubit_pairs:    List of (control, target) tuples from CX gates.
            fault_location: Key into circuit_error_params (default 'p_CX').
        """
        p = float(self.circuit_error_params[fault_location])
        if p <= 0:
            return
        targets = []
        for q1, q2 in qubit_pairs:
            targets.extend([q1, q2])
        circuit.append("DEPOLARIZE2", targets, p)


# ---------------------------------------------------------------------------
# CSS syndrome extraction (memory experiment)
# ---------------------------------------------------------------------------

class DEMSyndromeExtraction(BaseDEMSim):
    """Full CSS syndrome extraction circuit builder.

    Builds a stim memory-experiment circuit that interleaves X- and
    Z-stabilizer measurements using a graph-coloring-based CX gate
    schedule (see :func:`~sim_qec.circuit_scheduling.ColorationCircuit`).

    The scheduling dict format (produced by ``ColorationCircuit``)::

        scheduling['X'][time_step] = {ancilla_row_idx: data_col_idx, ...}
        scheduling['Z'][time_step] = {ancilla_row_idx: data_col_idx, ...}

    Each entry maps a stabilizer (row index in Hx/Hz) to the data qubit
    (column index) it interacts with during that time step.

    CX gate direction follows the CSS convention:
        * X stabilizers:  CX  ancilla -> data   (ancilla is control)
        * Z stabilizers:  CX  data -> ancilla    (data is control)
    """

    def __init__(self,
                 code: css_code,
                 num_cycles: int,
                 circuit_error_params: Union[dict, CircuitErrorParams],
                 physical_error_rate: Optional[float] = None,
                 ):
        super().__init__(code, num_cycles, circuit_error_params, physical_error_rate)

        # Compute CX gate schedules from the parity-check matrices via
        # bipartite edge coloring.  Each schedule is a list of dicts
        # (one per time step), mapping ancilla-row -> data-column.
        scheduling_X = ColorationCircuit(code.hx)
        scheduling_Z = ColorationCircuit(code.hz)
        print("Syndrome extraction circuit scheduling for X stabilizers:", scheduling_X)
        print("Syndrome extraction circuit scheduling for Z stabilizers:", scheduling_Z)
        self.scheduling = {
            "X": scheduling_X,
            "Z": scheduling_Z,
        }

    # ------------------------------------------------------------------
    # Full CSS circuit
    # ------------------------------------------------------------------

    def build_circuit(self,
                      fault_type: str = 'DEPOLARIZE1') -> stim.Circuit:
        """Build the full CSS memory-experiment circuit.

        Circuit structure::

            circuit_init  +  circuit_rep1  +  (num_cycles-1)*circuit_rep2  +  final_circuit

        * **circuit_init** — Reset all qubits (data in X-basis, ancillas in
          Z-basis) and inject state-preparation noise on data qubits.
        * **circuit_rep1** — First noisy syndrome round.  Detectors reference
          only the current round's X-ancilla measurements (no prior round
          to compare against for X; Z detectors are implicit in _noisy_sec_round).
        * **circuit_rep2** — Subsequent noisy rounds.  Detectors compare the
          current round with the previous round (pairwise difference).
        * **final_circuit** — Transversal X-basis measurement of data qubits
          with detectors tied back to the last noisy round.

        Args:
            fault_type: Stim single-qubit noise instruction applied at each
                        noise location (default ``'DEPOLARIZE1'``).

        Returns:
            A complete stim.Circuit ready for sampling.
        """
        data_indices = self.data_indices
        Z_ancilla_indices = self.Z_ancilla_indices
        X_ancilla_indices = self.X_ancilla_indices

        # ---- Initialization layer ----
        circuit_init = stim.Circuit()
        self._initialize_circuit(circuit_init, reset="RX", mode='CSS')
        # State preparation noise on data qubits only.
        self._add_one_qubit_fault(circuit_init, data_indices, fault_location='p_state_p', fault_type=fault_type)

        # ---- First noisy syndrome round (rep1) ----
        # No prior round exists, so detectors for X-ancilla measurements
        # reference only the current round (single-reference detectors).
        circuit_rep1 = stim.Circuit()
        self._noisy_sec_round(circuit_rep1, data_indices, fault_type=fault_type, mode='CSS')

        # Hadamard on X ancillas to return from X-basis back to Z-basis
        # before measurement, then inject measurement noise.
        circuit_rep1.append("H", X_ancilla_indices)
        self._add_one_qubit_fault(circuit_rep1, X_ancilla_indices, fault_location='p_m', fault_type=fault_type)

        # Measure-and-reset all ancillas.  Order: Z ancillas first, then X
        # ancillas — this order must be consistent with detector indexing.
        circuit_rep1.append("MR", Z_ancilla_indices + X_ancilla_indices)
        circuit_rep1.append("SHIFT_COORDS", [], (1))

        # First-round detectors: only X-ancilla results (single reference).
        for i in range(len(X_ancilla_indices)):
            circuit_rep1.append("DETECTOR", [stim.target_rec(- len(X_ancilla_indices) + i)], (0))
        circuit_rep1.append("TICK")

        # ---- Subsequent noisy rounds (rep2) ----
        # Detectors compare current round with the previous round
        # (pairwise-difference detectors).
        circuit_rep2 = stim.Circuit()
        self._noisy_sec_round(circuit_rep2, data_indices, fault_type=fault_type, mode='CSS')

        circuit_rep2.append("H", X_ancilla_indices)
        self._add_one_qubit_fault(circuit_rep2, X_ancilla_indices, fault_location='p_m', fault_type=fault_type)

        circuit_rep2.append("MR", Z_ancilla_indices + X_ancilla_indices)
        circuit_rep2.append("SHIFT_COORDS", [], (1))

        # Pairwise-difference detectors: current X-ancilla XOR previous X-ancilla.
        num_ancilla_total = len(Z_ancilla_indices) + len(X_ancilla_indices)
        for i in range(len(X_ancilla_indices)):
            circuit_rep2.append("DETECTOR", [
                stim.target_rec(- len(X_ancilla_indices) + i),
                stim.target_rec(- len(X_ancilla_indices) + i - num_ancilla_total),
            ], (0))
        circuit_rep2.append("TICK")

        # ---- Terminal transversal measurement ----
        # Noise acts on data qubits before the measurement; the measurement
        # itself carries the stim flip probability from p_m.
        final_circuit = stim.Circuit()
        self._transversal_measurement(final_circuit, basis='X', add_faults=True, pairwise_diff=True)

        # ---- Assemble full circuit ----
        circuit_syn_meas = circuit_init + circuit_rep1 + (self.num_cycles - 1) * circuit_rep2 + final_circuit
        return circuit_syn_meas

    # ------------------------------------------------------------------
    # Repetition-code circuit (Z-only stabilizers)
    # ------------------------------------------------------------------

    def build_repetition_circuit(self,
                                 fault_type: str = 'X_ERROR',
                                 pairwise_diff: bool = True) -> stim.Circuit:
        """Build a Z-only repetition-code syndrome extraction circuit.

        This simplified circuit measures only Z-type stabilizers and is
        used as a demo for learnability / unlearnability analysis.

        Circuit structure is the same init + rep1 + (N-1)*rep2 + final
        pattern, but with no X-ancilla logic.

        Args:
            fault_type:     Stim noise instruction (default ``'X_ERROR'``).
            pairwise_diff:  If ``True``, the final transversal measurement's
                            detectors reference the last noisy round.

        Returns:
            A complete stim.Circuit.
        """
        data_indices = self.data_indices
        Z_ancilla_indices = self.Z_ancilla_indices

        # ---- Initialization ----
        circuit_init = stim.Circuit()
        self._initialize_circuit(circuit_init, reset="R", mode='repetition')
        self._add_one_qubit_fault(circuit_init, data_indices, fault_location='p_state_p', fault_type=fault_type)

        # ---- First noisy Z-syndrome round ----
        # Single-reference detectors (no prior round).
        circuit_rep1 = stim.Circuit()
        self._noisy_sec_round(circuit_rep1, data_indices, fault_type=fault_type, mode='repetition')
        circuit_rep1.append("MR", Z_ancilla_indices)
        circuit_rep1.append("SHIFT_COORDS", [], (1))
        for i in range(len(Z_ancilla_indices)):
            circuit_rep1.append("DETECTOR", [stim.target_rec(- len(Z_ancilla_indices) + i)], (0))
        circuit_rep1.append("TICK")

        # ---- Subsequent noisy rounds ----
        # Pairwise-difference detectors.
        circuit_rep2 = stim.Circuit()
        self._noisy_sec_round(circuit_rep2, data_indices, fault_type=fault_type, mode='repetition')
        circuit_rep2.append("MR", Z_ancilla_indices)
        circuit_rep2.append("SHIFT_COORDS", [], (1))
        for i in range(len(Z_ancilla_indices)):
            circuit_rep2.append("DETECTOR", [
                stim.target_rec(- len(Z_ancilla_indices) + i),
                stim.target_rec(- 2 * len(Z_ancilla_indices) + i),
            ], (0))
        circuit_rep2.append("TICK")

        # ---- Terminal transversal Z-measurement ----
        final_circuit = stim.Circuit()
        self._transversal_measurement(final_circuit, basis='Z', add_faults=True, pairwise_diff=pairwise_diff)

        circuit_syn_meas = circuit_init + circuit_rep1 + (self.num_cycles - 1) * circuit_rep2 + final_circuit
        return circuit_syn_meas

    # ------------------------------------------------------------------
    # Demo circuit for learnability / unlearnability
    # ------------------------------------------------------------------

    def build_demo_rep_circuit(self,
                               fault_type: str = 'X_ERROR',
                               type: str = 'midcircuit') -> stim.Circuit:
        """Build a demo circuit illustrating learnable vs. unlearnable logical noise.

        Args:
            fault_type: Stim noise instruction.
            type:       ``'midcircuit'`` — single noisy syndrome round followed
                        by logical measurement only (no stabilizer detectors at
                        the terminal round).  Shows how mid-circuit measurement
                        noise can create unlearnable logical noise.
                        ``'logical'`` — delegates to ``build_logical_demo``.

        Returns:
            A stim.Circuit.
        """
        if type == 'midcircuit':
            data_indices = self.data_indices
            Z_ancilla_indices = self.Z_ancilla_indices

            # Init without state-preparation noise (isolate measurement noise).
            circuit_init = stim.Circuit()
            self._initialize_circuit(circuit_init, reset="R", mode='repetition')

            # Single noisy syndrome extraction round.
            circuit_rep1 = stim.Circuit()
            self._noisy_sec_round(circuit_rep1, data_indices, fault_type=fault_type, mode='repetition')
            circuit_rep1.append("MR", Z_ancilla_indices)
            circuit_rep1.append("SHIFT_COORDS", [], (1))
            for i in range(len(Z_ancilla_indices)):
                circuit_rep1.append("DETECTOR", [stim.target_rec(- len(Z_ancilla_indices) + i)], (0))
            circuit_rep1.append("TICK")

            # Terminal: only logical observable declarations (no stabilizer
            # detectors), so the logical channel depends entirely on the
            # noisy syndrome round.
            final_circuit = stim.Circuit()
            lz = self.eval_code.lz
            for i in range(lz.shape[0]):
                supported = list(np.where(lz[i, :] == 1)[0])
                rec_indices = [- len(data_indices) + j for j in supported]
                final_circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(r) for r in rec_indices], (i))

            return circuit_init + circuit_rep1 + final_circuit

        elif type == 'logical':
            return self.build_logical_demo(fault_type=fault_type)

    # ------------------------------------------------------------------
    # Noisy syndrome extraction round
    # ------------------------------------------------------------------

    def _noisy_sec_round(self,
                         circuit: stim.Circuit,
                         data_indices: List[int],
                         fault_type: str = 'X_ERROR',
                         mode: str = 'repetition',
                         ) -> stim.Circuit:
        
        """
        Append one noisy stabilizer-measurement round to *circuit*.

        This is the core of the syndrome extraction circuit.  It applies
        CX gates between data and ancilla qubits according to the
        stabilizer parity-check matrix, with noise injected at each step.

        Args:
            circuit:      The stim circuit (modified in-place).
            data_indices: Indices of data qubits.
            fault_type:   Stim noise instruction for single-qubit errors.
            mode:         ``'repetition'`` — measure only Z stabilizers
                          (no scheduling, one CX layer per stabilizer row).
                          ``'CSS'`` — measure both X and Z stabilizers using
                          the graph-coloring schedule stored in
                          ``self.scheduling``.

        **Repetition mode** (Z-only):
            For each Z-stabilizer row *r* in Hz:
              1. Apply idling noise (``p_i``) on all data + Z-ancilla qubits.
              2. For each data qubit *j* with Hz[r, j] == 1, append
                 ``CX data[j] -> Z_ancilla[r]``.
              3. ``DEPOLARIZE2`` (``p_CX``) on each CX pair.
              4. TICK.
            After all rows: apply measurement noise (``p_m``) on Z ancillas.

        **CSS mode** (X then Z):
            *X stabilizers:*
              1. Hadamard on X ancillas (prepare in X-basis for X-type
                 stabilizer measurement).
              2. State-prep noise on X ancillas.
              3. For each scheduling time step:
                 a. Idling-gate noise on all data + X-ancilla qubits.
                 b. CX gates: ``CX X_ancilla[row] -> data[col]``
                    (ancilla is control — this measures X on the data qubit).
                 c. ``DEPOLARIZE2`` (``p_CX``) on each CX pair.
                 d. Idling noise (``p_i``) on data qubits not involved in
                    a CX gate this step.
                 e. TICK.
            *Z stabilizers:*
              For each scheduling time step:
                a. Idling-gate noise on all data + Z-ancilla qubits.
                b. CX gates: ``CX data[col] -> Z_ancilla[row]``
                   (data is control — this measures Z on the data qubit).
                c. ``DEPOLARIZE2`` (``p_CX``) on each CX pair.
                d. Idling noise on uninvolved data qubits.
                e. TICK.

        Returns:
            The same *circuit* object.
        """
        Z_ancilla_indices = self.Z_ancilla_indices
        X_ancilla_indices = self.X_ancilla_indices

        if mode == 'repetition':
            hz = self.eval_code.hz
            for time_step in range(hz.shape[0]):
                # Idling noise on all data + Z-ancilla qubits this step.
                idling_qubits = data_indices + Z_ancilla_indices
                self._add_one_qubit_fault(circuit, idling_qubits, fault_location='p_i', fault_type=fault_type)

                # CX gates: data[j] controls Z_ancilla[time_step] for each
                # nonzero entry in the current stabilizer row.
                for j in range(hz.shape[1]):
                    if hz[time_step, j] == 1:
                        circuit.append("CX", [j, Z_ancilla_indices[time_step]])

                # Two-qubit depolarizing noise on the CX pairs this step.
                cx_pairs = [(j, Z_ancilla_indices[time_step])
                            for j in range(hz.shape[1]) if hz[time_step, j] == 1]
                self._add_two_qubit_fault(circuit, cx_pairs)

                circuit.append("TICK")

            # Measurement noise applied once after all CX layers.
            self._add_one_qubit_fault(circuit, Z_ancilla_indices, fault_location='p_m', fault_type=fault_type)

        elif mode == 'CSS':
            # ---- X-stabilizer measurements ----
            # Hadamard on X ancillas: transforms Z-basis ancilla into X-basis
            # so that the subsequent CX gates effectively measure X on data.
            circuit.append("H", X_ancilla_indices)
            self._add_one_qubit_fault(circuit, X_ancilla_indices, fault_location='p_state_p', fault_type=fault_type)
            circuit.append("TICK")

            for time_step in range(len(self.scheduling['X'])):
                # Idling-gate noise on all qubits involved in this layer.
                idling_qubits = data_indices + X_ancilla_indices
                idling_data_indices = list(copy.deepcopy(data_indices))
                self._add_one_qubit_fault(circuit, idling_qubits, fault_location='p_idling_gate', fault_type=fault_type)

                # CX gates for X stabilizers: ancilla (control) -> data (target).
                # scheduling['X'][time_step] = {ancilla_row_idx: data_col_idx}
                for j in self.scheduling['X'][time_step]:
                    X_ancilla_index = X_ancilla_indices[j]
                    data_index = self.scheduling['X'][time_step][j]
                    circuit.append("CX", [X_ancilla_index, data_index])
                    # Track which data qubits are *not* involved in a gate
                    # this step — they receive idling noise instead.
                    if data_index in idling_data_indices:
                        idling_data_indices.remove(data_index)

                # Two-qubit depolarizing noise on X-stabilizer CX pairs.
                cx_pairs = [(X_ancilla_indices[j], self.scheduling['X'][time_step][j])
                            for j in self.scheduling['X'][time_step]]
                self._add_two_qubit_fault(circuit, cx_pairs)

                # Idling noise on data qubits not touched by a CX this step.
                self._add_one_qubit_fault(circuit, idling_data_indices, fault_location='p_i', fault_type=fault_type)
                circuit.append("TICK")

            # ---- Z-stabilizer measurements ----
            for time_step in range(len(self.scheduling['Z'])):
                idling_qubits = data_indices + Z_ancilla_indices
                self._add_one_qubit_fault(circuit, idling_qubits, fault_location='p_idling_gate', fault_type=fault_type)
                idling_data_indices = list(copy.deepcopy(data_indices))

                # CX gates for Z stabilizers: data (control) -> ancilla (target).
                # scheduling['Z'][time_step] = {ancilla_row_idx: data_col_idx}
                for j in self.scheduling['Z'][time_step]:
                    Z_ancilla_index_sch = Z_ancilla_indices[j]
                    data_index_sch = self.scheduling['Z'][time_step][j]
                    circuit.append("CX", [data_index_sch, Z_ancilla_index_sch])
                    if data_index_sch in idling_data_indices:
                        idling_data_indices.remove(data_index_sch)

                # Two-qubit depolarizing noise on Z-stabilizer CX pairs.
                cx_pairs = [(self.scheduling['Z'][time_step][j], Z_ancilla_indices[j])
                            for j in self.scheduling['Z'][time_step]]
                self._add_two_qubit_fault(circuit, cx_pairs)

                self._add_one_qubit_fault(circuit, idling_data_indices, fault_location='p_i', fault_type=fault_type)
                circuit.append("TICK")


# ---------------------------------------------------------------------------
# Non-CSS code stub (placeholder)
# ---------------------------------------------------------------------------

class DEMSyndromeExtractionNonCSS(BaseDEMSim):
    """Syndrome extraction for non-CSS codes (not yet implemented)."""

    def __init__(self,
                 code: css_code,
                 num_cycles: int,
                 circuit_error_params: Union[dict, CircuitErrorParams],
                 physical_error_rate: Optional[float] = None,
                 ):
        super().__init__(code, num_cycles, circuit_error_params, physical_error_rate)


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    CIRCUIT_ERROR_PARAMS = {"p_i": 0.01, "p_state_p": 0.01, "p_m": 0.01, "p_CX":0.0, "p_idling_gate": 0.0}

    X_stabilizers = [[3,4,5,6],[1,2,5,6],[0,2,4,6]]
    Z_stabilizers = [[3,4,5,6],[1,2,5,6],[0,2,4,6]]

    from sim_qec.utils import get_parity_check_matrix

    Hx = get_parity_check_matrix(X_stabilizers, 7)
    Hz = get_parity_check_matrix(Z_stabilizers, 7)
    rep_code = css_code(Hx, Hz)

    # Circuit error knobs (per-location probabilities).
    # A physical_error_rate "p" will uniformly scale these inside DEMSyndromeExtraction.
    CIRCUIT_ERROR_PARAMS = {
        "p_i": 1.0,          # idling single-qubit error
        "p_state_p": 1.0,    # state prep error
        "p_m": 1.0,          # measurement error
        "p_CX": 0.0,         # 2-qubit depolarizing error (not used in 'repetition' mode below)
        "p_idling_gate": 0.0 # idling during gates
    }

    num_cycles = 3
    p = 1e-5  # overall physical error scale; you can start with 0.01 and tweak

    dem = DEMSyndromeExtraction(
        code=rep_code,
        num_cycles=num_cycles,
        circuit_error_params=CIRCUIT_ERROR_PARAMS,
        physical_error_rate=p,
    )

    # Build the generic CSS code circuit; fault_type "DEPOLARIZE1" injects depolarizing noise where configured
    circ = dem.build_circuit(fault_type="DEPOLARIZE1")
    print(circ)  # helpful when debugging
    # print(circ.diagram("timeline"))

    # Get the detector error model (DEM) and a sampler
    det_model = circ.detector_error_model(flatten_loops=True)
    print("Detectors:", det_model.num_detectors)
    print("Observables:", det_model.num_observables)

    # sampler = circ.compile_detector_sampler()
    # # Each shot returns: (detector bits, obs bits) if separate_observables=True
    # det_vals, log_vals = sampler.sample(
    #     shots=1000000, separate_observables=True
    # )
    # print("detector bit array shape:", det_vals.shape)
    # print("observable bit array shape:", log_vals.shape)
    # # print("First 10 detector results:\n", det_vals[:10,0])
    # det_vals = det_vals.astype(int) #convert from bool to int
    # log_vals = log_vals.astype(int)
    # # print("First 10 detector results:\n", det_vals[:10,0])
    # q = det_vals.mean(axis=0)
    # parity_expectation = 1.0 - 2.0 * q
    # print("Syndrome expectation values:", parity_expectation)

    # #now we compute the syndrome expectation values

    # from beliefmatching import detector_error_model_to_check_matrices

    # dem = circ.detector_error_model(flatten_loops=True)
    # dem_matrix = detector_error_model_to_check_matrices(dem, allow_undecomposed_hyperedges=True)
    # h = dem_matrix.check_matrix.toarray()
    # l = dem_matrix.observables_matrix.toarray()
    # channel_probs = dem_matrix.priors
    # print(f'Check matrix shape: {h.shape}, number of faults: {len(channel_probs)}')
    # print(f'Channel error probabilities: {channel_probs}')
    # # print(f'sum of priors" {sum(p)}')
    # space_time_code_params = {'H': h, 'L': l, 'channel_probs': channel_probs}




    # from sim_qec.noise_model import PredictPriors
    # dem_samples = PredictPriors(
    #     dectector_samples=det_vals,
    #     check_matrix=h,
    #     subsample=True,
    # )
    # A_syndrome, sample_stabs = dem_samples._build_A_matrix_syndromes()
    # print(f'A syndrome matrix shape: {A_syndrome.shape}')
    # print(f'checking the rank of this matrix: {np.linalg.matrix_rank(A_syndrome)}')
    # sample_stab_eigs = dem_samples._get_syndrome_expectations(sample_stabs=sample_stabs)
    # print(f'sample stabilizer eigenvalues: {sample_stab_eigs}')
    # # we now solve the linear system to get the priors
    # predicted_priors = dem_samples.predict_priors(A_syndrome, sample_stab_eigs)
    # print('-' * 20 )
    # print(f'testing the prioers predictions here')
    # print('-' * 20 )
    # print(f'predicted priors: {predicted_priors}')
    # print(f'original channel priors: {channel_probs}')


    # # decoder = relay_decoder
    # # decoder = bplsd_decoder
    # # decoder.set_decoder(space_time_code_params)
    # # corrections = decoder.decode(detector_vals)

    # # les = 1 * ((logical_vals + corrections @ l.T % 2) % 2).any(axis=1)

    # # lep = np.average(les)
    # # print('Logical error probability:', lep)
