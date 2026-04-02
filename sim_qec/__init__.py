"""
Package initialization for sim_qec.

This module patches the third-party bposd.css.css_code implementation to
handle SciPy sparse matrices returned by ldpc.mod2 helper functions. The
default version uses numpy.vstack directly, which fails when the mod2
helpers yield csr_matrix outputs (as seen for the 17-qubit color code),
leading to a ValueError when scipy tries to coerce the sparse matrices.
"""

from __future__ import annotations

from typing import Any

_PATCHED_CSS_CODE_LOGICALS = False


def _patch_bposd_css_compute_logicals() -> None:
    """
    Replace css_code.compute_logicals with a sparse-aware version that first
    densifies mod2.nullspace / mod2.row_basis outputs before stacking.
    """
    global _PATCHED_CSS_CODE_LOGICALS
    if _PATCHED_CSS_CODE_LOGICALS:
        return

    try:
        import numpy as np
    except ModuleNotFoundError:
        return

    try:
        from bposd.css import css_code as _css_code
        from ldpc import mod2 as _mod2
    except ModuleNotFoundError:
        # Optional dependency; simply skip patching if bposd/ldpc are absent.
        return

    def _to_dense_uint8(matrix: Any) -> np.ndarray:
        """
        Convert csr_matrix / numpy.matrix outputs from ldpc.mod2 helpers into a
        proper 2-D numpy uint8 array without altering the logical content.
        """
        if hasattr(matrix, "toarray"):
            matrix = matrix.toarray()
        array = np.asarray(matrix, dtype=np.uint8)
        if array.ndim == 1:
            array = array[np.newaxis, :]
        elif array.ndim == 0:
            array = array.reshape(1, 1)
        return array

    def _compute_logicals(self):
        def compute_lz(hx, hz):
            ker_hx = _to_dense_uint8(_mod2.nullspace(hx))
            im_hzT = _to_dense_uint8(_mod2.row_basis(hz))

            stack_parts = [
                part for part in (im_hzT, ker_hx) if part.size > 0
            ]
            if stack_parts:
                log_stack = np.vstack(stack_parts)
            else:
                log_stack = np.zeros((0, hx.shape[1]), dtype=np.uint8)

            if log_stack.size == 0:
                return log_stack

            pivots = _mod2.row_echelon(log_stack.T)[3]
            start = im_hzT.shape[0]
            log_op_indices = [
                i for i in range(start, log_stack.shape[0]) if i in pivots
            ]
            return log_stack[log_op_indices]

        if self.K == np.nan:
            self.compute_dimension()
        self.lx = compute_lz(self.hz, self.hx)
        self.lz = compute_lz(self.hx, self.hz)
        return self.lx, self.lz

    _css_code.compute_logicals = _compute_logicals
    _PATCHED_CSS_CODE_LOGICALS = True


_patch_bposd_css_compute_logicals()
