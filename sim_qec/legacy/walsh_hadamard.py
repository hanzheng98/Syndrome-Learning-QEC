import numba
import numpy as np
import itertools
import time
from sim_qec.utils import symplectic_to_pauli


def pauli_str_to_bits(s: str) -> np.ndarray:
    """
    Convert a k-bit string s (e.g. '0101101') to a NumPy int8 array of shape (k,).
    """
    return np.array([int(ch) for ch in s], dtype=np.int8)

def build_operator_arrays(eigvals_dict: dict, k: int):
    """
    Convert erorr_eignvalues dict from {(sx, sz): value} into 
    three arrays suitable for Numba:

    Returns:
      op_sx: shape (N, k), each row is the sx bits
      op_sz: shape (N, k), each row is the sz bits
      op_vals: shape (N,) float array of eigenvalues

    where N = number of operators in the dictionary.
    """
    ops = list(eigvals_dict.items())  # [((sx,sz), value), ...]
    N = len(ops)

    op_sx = np.zeros((N, k), dtype=np.int8)
    op_sz = np.zeros((N, k), dtype=np.int8)
    op_vals = np.zeros(N, dtype=np.float64)
    
    for i, ((sx_str, sz_str), val) in enumerate(ops):
        op_sx[i] = pauli_str_to_bits(sx_str)
        op_sz[i] = pauli_str_to_bits(sz_str)
        op_vals[i] = val
    
    return op_sx, op_sz, op_vals

def build_target_operators(k: int):
    """
    Build the list of all 4^k target operators in symplectic form 
    as two arrays tar_sx, tar_sz (each shape (4^k, k)).
    """
    bits = ['0','1']
    all_ops = [(sx, sz) 
                for sx in itertools.product(bits, repeat=k)
                for sz in itertools.product(bits, repeat=k)]
    M = len(all_ops)  # M=4^k
    tar_sx = np.zeros((M, k), dtype=np.int8)
    tar_sz = np.zeros((M, k), dtype=np.int8)
    
    for i, (sx_bits, sz_bits) in enumerate(all_ops):
        tar_sx[i] = [int(x) for x in sx_bits]
        tar_sz[i] = [int(x) for x in sz_bits]
    
    return tar_sx, tar_sz


@numba.njit
def commutation_sign_bits(
    sxA: np.ndarray, szA: np.ndarray,
    sxB: np.ndarray, szB: np.ndarray
    ) -> float:
    """
    Return +1 if the dot product mod 2 is 0, else -1.
    dot = sum( sxA[i]*szB[i] + szA[i]*sxB[i] ) mod 2
    """
    dot = 0
    k = sxA.size
    for i in range(k):
        dot += sxA[i]*szB[i] + szA[i]*sxB[i]
    if (dot % 2) == 0:
        return 1.0
    else:
        return -1.0
    

@numba.njit(parallel=True)
def compute_error_rates_numba(
    tar_sx: np.ndarray, tar_sz: np.ndarray,  # shape (M, k)
    op_sx: np.ndarray, op_sz: np.ndarray, op_vals: np.ndarray,  # shape (N, k) + shape (N,)
    k: int
    ) -> np.ndarray:
    """
    For each target operator T in [0..M-1], sum over N known eigenvalue operators:
      error_rate_T = (1/(4^k)) sum_{j in [N]} [ commutation_sign_bits(T, op_j ) * op_vals[j] ].

    M = 4^k (all target operators).
    N = len(op_sx) (the dictionary size, possibly also up to 4^k).
    """
    M = tar_sx.shape[0]
    N = op_sx.shape[0]
    out = np.zeros(M, dtype=np.float64)

    scale = 1.0 / (4**k)

    # Parallelize the outer loop using prange
    for i in numba.prange(M):
        sum_val = 0.0
        for j in range(N):
            sign = commutation_sign_bits(tar_sx[i], tar_sz[i], op_sx[j], op_sz[j])
            sum_val += sign * op_vals[j]
        out[i] = sum_val * scale
    return out

def logical_convert_2_probability_numba(eigen_dict: dict, k: int, max_workers=8) -> dict:
    """
    Numba-accelerated approach for computing the error rates from Pauli eigenvalues:
      error_rate_p = 1/(4^k) sum_{p1} [commutation_sign(p, p1) * eig(p1)].
    
    Potentially large, but with parallelization and Numba, might be okay for k=7.
    
    Returns a dictionary {(sx, sz): float} of length 4^k.
    """
    # 1) Convert input dictionary to arrays
    op_sx, op_sz, op_vals = build_operator_arrays(eigen_dict, k)  # shape (N,k), (N,k), (N,)
    # 2) Build the entire set of 4^k target operators
    tar_sx, tar_sz = build_target_operators(k)  # shape (M=4^k, k), (M, k)

    # 3) Summation in parallel
    out_arr = compute_error_rates_numba(
        tar_sx, tar_sz, op_sx, op_sz, op_vals, k
    )  # shape (M,)

    # 4) Build the final dictionary
    M = tar_sx.shape[0]  # should be 4^k
    final_dict = {}
    for i in range(M):
        sx_bits = ''.join(str(bit) for bit in tar_sx[i])
        sz_bits = ''.join(str(bit) for bit in tar_sz[i])
        final_dict[symplectic_to_pauli((sx_bits, sz_bits))] = out_arr[i]
    
    return final_dict


if __name__ == "__main__":
    
    
    k = 1
    # Suppose eigen_dict is up to 4^k items. We'll do a small example:
    eigen_dict = {
        ("0","0"): 1.0,
        ("1","0"): 0.9,
        ("0","1"): 0.9,
        ("1","1"): 0.9,
        # anything not listed is effectively 0
    }
    start = time.time()
    rates_dict = logical_convert_2_probability_numba(eigen_dict, k, max_workers=8)
    end = time.time()
    print(f"Computed {len(rates_dict)} error rates for k=7 in {end - start:.2f} seconds.")
    print(rates_dict)
    # rates_dict now has 4^7 = 16384 entries.
