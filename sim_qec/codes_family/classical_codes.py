import numpy as np 
from scipy import sparse
import random 
from typing import Tuple, Dict, List, Optional, Union
import itertools



def generate_random_binary_matrix(m: int, n: int, k: int=2) -> np.ndarray:
    """
    Generate an m x n binary matrix where the maximum row or column weight is k.

    :param m: Number of rows
    :param n: Number of columns
    :param k: Maximum weight of any row or column
    :return: m x n binary matrix
    """
    if k > min(m, n):
        raise ValueError("k cannot be greater than the minimum of m and n")
    
    matrix = np.zeros((m, n), dtype=int)
    
    # To ensure row and column weights do not exceed k, we use a greedy approach.
    row_counts = np.zeros(m, dtype=int)
    col_counts = np.zeros(n, dtype=int)
    
    for _ in range(k):
        for i in range(m):
            for j in range(n):
                if row_counts[i] < k and col_counts[j] < k:
                    matrix[i, j] = 1
                    row_counts[i] += 1
                    col_counts[j] += 1

    return matrix


def generate_even_ones_matrix(m, n):
    # Step 1: Create a random binary matrix
    matrix = np.random.randint(0, 2, size=(m, n))

    # Step 2: Ensure each column has an even number of 1s
    for col in range(n):
        # Calculate the sum of 1s in the column
        col_sum = np.sum(matrix[:, col])
        
        # If the sum is odd, flip one of the bits in the column to make it even
        if col_sum % 2 != 0:
            # Choose a random row to flip the bit
            row_to_flip = np.random.randint(0, m)
            matrix[row_to_flip, col] = 1 - matrix[row_to_flip, col]

    return matrix

def generate_even_support_matrix(m, n):
    # Start with an empty m x n binary matrix
    matrix = np.zeros((m, n), dtype=int)

    # Randomly fill the matrix ensuring each row has an even number of 1s
    for i in range(m):
        # Choose random indices to set to 1; we choose even number of indices
        indices = np.random.choice(n, size=(n//2)*2, replace=False)  # Choose n//2 pairs, double for even
        matrix[i, indices] = 1

    # Adjust columns to have even number of 1s
    for j in range(n):
        col_sum = np.sum(matrix[:, j])
        if col_sum % 2 != 0:  # If the column sum is odd
            # Find a row to toggle a bit to make it even
            for i in range(m):
                if matrix[i, j] == 0:
                    matrix[i, j] = 1
                    break
                elif matrix[i, j] == 1:
                    matrix[i, j] = 0
                    break

    return matrix



def cyclic_square_matrix(column_weight, num_rows):
    if column_weight < 1 or column_weight > num_rows:
        raise ValueError("Invalid column weight")

    matrix = np.zeros((num_rows, num_rows), dtype=int)

    for i in range(num_rows):
        for j in range(column_weight):
            matrix[i, (i + j) % num_rows] = 1

    return matrix


'''
Below we focus on the QC-LDPC codes based on the polynomial representation and check their cycle properties. 
'''


LiftMat = Tuple[Tuple[int, int], Dict[Tuple[int, int], Union[int, None, List[int]]]]

def _normalize_entry(val: Union[int, None, List[int]], ell: int) -> Optional[int]:
    """
    Return an exponent in {0,...,ell-1} or None (meaning zero).
    Allowed encodings:
      - int in [0, ell-1] -> that exponent
      - int == ell -> zero
      - None -> zero
      - [] -> zero
      - [e] with e in [0, ell-1] -> that exponent
    Any other form raises ValueError.
    """
    if val is None:
        return None
    if isinstance(val, list):
        if len(val) == 0:
            return None
        if len(val) == 1:
            val = val[0]
        else:
            raise ValueError("Non-monomial entry (list has length > 1).")
    if isinstance(val, int):
        if val == ell:
            return None
        if 0 <= val < ell:
            return val
        raise ValueError(f"Exponent {val} out of range for ell={ell}.")
    raise TypeError(f"Unsupported entry type: {type(val)}")

def first_4cycle_witness(lift_mat: LiftMat, ell: int):
    """
    Return (has_4cycle, witness) where witness is a dict describing
    a found 4-cycle, or None if none exists.
    Assumes A is in SMCF so we skip column pairs involving the first m columns.
    Indices are 0-based.
    """
    (m, n), data = lift_mat

    def E(i: int, j: int) -> Optional[int]:
        # Missing keys count as zero as well.
        val = data.get((i, j), None)
        return _normalize_entry(val, ell)

    for j in range(m, n):                # free columns only
        for jp in range(j + 1, n):
            first_row_for_diff = {}      # diff -> row index
            for i in range(m):
                a, b = E(i, j), E(i, jp)
                if a is None or b is None:
                    continue
                d = (a - b) % ell
                if d in first_row_for_diff:
                    i0 = first_row_for_diff[d]
                    return True, {
                        "rows": (i0, i),
                        "cols": (j, jp),
                        "diff_mod_ell": d,
                        "exponents": {
                            (i0, j): E(i0, j), (i0, jp): E(i0, jp),
                            (i,  j): a,        (i,  jp): b
                        }
                    }
                first_row_for_diff[d] = i
    return False, None

def has_no_4cycles(lift_mat: LiftMat, ell: int) -> bool:
    has4, _ = first_4cycle_witness(lift_mat, ell)
    return not has4
