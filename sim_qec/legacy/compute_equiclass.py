import numpy as np 
from scipy import sparse
import random 
from typing import Any, Callable, Dict, List, Optional, Type, Union
import itertools
import galois 
from sim_qec.utils import (
    find_logical_operators,
    get_parity_check_matrix,
)
'''
File written to compute the equivalence class of 
logical operators and its associated moments or error rate either analytically or numerically
'''

GF2 = galois.GF(2)




def convert_simplectic_matrix(Hx, Hz, Lx, Lz):
    """
    Input:
        Hx: rx x n matrix or None
        Hz: rz x n matrix or None
        Lx: k x n matrix or None
        Lz: k x n matrix or None

    Output:
        H_block: (r + rz) x 2n matrix (GF2 converted)
        L_block: 2k x 2n matrix (GF2 converted)

    If one of the H matrices is None, it is replaced by an empty matrix with 0 rows.
    Similarly, if one of the L matrices is None, it is replaced by a zero matrix with the
    appropriate number of rows.
    """


    # Determine number of columns n using Hx or Hz (at least one must be non-None)
    if Hx is not None:
        n = Hx.shape[1]
    elif Hz is not None:
        n = Hz.shape[1]
    else:
        raise ValueError("At least one of Hx or Hz must be provided (non-None) to determine the number of columns.")

    # If Hx or Hz is None, replace with a zero matrix with 0 rows (empty) and n columns
    if Hx is None:
        Hx = np.zeros((0, n), dtype=int)
    if Hz is None:
        Hz = np.zeros((0, n), dtype=int)

    # For the L matrices, determine n using Lx or Lz (at least one must be non-None)
    if Lx is None and Lz is None:
        raise ValueError("At least one of Lx or Lz must be provided (non-None) to determine the number of columns for the L block.")
    if Lx is None:
        # Use Lz to get number of rows (k) for Lx
        k = Lz.shape[0]
        Lx = np.zeros((k, n), dtype=int)
    if Lz is None:
        # Use Lx to get number of rows (k) for Lz
        k = Lx.shape[0]
        Lz = np.zeros((k, n), dtype=int)

    # Create zero matrices for the off-diagonal blocks
    zero_upper_right = np.zeros(Hx.shape, dtype=int)
    zero_lower_left = np.zeros(Hz.shape, dtype=int)

    # Build the block-diagonal matrix using np.block
    H_block = GF2(np.block([
        [Hx, zero_upper_right],
        [zero_lower_left, Hz]
    ]))
    
    L_block = GF2(np.block([
        [Lx, np.zeros(Lx.shape, dtype=int)],
        [np.zeros(Lz.shape, dtype=int), Lz]
    ]))

    return H_block, L_block


def compute_logical_equivalence(H_block: np.array, 
                                L_block: np.array
                                ):
    '''
    Input:  H_block: r x 2n
            L_block: 2k x 2n 

    Output: 
            L_equiv: dictionary (key, value) key : (ix, iz) value: list of all 2^r equivalent logical representation

    First parts are default to be Hx and Lx. 
    '''

    stab_combs = [list(bits) for bits in itertools.product([0, 1], repeat=H_block.shape[0])]
    log_combs = [list(bits) for bits in itertools.product([0, 1], repeat=L_block.shape[0])] # 4^k many
    n = int(H_block.shape[1]/2)
    # print(f'log_combs: {log_combs}')
    logical_equiv = {}
    logical_weight = {}
    for logcomb in log_combs:
        logcomb = GF2(logcomb)
        logcomb_basis = logcomb @ L_block # one of the 4^k logical basis given by (x, z) representation
        logical_equiv[tuple(logcomb.tolist())] = []
        logical_weight[tuple(logcomb.tolist())] = []
        for stabcomb in stab_combs:
            stabcomb = GF2(stabcomb)
            # print(stabcomb.shape)
            # break
            logcomb_basis_stab = logcomb_basis + stabcomb @ H_block
            # print(f'logcomb_basis shape: {logcomb_basis_stab.shape}')
            logical_equiv[tuple(logcomb.tolist())].append(logcomb_basis_stab)
            logcomb_basis_stab_x, logcomb_basis_stab_z = logcomb_basis_stab[:n], logcomb_basis_stab[n:]
            # print(f'logcomb_x shape: {logcomb_basis_stab_x.shape}')
            # print(f'logcomb_z sahape: {logcomb_basis_stab_z.shape}')
            
            weight = sum(a | b for a, b in zip(logcomb_basis_stab_x.tolist(), logcomb_basis_stab_z.tolist()))
            logical_weight[tuple(logcomb.tolist())].append(weight) 



    return logical_equiv, logical_weight


def compute_logical_equivalence2(Hx: Union[np.array, sparse.csc_matrix],
                                Hz: Union[np.array, sparse.csc_matrix],
                                Lx: Union[np.array, sparse.csc_matrix],
                                Lz: Union[np.array, sparse.csc_matrix]):
    
    '''
    Input: Hx: rx x n 
           Hz: rz x n 
           Lx: k x n 
           Lz: k x n 

    Output: Lx: dictionary (key, value) key : (ix) value: list of all 2^rx equivalent logical representation
            Lz: dictionary (key, value) key : (iz) value: list of all 2^rz equivalent logical representation 
    
    '''
    stab_combs_x = [list(bits) for bits in itertools.product([0, 1], repeat=Hx.shape[0])]
    stab_combs_z = [list(bits) for bits in itertools.product([0, 1], repeat=Hz.shape[0])]
    log_combs = [list(bits) for bits in itertools.product([0, 1], repeat=Lx.shape[0])] 

    # compute the logical x equivalence
    logx_equiv = {}
    logx_weight = {}
    for logcomb in log_combs:
        logcomb = GF2(logcomb) 
        logcomb_basis = logcomb @ GF2(Lx)
        logx_equiv[tuple(logcomb.tolist())] = []
        logx_weight[tuple(logcomb.tolist())] = []
        for stabcomb in stab_combs_x:
            stabcomb = GF2(stabcomb)
            logcomb_basis_stab = logcomb_basis + stabcomb @ GF2(Hx)
            logx_equiv[tuple(logcomb.tolist())].append(logcomb_basis_stab)
            # print(f'logcomb_basis_stab: {logcomb_basis_stab}')
            logx_weight[tuple(logcomb.tolist())].append(sum(logcomb_basis_stab.tolist()))

    # compute the logical z equivalence
    logz_equiv = {}
    logz_weight = {}
    for logcomb in log_combs:
        logcomb = GF2(logcomb) 
        logcomb_basis = logcomb @ GF2(Lz)
        logz_equiv[tuple(logcomb.tolist())] = []
        logz_weight[tuple(logcomb.tolist())] = []
        for stabcomb in stab_combs_z:
            stabcomb = GF2(stabcomb)
            logcomb_basis_stab = logcomb_basis + stabcomb @ GF2(Hz)
            logz_equiv[tuple(logcomb.tolist())].append(logcomb_basis_stab)
            logz_weight[tuple(logcomb.tolist())].append(sum(logcomb_basis_stab.tolist()))
    
    return logx_equiv, logz_equiv, logx_weight, logz_weight

if __name__ == "__main__":
    system_size = 7
    X_stabilizers = [[0, 1, 2, 3], [0, 1, 4, 5], [0, 2, 4, 6]]
    Z_stabilizers = [[0, 1, 2, 3], [0, 1, 4, 5], [0, 2, 4, 6]]

    Hx = get_parity_check_matrix(X_stabilizers, system_size)
    Hz = get_parity_check_matrix(Z_stabilizers, system_size) 

    Lx, Lz = find_logical_operators(Hx, Hz)

    # H_block, L_block = convert_simplectic_matrix(Hx, Hz, Lx, Lz)
    logx_equiv, logz_equiv, logx_weight, logz_weight = compute_logical_equivalence2(Hx, Hz, Lx, Lz)
    print(f'the logical x basis equivalence is: {logx_equiv}')
    print('----------------------------')
    print(f'the weight  is: {logx_weight}')
    print('----------------------------')

    # logical_equiv, logical_weight = compute_logical_equivalence(H_block, L_block)

    # print(f'the logical weight for the first one: {logical_weight[(1, 0)]}')


