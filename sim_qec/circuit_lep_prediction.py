import stim
import copy
import galois
import numpy as np
import time
import random
import itertools
from scipy import sparse
import random 
from typing import Any, Callable, Dict, List, Optional, Type, Union, Tuple
from sim_qec.codes_family.classical_codes import cyclic_square_matrix
from sim_qec.codes_family.hpc_lp import HGP
from sim_qec.legacy.decoders import MLEDecoder # for future impelmentation, add Decoder base class from which to inherit


# --- Optional accelerator (falls back automatically) ---
try:
    from numba import njit, prange
    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    _NUMBA_AVAILABLE = False

import math

@njit(inline='always')
def _parity64(x):  # x: np.uint64
    # Branchless parity of 64-bit word.
    x ^= x >> np.uint64(32)
    x ^= x >> np.uint64(16)
    x ^= x >> np.uint64(8)
    x ^= x >> np.uint64(4)
    x ^= x >> np.uint64(2)
    x ^= x >> np.uint64(1)
    return np.uint8(x & np.uint64(1))

@njit(parallel=True, fastmath=True)
def _compute_chunk(nums, row_masks_H, row_masks_L, weights, const, E):
    """
    JIT'ed inner loop.
    nums:          (Nc,) uint64  fault bitmasks
    row_masks_H:   (D,)  uint64  parity mask per detector
    row_masks_L:   (O,)  uint64  parity mask per observable
    weights:       (E,)  float64 log(p/(1-p))
    const:         ()    float64 sum(log(1-p))
    E:             int   # faults
    Returns: syndromes (Nc,D) uint8, logicals (Nc,O) uint8, logp (Nc,) float64
    """
    Nc = nums.shape[0]
    D  = row_masks_H.shape[0]
    O  = row_masks_L.shape[0]

    synd = np.empty((Nc, D), dtype=np.uint8)
    lam  = np.empty((Nc, O), dtype=np.uint8)
    logp = np.empty(Nc, dtype=np.float64)

    for i in prange(Nc):
        x = nums[i]

        # log probability of this fault
        s = const
        # Sum weights for the 1-bits (simple and Numba-friendly)
        for j in range(E):
            if ((x >> j) & np.uint64(1)) != 0:
                s += weights[j]
        logp[i] = s

        # H @ fault % 2 (as parities)
        for r in range(D):
            synd[i, r] = _parity64(row_masks_H[r] & x)

        # L @ fault % 2 (as parities)
        for o in range(O):
            lam[i, o] = _parity64(row_masks_L[o] & x)

    return synd, lam, logp

def _build_row_masks(A_uint8):
    """
    Build one uint64 mask per row. Requires A.shape[1] <= 63.
    """
    A = np.asarray(A_uint8, dtype=np.uint8)
    rows, E = A.shape
    masks = np.zeros(rows, dtype=np.uint64)
    for r in range(rows):
        m = np.uint64(0)
        row = A[r]
        # set bit j if entry is 1
        for j in range(E):
            if row[j] & 1:
                m |= (np.uint64(1) << np.uint64(j))
        masks[r] = m
    return masks

def _rows_to_keys(arr_uint8):
    """
    Turn each 0/1 row into a compact bytes key (stable with 'little' bitorder).
    """
    arr = np.ascontiguousarray(arr_uint8, dtype=np.uint8)
    packed = np.packbits(arr, axis=1, bitorder='little')
    # Return a Python list of bytes objects, one per row (fast to dict-key)
    return [packed[i].tobytes() for i in range(packed.shape[0])]

def _ints_for_weights_leq(E, max_w):
    """
    Generate all uint64 fault bitmasks with Hamming weight <= max_w (includes 0).
    Python-level generator; cheap relative to the JIT'd inner loop.
    """
    yield np.uint64(0)
    for w in range(1, max_w + 1):
        for comb in itertools.combinations(range(E), w):
            v = 0
            for j in comb:
                v |= (1 << j)
            yield np.uint64(v)




class PredictPriors:

    '''
        A class to predict the priors of a noise model from the syndrome expectation values
    
    '''

    def __init__(self,
                dectector_samples: Union[np.ndarray, sparse.csr_matrix],
                check_matrix: Union[np.ndarray, sparse.csr_matrix],
                subsample:bool=True,
                ):
        '''

        detector_samples: shape (num_samples, num_detectors) called from det_vals, log_vals = sampler.sample(shots=1000000, separate_observables=True)

        check_matrix: shape (num_detectors, num_faults) #this the fualts given from the priors

        '''
        self.detector_samples = dectector_samples
        self.check_matrix = check_matrix
        self.subsample = subsample 

    
    def _build_A_matrix_syndromes(self,
                                  ) -> np.ndarray:
        '''
            Build the A matrix: shape (num_syndrome_classes, num_faults)
            subsample: if True, we randomly subsample to reduce the row weight and ensure the full column rank

            A is built from the check matrix so that each row is given by the product of detectors (or the measurement group)
        '''
        # detector_samples = self.detector_samples
        check_matrix= self.check_matrix
        # get the 0/1 strings 
        # stab_sets = [''.join(b) for b in itertools.product('01', repeat=check_matrix.shape[0]) if '1' in b]
        if self.subsample:
            # sample_stabs = random.sample(stab_sets, k=2 * check_matrix.shape[1])
            # sample_stabs = [format(i, f'0{check_matrix.shape[0]}b') for i in random.sample(range(1, 1 << check_matrix.shape[0]), 2 * check_matrix.shape[1])]
            sample_stabs = list({format(random.randrange(1, 1 << check_matrix.shape[0]), f'0{check_matrix.shape[0]}b') for _ in range(10 * check_matrix.shape[1])})[: 2 * check_matrix.shape[1]]
        else: 
            stab_sets = [''.join(b) for b in itertools.product('01', repeat=check_matrix.shape[0]) if '1' in b]
            sample_stabs = stab_sets
        
        A_syndrome = np.zeros((len(sample_stabs), check_matrix.shape[1]), dtype=int)

        for i, stab in enumerate(sample_stabs):

            indices = [j for j, bit in enumerate(stab) if bit == '1']
            sub_check = check_matrix[indices, :]
            vals = np.sum(sub_check, axis=0) % 2. #take it to the binary form
            A_syndrome[i, :] = vals
        
        return A_syndrome, sample_stabs



    def _get_syndrome_expectations(self,
                                   sample_stabs: List[str] ) -> np.ndarray:
        '''
            Get the syndrome expectation values from the detector samples

            sample_stabs: the randomly subsampled stabilizers as strings from their generators
        '''
        detector_samples = self.detector_samples
        print(f'the detector shape is: {detector_samples.shape}')
        syndrome_eig_vals = np.zeros((len(sample_stabs),))
        for i, stab in enumerate(sample_stabs):
            indices = [j for j, bit in enumerate(stab) if bit == '1']
            sub_samples_stab = detector_samples[:,indices]
            sub_samples_stab = (-1)**sub_samples_stab
            column_products = np.prod(sub_samples_stab, axis=1)
            syndrome_eig_vals[i] = np.sum(column_products) / detector_samples.shape[0]
        
        return syndrome_eig_vals

            

        

    
    def predict_priors(self,
                       A_syndrome: np.ndarray,
                       syndrome_eig_vals: np.ndarray,
                       mode: str= 'direct' # 'direct' or 'rip'
                       ) -> np.ndarray:

        '''

            Predict the priors from the syndrome expectation values

        '''
        log_syndrome_eig_vals = -np.log(syndrome_eig_vals + 1e-10)

        if mode=='direct':

            AtA = A_syndrome.transpose() @ A_syndrome
            Ata_inv = np.linalg.inv(AtA)
            log_priors = Ata_inv @ A_syndrome.transpose() @ log_syndrome_eig_vals

           

            
        elif mode=='rip': 
            H_syndrome = np.ones(A_syndrome.shape) - 2 *A_syndrome
            H_syndrome = np.hstack((H_syndrome, np.ones((H_syndrome.shape[0], 1))))
            HtH_syndrome = (H_syndrome.T @ H_syndrome)
            log_priors = 2 * np.linalg.inv(HtH_syndrome) @ H_syndrome.T  @ log_syndrome_eig_vals 
            log_priors = -log_priors[:-1]

        priors = (1- np.exp(-log_priors))/2
        return priors 
         
    
    #ToDo: implement the leading order
    def predict_logical_error(self,
                              decoder: Callable,
                              observables_matrix: Union[np.ndarray, sparse.csr_matrix],
                              priors: np.ndarray, 
                              max_order: Union[int, None]=None,  
                              ) -> float:
        '''
        
            Predict the logical error rate from a fixed decoder correction and the observable matrix

            decoder: Callable the decoder class 
            observables_matrix: shape (num_logical_measurements, num_faults)
            priors: shape(num_faults,)
            max_order: (Optional) the maximum order to suppression of the physical error rate. If None, then we compute the exact result. 

            Note that the 
            check_matrix: also have shape (num_detectors, num_faults)
        
        '''
        check_matrix = self.check_matrix
        if max_order is None: 
            bit_rows = [bits for bits in itertools.product((0, 1), repeat=check_matrix.shape[1])] # all possible faults
            faults = np.array(bit_rows, dtype=int).T
        else:
            bit_rows = [bits for bits in itertools.product((0, 1), repeat=check_matrix.shape[1]) if 0 <= sum(bits) <= max_order]
            faults = np.array(bit_rows, dtype=int).T

        logical_error_rate = 0.0
        # now get the decoder correction for all possible syndromes appeared from the priors 
        space_time_code_params = {'H': check_matrix, 'L': observables_matrix, 'channel_probs': priors}
        decoder.set_decoder(space_time_code_params)
        detector_vals = (check_matrix @ faults) % 2  # shape (num_detectors, num_faults)
        decoder_correction = decoder.decode(detector_vals.T)  # shape (num_faults


        for i in range(faults.shape[1]):
            fault = faults[:, i]
            if not np.all((observables_matrix @ fault + observables_matrix @ decoder_correction[i,:]) % 2 == 0):
                prob = 1.0
                for j in range(len(fault)):
                    if fault[j] == 1:
                        prob *= priors[j]
                    else:
                        prob *= (1 - priors[j])
                logical_error_rate += prob
        return logical_error_rate
    

    # def predict_logical_error_sampler(self,
    #                                decoder: Callable,
    #                                observables_matrix: Union[np.ndarray, sparse.csr_matrix],
    #                                priors: np.ndarray,) -> float:
    #     '''

    #     We construct the sampler to compute the logical error rate
    #     using importance sampling.
    #     '''



    def predict_logical_error_efficient(self,
                                    decoder,
                                    observables_matrix,
                                    priors,
                                    max_order=None,
                                    chunk_size: int = 1 << 18,   # ~262k faults per chunk
                                    use_numba: bool = True) -> float:
        """
        Fast (parallel) logical error estimator.

        * Streams through the fault space in chunks to bound memory.
        * Uses bit-packed faults + Numba to compute syndromes, logicals and probabilities.
        * Decodes each unique syndrome exactly once and reuses it across chunks.

        Falls back to the original NumPy-vectorized path if:
        - numba is not available,
        - or number of faults E > 63,
        - or use_numba=False.

        Args:
            decoder:  object with set_decoder({...}) and decode(syndromes) -> corrections
            observables_matrix: O×E matrix (np.ndarray or scipy.sparse), entries in {0,1}
            priors:  (E,) physical fault probabilities in [0,1]
            max_order:  Optional[int]; if set, only weights ≤ max_order are enumerated
            chunk_size: number of faults processed per chunk (numba path)
            use_numba:  enable/disable the numba-accelerated streaming backend

        Returns:
            float: logical error probability (LEP)
        """
        # --- Normalize inputs ---
        H = (self.check_matrix.astype(np.uint8) & 1)   # D×E
        L = observables_matrix.toarray() if sparse.issparse(observables_matrix) else np.asarray(observables_matrix)
        L = (L.astype(np.uint8) & 1)                   # O×E
        p = np.asarray(priors, dtype=np.float64)       # (E,)

        D, E = H.shape
        O = 0 if L.size == 0 else L.shape[0]
        if O == 0:
            return 0.0

        # Clamp priors to avoid log under/overflow
        p_clip  = np.clip(p,         1e-300, 1.0 - 1e-12)
        q_clip  = np.clip(1.0 - p,   1e-300, 1.0)
        weights = np.log(p_clip) - np.log(q_clip)      # (E,)
        const   = float(np.log(q_clip).sum())          # scalar

        # Always (re)initialize the decoder with the channel
        decoder.set_decoder({'H': H, 'L': L, 'channel_probs': p})

        # Helper: pure NumPy fallback (your original vectorized approach)
        def _fallback_numpy():
            if max_order is None:
                if E <= 22:
                    nums   = np.arange(1 << E, dtype=np.uint32)
                    faults = ((nums[:, None] >> np.arange(E, dtype=np.uint32)) & 1).T.astype(np.uint8)  # E×N
                else:
                    raise ValueError(f"E={E} too large for full enumeration; set max_order or use_numba streaming.")
            else:
                cols = [np.zeros(E, dtype=np.uint8)]
                for w in range(1, max_order + 1):
                    for comb in itertools.combinations(range(E), w):
                        col = np.zeros(E, dtype=np.uint8); col[list(comb)] = 1
                        cols.append(col)
                faults = np.stack(cols, axis=1) if cols else np.zeros((E, 1), dtype=np.uint8)

            # Deduplicate syndromes once
            alphas = (H @ faults) % 2                                # D×N
            uniq, inverse = np.unique(alphas.T, axis=0, return_inverse=True)

            ehat_unique = decoder.decode(uniq)                       # (U×E)
            ehat_all    = ehat_unique[inverse, :].T.astype(np.uint8) # (E×N)

            lam     = (L @ faults)    % 2                            # (O×N)
            lam_hat = (L @ ehat_all) % 2                             # (O×N)

            # Per-fault probabilities (log-space trick)
            log1mp = np.log(q_clip)                                  # (E,)
            logp1  = np.log(p_clip)                                  # (E,)
            wts    = logp1 - log1mp                                  # (E,)
            cst    = float(log1mp.sum())
            logp_vec = faults.T @ wts + cst                          # (N,)
            probs    = np.exp(logp_vec)

            fail_mask = (lam ^ lam_hat).any(axis=0)
            return float(probs[fail_mask].sum())

        # Decide backend
        use_streaming = (use_numba and _NUMBA_AVAILABLE and E <= 63)

        if not use_streaming:
            return _fallback_numpy()

        # --- Streaming numba path (bit-packed, parallel inner loop) ---
        row_masks_H = _build_row_masks(H)  # (D,) uint64
        row_masks_L = _build_row_masks(L)  # (O,) uint64

        # Fault enumerator: either all faults or weight-restricted
        def _fault_chunks():
            if max_order is None:
                total = (1 << E)
                for base in range(0, total, chunk_size):
                    hi = min(base + chunk_size, total)
                    yield np.arange(base, hi, dtype=np.uint64)
            else:
                buf = []
                flush = lambda b: (np.array(b, dtype=np.uint64), b.clear())[0]
                for val in _ints_for_weights_leq(E, max_order):
                    buf.append(val)
                    if len(buf) >= chunk_size:
                        yield flush(buf)
                if buf:
                    yield np.array(buf, dtype=np.uint64)

        lep_sum = 0.0
        # Map: syndrome_key (bytes) -> lam_hat row (uint8, shape (O,))
        syn2_lamhat = {}

        for nums in _fault_chunks():
            # Compute syndromes, logicals, and probabilities for this chunk
            synd_chunk, lam_chunk, logp_chunk = _compute_chunk(
                nums, row_masks_H, row_masks_L, weights, const, E
            )
            probs = np.exp(logp_chunk)

            # Unique syndromes within the chunk
            uniq_s, inv = np.unique(np.ascontiguousarray(synd_chunk), axis=0, return_inverse=True)
            keys = _rows_to_keys(uniq_s)

            # Decode any previously unseen syndromes (batch once per chunk)
            new_rows, new_keys, new_idx = [], [], []
            for u, k in enumerate(keys):
                if k not in syn2_lamhat:
                    new_rows.append(uniq_s[u])
                    new_keys.append(k)
                    new_idx.append(u)

            if new_rows:
                new_rows_arr = np.array(new_rows, dtype=np.uint8)          # (K,D)
                ehat_new     = decoder.decode(new_rows_arr)                # (K,E)
                # lam_hat_new: (K,O)
                lam_hat_new  = ((L @ ehat_new.T) % 2).T.astype(np.uint8)
                for k, vec in zip(new_keys, lam_hat_new):
                    syn2_lamhat[k] = vec

            # Gather lam_hat per row using inverse index
            lam_hat_uniq = np.vstack([syn2_lamhat[k] for k in keys])      # (U,O)
            lam_hat_all  = lam_hat_uniq[inv]                               # (Nc,O)

            fail_mask = (lam_chunk ^ lam_hat_all).any(axis=1)
            lep_sum  += float(probs[fail_mask].sum())

        return lep_sum
                