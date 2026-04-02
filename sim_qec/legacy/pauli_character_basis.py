import stim
import copy
import pickle
import galois
import numpy as np
import time
import random
import multiprocessing
import pickle 
import itertools
from scipy import sparse
import random 
from typing import Any, Callable, Dict, List, Optional, Type, Union, Tuple
from sim_qec.codes_family.classical_codes import cyclic_square_matrix
from sim_qec.codes_family.hpc_lp import HGP
from sim_qec.legacy.decoders import MLEDecoder # for future impelmentation, add Decoder base class from which to inhert 
from sim_qec.compute_equiclass import convert_simplectic_matrix, compute_logical_equivalence
from sim_qec.utils import pauli_to_symplectic, get_parity_check_matrix, find_logical_operators, symplectic_to_pauli
from sim_qec.walsh_hadamard import logical_convert_2_probability_numba
import concurrent.futures
import json



'''

Given a Pauli channel, convert it to the form written in the Pauli character basis.

1. perform the Fourier transform to the Fouerier coefficients
2. do the linear matrix conversion. Note the sign issues 
3. return the character channel in the Fourier basis 
4. (Optional) convert back to the Pauli basis

'''


PauliType = Tuple[str, str]
GF2 = galois.GF(2)


#Stim implementtation using the probability: for analytical derivation we supply with the Fourier
def _targets_from_xz(pauli: PauliType):
    (x, z) = pauli
    t = []
    for i, (xb, zb) in enumerate(zip(x, z)):
        if xb == '1' and zb == '1':
            t.append(stim.target_y(i))
        elif xb == '1':
            t.append(stim.target_x(i))
        elif zb == '1':
            t.append(stim.target_z(i))
    return t

def append_character_product(circuit: stim.Circuit,
                             error_model: dict[PauliType, float]):
    """model[(x_bits, z_bits)] = p_a; independent product over a."""
    for pauli, p in error_model.items():
        if p <= 0:
            continue
        t = _targets_from_xz(pauli)
        if t:
            circuit.append("CORRELATED_ERROR", t, p)


def add_depolarizing_noise(circuit: stim.Circuit,
                            qubits: List,
                            error_rate: float,
                            ):
        for qubit in qubits: 
            # Append a single-qubit depolarizing channel on the given qubit.
            # circuit.append("DEPOLARIZE1", [qubit], probability)
            random_number = random.uniform(0,1)
     
            if random_number < error_rate/4:
                #print('X', qubit)
                circuit.append_operation("X", [qubit])
                
            elif error_rate/4 <= random_number < error_rate/2:
                #print('Y', qubit)
                circuit.append_operation("Y", [qubit])

            elif error_rate/2 <= random_number < 3*error_rate/4:
                #print('Z', qubit)
                circuit.append_operation("Z", [qubit])


def partition_noise_syndrome(phys_errors: dict[PauliType, float],
                             Hx: np.ndarray,
                             Hz: np.ndarray,
                             ):
    
  

    x_block = np.hstack((Hx, np.zeros((Hx.shape), dtype=int)))
    z_block = np.hstack((np.zeros((Hz.shape), dtype=int), Hz))
    H = GF2(np.vstack((x_block, z_block)))
    A_syndromes = []
    B_syndromes = []
    seen = [] #A corresondes to the distinct syndromes

    for pauli in phys_errors.keys():
        pauli_x, pauli_z = pauli
        # print(f'pauli x and z: {pauli}')
        pauli_array = GF2(np.hstack((np.array(list(pauli_z), dtype=int), np.array(list(pauli_x), dtype=int)))) #reversed order. 
        syndrome_pauli = H @ pauli_array
        # print(f'syndrome: {syndrome_pauli}')
        syndrome_pauli =''.join(str(bit) for bit in syndrome_pauli)
        if syndrome_pauli not in seen:
            A_syndromes.append(pauli)
        else: 
            B_syndromes.append(pauli)
        seen.append(syndrome_pauli)
    return A_syndromes, B_syndromes





