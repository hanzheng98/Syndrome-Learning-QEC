import stim
import galois
import numpy as np
import time
import random
import pickle 
import itertools
import os
import sys
from scipy import sparse
from typing import Dict, List, Optional, Union, Tuple
from sim_qec.codes_family.classical_codes import cyclic_square_matrix
from sim_qec.codes_family.hpc_lp import HGP
from sim_qec.legacy.decoders import MLEDecoder # for future impelmentation, add Decoder base class from which to inhert 
from sim_qec.utils import (
    measure_qubits,
    reverse_circuit,
    add_circuits,
    reset_ancillas,
    generate_binary_strings,
    list_to_string,
    find_LI_rows,
    measurement_to_vector,
    unit_vectors_not_in_span,
    find_logical_operators,
    get_parity_check_matrix,
    str_to_vector,
    extract_physical_errors,
    write_pauli_data,
    read_pauli_data,
    check_y_errors,
    pauli_to_symplectic, 
    get_parity_check_matrix, 
    find_logical_operators, 
    symplectic_to_pauli
)
from sim_qec.analytic_log_channel import BaseAnalyticLogical, pauli_commutation_sign, AnalyticLogicalSyndrome
from sim_qec.legacy.pauli_character_basis import append_character_product, add_depolarizing_noise, partition_noise_syndrome

'''
The stabilizer circuit simulation using stim
written by Han Zheng
'''

GF2 = galois.GF(2)
PauliType = Tuple[str, str]





def initiate_logical_error_table(k:int):
    '''
    initialize the logical error table using the Pauli symbol encoding
    ''' 
    dictionary = {}
    paulis = ['I', 'X', 'Y', 'Z']
    all_errors = itertools.product(paulis, repeat=k)
    all_errors =[''.join(i) for i in all_errors]

    for error in all_errors:
        dictionary[error] = 0

    return dictionary





#Base class stim simulation on the simulation circuits with a given error-correcting codes
#Currently is the the skeleton model. Need to convert the detector error model
class BaseCircuit:

    def __init__(self,
                Hx: Union[np.ndarray, sparse.csr_matrix],
                Hz: Union[np.ndarray, sparse.csr_matrix],
                decoder: dict,
                Lx: Optional[Union[np.ndarray, sparse.csr_matrix]]=None,
                Lz: Optional[Union[np.ndarray, sparse.csr_matrix]]=None,
                #  phys_error: Union[dict[PauliType, float], None]=None,  
                 ):
       
        print(f'check CSS conditions: {np.all(Hx @ Hz.T ==0)}')

        self.num_qubits = Hx.shape[1]

        def _gf2_rows(matrix: np.ndarray, fallback_cols: int):
            rows = np.array(find_LI_rows(matrix), dtype=np.int8)
            if rows.size == 0:
                rows = np.zeros((0, fallback_cols), dtype=np.int8)
            return GF2(rows)

        # not necessarily needed though, used for stabilizer projections
        self.x_stab_rows = _gf2_rows(Hx, self.num_qubits)
        self.z_stab_rows = _gf2_rows(Hz, self.num_qubits)
        
       
        print(f'linearly independent rows in Hx is {len(self.x_stab_rows)}')
        print(f'linearly independent rows in Hz is {len(self.z_stab_rows)}')
        print('----------------------------------------------------------------')
        
        # self.phys_error = phys_error
        self.decoder = decoder  
        self.num_ancillas = int(len(self.x_stab_rows) + len(self.z_stab_rows)) #later change to a scheduling operation
        #first do X and Z: always fix this order
        self.X_anc = list(range(self.num_qubits, self.num_qubits + len(self.x_stab_rows)))
        self.Z_anc = list(range(self.num_qubits + len(self.x_stab_rows), self.num_qubits + len(self.x_stab_rows) + len(self.z_stab_rows)))
        # print('----------------------------------------------------------------')
        # print(f'The number of qubits is {self.num_qubits}')
        # self.X_anc = list(range(self.num_qubits, self.num_qubits + len(self.x_stab_rows)))
        # self.Z_anc = list(range(self.num_qubits + len(self.x_stab_rows), self.num_qubits + len(self.x_stab_rows) + len(self.z_stab_rows)))
        # print(f'The ancillas for X are {self.X_anc}')
        # print(f'The ancillas for Z are {self.Z_anc}')
        # print('----------------------------------------------------------------')
        # self.circuit = stim.Circuit()
        # The columns of these matrices are the stabilizers, logical operators, and the unit vectors that complete these into a full basis

    
    # 1) basic circuit operation 2) how to record the syndrome measurement from which to get the expectation

    def _full_stabilizer_sequence(self,
                                circuit: stim.Circuit, 
                                apply_noise: bool,
                                measure: bool,
                                error_model: Optional[Union[dict[PauliType, float], None]]=None,
                                error_rate: Optional[Union[float, None]]=0.01
                                ):
        '''
        Error model is a dict object whose key is the PauliType and the value is the error probability in the character basis
        
        '''

        if apply_noise:
            if error_model is None:
                if error_rate is None:
                    raise ValueError('must provide a (uniform) error rate for the depolarization ')
                else:
                    #add the single-qubit depolarization noise on data qubits
                    add_depolarizing_noise(circuit, range(self.num_qubits), error_rate=error_rate)
            else: 
                #use provided error model to apply the noise
                #Assume as the character basis
                append_character_product(circuit, error_model) 
            



        self._generate_stabilizer_circuits(circuit, mode='Z')
        self._generate_stabilizer_circuits(circuit, mode='X')

        if measure: 
            ancillas = self.X_anc + self.Z_anc
            measure_qubits(circuit ,ancillas)

          

    
    def _generate_stabilizer_circuits(self,
                                     circuit: stim.Circuit,
                                     mode: str='Z',
                                    ):  
        '''
        perform the stabilizer measurements 
        mode: "X" X-type stabilizer "Z" Z-type stabilizers
        '''
        if mode=='Z':
            stabilizers = self.z_stab_rows
            ancillas = self.Z_anc
        elif mode=='X':
            stabilizers = self.x_stab_rows
            ancillas = self.X_anc
        else:
            raise ValueError("Invalid mode. Use 'Z' or 'X'.")

        for i in range(len(ancillas)):
            anc= ancillas[i]
            stab_idx = np.nonzero(stabilizers[i])[0].tolist()
            # print(f'mode = {mode}')
            # print(f'stabilize index is {stab_idx}')
            # print(f'ancilla index is {anc}')
        

            circuit.append('H', [anc])
            for qubit in stab_idx:

                if mode=='Z':
                    circuit.append('CZ', [anc, qubit])
                    
                elif mode=='X':
                    circuit.append('CNOT', [anc, qubit])

            circuit.append('H', [anc])

    def _get_stabilizer_physreps(self,
                                 stab_label: str,
                                 ) -> PauliType:
        
        
        vec = GF2(np.array([int(bit) for bit in stab_label]))
        num_x = self.x_stab_rows.shape[0]
        num_z = self.z_stab_rows.shape[0]

        x_block = np.hstack((self.x_stab_rows.view(np.ndarray), np.zeros((num_x, self.num_qubits), dtype=int)))
        z_block = np.hstack((np.zeros((num_z, self.num_qubits), dtype=int), self.z_stab_rows.view(np.ndarray)))
        stab_basis = GF2(np.vstack((x_block, z_block)))


        # print(f'x dimensions: {self.x_stab_rows.shape}')
        # print(f'z dimensions: {self.z_stab_rows.shape}')
        stab_phys = vec @ stab_basis
        s_x = ''.join(str(bit) for bit in stab_phys[:self.num_qubits])
        s_z = ''.join(str(bit) for bit in stab_phys[self.num_qubits:])

        return (s_x, s_z)

    
    


class LogicalCircuit(BaseCircuit):
    '''
    Using stim to sample the circuit to obtain the logical channel
    
    '''
    

    def __init__(self, Hx, Hz, decoder, phys_error=None):
        super().__init__(Hx, Hz, decoder)
        self.phys_error = phys_error

        #get the full operator basis ready

        X_logical, Z_logical = find_logical_operators(Hx, Hz)
        self.k = len(X_logical)
        print(f'The logical X operators are {X_logical}')
        print(f'The logical Z operators are {Z_logical}')
        X_error_basis = unit_vectors_not_in_span(np.vstack((self.x_stab_rows, X_logical)))
        Z_error_basis = unit_vectors_not_in_span(np.vstack((self.z_stab_rows, Z_logical)))
        print('----------------------------------------------------------------')
        print(f'The X error basis shape is {X_error_basis.shape}')
        print(f'The Z error basis shape is {Z_error_basis.shape}')
        print(f'check rank of X error basis is {np.linalg.matrix_rank(X_error_basis)}')
        print(f'check rank of Z error basis is {np.linalg.matrix_rank(Z_error_basis)}')
        self.X_logical = X_logical
        self.Z_logical = Z_logical
        # The rows of these matrices are in order the stabilizers, logical operators, and the unit vectors that complete these into a full basis
        self.X_error_basis = X_error_basis
        self.Z_error_basis = Z_error_basis

        # Inverses are used to get a decomposition of the physical error
        X_basis_inverse = np.linalg.inv(X_error_basis)
        Z_basis_inverse = np.linalg.inv(Z_error_basis)
        self.X_error_basis_inverse = X_basis_inverse
        self.Z_error_basis_inverse = Z_basis_inverse

        bits = ["0", "1"]
        sx_list = ["".join(b) for b in itertools.product(bits, repeat=self.k)]
        sz_list = ["".join(b) for b in itertools.product(bits, repeat=self.k)]
        self.all_logical_labels = [(sx, sz) for sx in sx_list for sz in sz_list]
        self.logical_pauli_labels = [symplectic_to_pauli(lbl) for lbl in self.all_logical_labels]
        self.logical_label_to_index = {lbl: idx for idx, lbl in enumerate(self.all_logical_labels)}
    

    def sim_logicalchannel(self,
                           num_samples: int,
                           init_circuit: Optional[Union[stim.Circuit, None]]=None,
                           phys_error: Union[dict[PauliType, float], None]=None,
                           error_rate: Optional[Union[float, None]]=None,
                           num_shots: int=1
                           ) -> Dict[str, float]:
        '''
        Original sequential sampler for the logical channel.
        '''

        if init_circuit is None:
            circuit = stim.Circuit()
        else:
            circuit = init_circuit

        self._full_stabilizer_sequence(
            circuit=circuit,
            apply_noise=False,
            measure=False,
            error_model=phys_error,
            error_rate=error_rate,
        )

        log_error_table = initiate_logical_error_table(self.k)
        ancillas = list(range(self.num_qubits, self.num_qubits + self.num_ancillas))

        for _ in range(num_samples):
            noisy_circuit = stim.Circuit()
            self._full_stabilizer_sequence(
                circuit=noisy_circuit,
                apply_noise=True,
                measure=False,
                error_model=phys_error,
                error_rate=error_rate,
            )
            noisy_circuit = add_circuits(circuit, noisy_circuit)

            clean_circuit = stim.Circuit()
            clean_circuit = add_circuits(circuit, clean_circuit)
            clean_dagger = reverse_circuit(clean_circuit)
            physical_error_circuit = add_circuits(clean_dagger, noisy_circuit)

            extended_ancillas = ancillas + [x + len(ancillas) for x in ancillas]

            x_error, z_error, _ = extract_physical_errors(physical_error_circuit, range(self.num_qubits))

            measure_qubits(noisy_circuit, extended_ancillas)
            sampler = noisy_circuit.compile_sampler()
            syndrome_all = measurement_to_vector(sampler.sample(shots=num_shots)[0])
            syndrome_stab_noiseless = syndrome_all[:self.num_ancillas]
            syndrome_stab_noisy = syndrome_all[self.num_ancillas:2 * self.num_ancillas]
            syndrome = (GF2(syndrome_stab_noisy) + GF2(syndrome_stab_noiseless)).tolist()

            x_cor, z_cor = self.decoder[list_to_string(syndrome)]
            x_error = GF2(str_to_vector(x_cor)) + GF2(x_error)
            z_error = GF2(str_to_vector(z_cor)) + GF2(z_error)

            x_decomposition = (GF2(x_error) @ self.X_error_basis_inverse).tolist()
            z_decomposition = (GF2(z_error) @ self.Z_error_basis_inverse).tolist()

            x_decomposition = x_decomposition[len(self.x_stab_rows):len(self.x_stab_rows) + self.k]
            z_decomposition = z_decomposition[len(self.z_stab_rows):len(self.z_stab_rows) + self.k]
            logi_error: PauliType = (list_to_string(x_decomposition), list_to_string(z_decomposition))
            log_error_table[symplectic_to_pauli(logi_error)] += 1 / num_samples

        return log_error_table


#ToDo: be very careful on the ordering on the stabilizer measurement and the input sample_stabs: they must be consistent
class SyndromeExtractionCircuit(BaseCircuit):
    '''
    The code implements the syndrome extraction circuit and outputs the stabilizer expectation values instead of logical channels
    '''

    def __init__(self, Hx, Hz, decoder, sample_stabs: List[str]):
        super().__init__(Hx, Hz, decoder)

        self.sample_stabs = sample_stabs #record the samples stabs needed for the simulation: it needs to be full rank and checked before proceeding
    

    def sim_syndromeigs(self,
                        num_samples: int,
                        init_circuit: Optional[Union[stim.Circuit, None]]=None,
                        phys_error: Union[dict[PauliType, float], None]=None,
                        error_rate: Optional[Union[float, None]]=None, 
                        num_shots: int=1
                        ) -> Dict[PauliType, float]:
        '''

        Original sequential sampler for stabilizer expectation values.
        '''

        if init_circuit is None:
            circuit = stim.Circuit()
        else:
            circuit = init_circuit

        sample_stabs_eigs: Dict[PauliType, float] = {}
        for stab in self.sample_stabs:
            sample_stabs_eigs[self._get_stabilizer_physreps(stab)] = 0.0

        self._full_stabilizer_sequence(
            circuit=circuit,
            apply_noise=False,
            measure=False,
            error_model=phys_error,
            error_rate=error_rate,
        )

        ancillas = list(range(self.num_qubits, self.num_qubits + self.num_ancillas))

        for _ in range(num_samples):
            noisy_circuit = stim.Circuit()
            self._full_stabilizer_sequence(
                circuit=noisy_circuit,
                apply_noise=True,
                measure=False,
                error_model=phys_error,
                error_rate=error_rate,
            )
            noisy_circuit = add_circuits(circuit, noisy_circuit)

            extended_ancillas = ancillas + [x + len(ancillas) for x in ancillas]
            measure_qubits(noisy_circuit, extended_ancillas)

            sampler = noisy_circuit.compile_sampler()
            syndrome_all = measurement_to_vector(sampler.sample(shots=num_shots)[0])
            syndrome_stab_noiseless = syndrome_all[:self.num_ancillas]
            syndrome_stab_noisy = syndrome_all[self.num_ancillas:2 * self.num_ancillas]
            syndrome = (GF2(syndrome_stab_noisy) + GF2(syndrome_stab_noiseless)).tolist()

            for stab in self.sample_stabs:
                val = sum(int(a) & int(b) for a, b in zip(stab, syndrome)) % 2
                stab_paulirep = self._get_stabilizer_physreps(stab)
                sample_stabs_eigs[stab_paulirep] += (-1) ** val

        for key in sample_stabs_eigs:
            sample_stabs_eigs[key] /= max(1, num_samples)

        return sample_stabs_eigs


if __name__ == '__main__':



    system_size = 7
    X_stabilizers = [[0, 1, 2, 3], [0, 1, 4, 5], [0, 2, 4, 6]]
    Z_stabilizers = [[0, 1, 2, 3], [0, 1, 4, 5], [0, 2, 4, 6]]

    X_stabilizers = [[3,4,5,6],[1,2,5,6],[0,2,4,6]]
    Z_stabilizers = [[3,4,5,6],[1,2,5,6],[0,2,4,6]] 

    Hx = get_parity_check_matrix(X_stabilizers, system_size)
    Hz = get_parity_check_matrix(Z_stabilizers, system_size) 

    Lx, Lz = find_logical_operators(Hx, Hz)
    print(f'Lx: {Lx}')
    print(f'Lz: {Lz}')
    

    # decoders= load_MLD_decoder('/home/hanzheng/qldpc/sim-LDPC/data/decoders_files/Steane-full-decoder.pickle')
    # decoders= load_MLD_decoder('./files/decoder_files/Steane-full-decoder.pickle') # use this new one
    with open('./../demo/files/decoder_files/Steane-full-decoder.pickle', 'rb') as file:
        decoder = pickle.load(file)
    mle_decoder = {}
    for key, val in decoder.items():
        mle_decoder[key] = pauli_to_symplectic(val)

    # analytical_channel = AnalyticLogical(Hx=Hx, Lx=Lx, Hz=Hz, Lz=Lz, num_qubits=system_size ,decoder=mle_decoder, error_rate=0.15).get_logical_eigvals(mode='probability')

    # print(analytical_channel)
    # Create an instance of AnalyticLogical
    analyticlogical = AnalyticLogicalSyndrome(
        Hx=Hx,
        Lx=Lx,
        Hz=Hz,
        Lz=Lz,
        num_qubits=system_size,
        decoder=mle_decoder,
        error_rate=0.1
    )
    w1_paulis  = [format(1 << i, f'0{system_size}b') for i in range(system_size)]
    error_dict = {}
    for pauli in w1_paulis:
        pauli_x = (pauli, '0' * system_size)
        pauli_z = ('0' * system_size, pauli)
        pauli_y = (pauli, pauli) 
        error_dict[pauli_x] = None
        error_dict[pauli_y] = None
        error_dict[pauli_z] = None
    phys_errors = error_dict


    #Step 3: sample the stabilizer to build the full-rank matrix
    A_syndromes, B_syndromes = partition_noise_syndrome(phys_errors=phys_errors, Hx=Hx, Hz=Hz)
    stab_sets = [''.join(b) for b in itertools.product('01', repeat=Hx.shape[0]+ Hz.shape[0]) if '1' in b] #all the stabilizers in r-bit strings exlcluding the all-zeros

    if 2 ** (Hx.shape[0]+ Hz.shape[0]) > 2 * len(A_syndromes):
        #we randomly subsample rows to ensure
        # sample_size = 2 * len(A_syndromes)
        sample_stabs = random.sample(stab_sets, k=2 * len(A_syndromes)) 
    else: 
        sample_stabs = stab_sets


    base_logical = BaseAnalyticLogical(Hx=Hx, Lx=Lx, Hz=Hz, Lz=Lz, num_qubits=system_size, decoder=mle_decoder)

    Amat, _ = base_logical.build_commutationtable(sample_stabs=sample_stabs,A_paulis=A_syndromes, B_paulis=B_syndromes)
    print(f'the rank of Amat: {np.linalg.matrix_rank(Amat)} and the shape: {Amat.shape}')

    start_time = time.perf_counter()
    syndrome_extract = SyndromeExtractionCircuit(Hx=Hx, Hz=Hz, decoder=decoder, sample_stabs=sample_stabs)
    sample_stabs_eigvals = syndrome_extract.sim_syndromeigs(num_samples=1000, error_rate=0.1)
    elapsed = time.perf_counter() - start_time
    print(f'sample eigenvalues from stim sampling: {sample_stabs_eigvals}')
    print(f'syndrome sampling runtime: {elapsed:.3f} seconds')

    print(''*30)

    sample_stabs_phys = [symplectic_to_pauli(syndrome_extract._get_stabilizer_physreps(stab)) for stab in syndrome_extract.sample_stabs]
    print(f'the list of sampled stabilziers: {sample_stabs_phys}')
    

            









    
