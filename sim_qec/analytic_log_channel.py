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
from fractions import Fraction

'''
Implementation based on the anlytical formula on computing the effective logical channel and decodings. 
'''

GF2 = galois.GF(2)
PauliType = Tuple[str, str]
#ToDo: Clear the data type, only the output datatype in strings, the input datatype is in numpy datatype

def write_dict_to_file(dictionary: dict, file_name: str) -> None:
    """
    Write a dictionary to a text file using JSON formatting.
    
    Args:
        dictionary (dict): The dictionary to write.
        file_name (str): The name of the file to write to (e.g., "output.txt").
    """
    with open(file_name, 'w') as f:
        json.dump(dictionary, f, indent=4)

def load_MLD_decoder(file_name):
    with open(file_name, 'rb') as file:
        MLD_decoder = pickle.load(file)
    return MLD_decoder




def pauli_commutation_sign(sx1: str, sz1: str, sx2: str, sz2: str) -> int:
    """
    Given two Pauli operators in symplectic representation:
      P = (sx1, sz1) and Q = (sx2, sz2),
    returns 1 if they commute and -1 if they anti-commute.
    
    The operators commute if:
         (sx1 . sz2 + sx2 . sz1) mod 2 == 0,
    and anti-commute otherwise.
    
    Args:
        sx1: Binary string for the X-part of the first operator.
        sz1: Binary string for the Z-part of the first operator.
        sx2: Binary string for the X-part of the second operator.
        sz2: Binary string for the Z-part of the second operator.
    
    Returns:
        1 if the operators commute, -1 if they anti-commute.
    """
    dot1 = sum(int(a) * int(b) for a, b in zip(sx1, sz2)) % 2
    dot2 = sum(int(a) * int(b) for a, b in zip(sx2, sz1)) % 2
    symplectic_product = (dot1 + dot2) % 2
    
    # If the symplectic product is 0, they commute, so output 1.
    # If it is 1, they anti-commute, so output -1.
    return (-1)** symplectic_product



def check_correction_2_syndrome(correction: Tuple[str, str],
                                syndrome: str, 
                                H: np.array):
    '''
    Given a parity check matrix whose row is given by correction term 
    '''
    syn_vector = []
    n = int(H.shape[1]/2)
    for i in range(H.shape[0]):
        stab = H[i]
        sx = ''.join(str(bit) for bit in stab[:n])
        sz = ''.join(str(bit) for bit in stab[n:])
        if pauli_commutation_sign(correction[0]. correction[1], sx, sz) == -1:
            syn_vector.append(1)
        else: 
            syn_vector.append(0)
    syn_vector = ''.join(str(bit) for bit in syn_vector)
    if syn_vector != syndrome:
        print(f'syndrome: {syndrome}')
        print(f'correction terms: {correction}')
        raise ValueError('the correction terms does not agree with the syndrome')



class BaseAnalyticLogical:
    '''
    Base class for analytical logical noise channel computation 
    '''

    def __init__(self,
                 Hx: Union[np.ndarray, sparse.csr_matrix],
                 Lx: Union[np.ndarray, sparse.csr_matrix],
                 Hz: Union[np.ndarray, sparse.csr_matrix],
                 Lz: Union[np.ndarray, sparse.csr_matrix],
                 num_qubits: int,
                 decoder: Dict[str, PauliType],
                 ):
        """
        Args:
            Hx, Hz: CSS check matrices for X and Z (shape r_x x n and r_z x n).
            Lx, Lz: Logical generator matrices in symplectic form.
            num_qubits: n.
            decoder: maps syndrome bitstring (length r) -> (sx_corr, sz_corr), each length n.
            phys_errors: optional Pauli mixture over physical strings in symplectic form:
                         dict[((sx_phys, sz_phys))] = probability. If None, use
                         i.i.d. single-qubit depolarizing channel with rate `error_rate`.
            error_rate: depolarizing rate p (per qubit) used when phys_errors is None.
        """
        self.Hx = Hx
        self.Hz = Hz
        if self.Hx is not None and self.Hz is not None:
            inter = self.Hx @ self.Hz.T
            ok = (inter.nnz == 0) if sparse.issparse(inter) else np.all(inter == 0)
            print(f'check CSS conditions: {ok}')

        # Convert CSS pieces into full (H,L) symplectic form and wrap in GF(2)
       
        H_np, L_np = convert_simplectic_matrix(Hx, Hz, Lx, Lz)
        self.H = GF2(H_np)
        self.L = GF2(L_np)

        # Dimensions
        self.r = H_np.shape[0]             # number of stabilizer generators
        self.k = L_np.shape[0] // 2        # number of logical qubits
        self.n = num_qubits

        

        print('----------------------------------------------------------------')
        print('defining projections')
        self.decoder = decoder
        print('----------------------Finishing initialization------------------------------------------')

    def _debug_logical(self,
                       ):
        '''
        debug effective logical channel so that it is trace-perserving

        Output: a table where each column is a stabilizer and each row is a syndrome class 
        so in total 2**r x 2**r matrix

        '''

        # p_identity = ('0' * self.k, '0' * self.k)
        syndrome_sets = [''.join(bits) for bits in itertools.product('01', repeat=self.r)]
        
        stab_tables = np.zeros((2**self.r, 2**self.r), dtype=int)
        for i, syndrome in enumerate(syndrome_sets):
            correction = self.decoder[syndrome]
            for j, stab in enumerate(syndrome_sets):
                # stab_phys = self._get_stabilizer_physreps(stab)

                # s_x = ''.join(str(bit) for bit in stab_phys[:self.n])
                # s_z = ''.join(str(bit) for bit in stab_phys[self.n:])
                s_x, s_z = self._get_stabilizer_physreps(stab)
                sign = pauli_commutation_sign(s_x, s_z, correction[0], correction[1])
                # sign = (-1)**(sum(int(a) | int(b) for a, b in zip(syndrome, stab)))
                stab_tables[i, j] = sign

        return stab_tables

    def _pauli_addition(self,
                        P1: PauliType,
                        P2: PauliType) -> PauliType:
        P1x, P1z = P1
        P2x, P2z = P2

        if len(P1x) != len(P1z) or len(P1x) != len(P2x) or len(P2x) != len(P2z): 
            raise ValueError('inputs P1 and P2 are of different dimensions')
        qx = format(int(P1x, 2) ^ int(P2x, 2), f'0{self.n}b')
        qz = format(int(P1z, 2) ^ int(P2z, 2), f'0{self.n}b')
        return (qx, qz)


    def _get_logical_physreps(self,
                              p: Tuple[str, str]):
        '''
        given a logical operator in (k, k) symplectic repersentation 
        return a logical operator in (n, n) representation
        '''
        sx, sz = p
        vec = GF2(np.array([int(bit) for bit in sx + sz]))
        log_phys = vec @ self.L
        l_x = ''.join(str(bit) for bit in log_phys[:self.n])
        l_z = ''.join(str(bit) for bit in log_phys[self.n:])

        return (l_x, l_z)
    
    def _get_stabilizer_physreps(self,
                                 p: str) -> PauliType:
        
        
        vec = GF2(np.array([int(bit) for bit in p]))
        stab_phys = vec @ self.H
        s_x = ''.join(str(bit) for bit in stab_phys[:self.n])
        s_z = ''.join(str(bit) for bit in stab_phys[self.n:])

        return (s_x, s_z)
    
    def _get_commutation(self,
                         P1: Tuple[str, str], 
                         P2: Tuple[str, str])-> int: 
        P1x, P1z = P1
        P2x, P2z = P2
        return ((int(P1x, 2) & int(P2z, 2)).bit_count() + (int(P1z, 2) & int(P2x, 2)).bit_count()) % 2
    



    def _get_syndrome_projections(self,
                                  syndrome: Union[str, List],
                                  ):
        '''
        Return the relative signs of stabilziers for a given syndrome projection
        '''
        result = {}
        # Iterate over all possible r-bit binary strings.
        for bits in itertools.product('01', repeat=self.r):
            key = ''.join(bits)
            # Count positions where both syndrome and key are '1'
            count = sum(1 for i in range(self.r) if syndrome[i] == '1' and key[i] == '1')
            result[key] = -1 if count % 2 == 1 else 1
        return result 
    

    # def _get_single_depolarization(self,
    #                                error_rate: float, 
    #                                mode: str ='eigenvalues'):
        
    #     bits = ['0', '1']
    #     # Generate all n-bit strings for s_x and s_z.
    #     sx_list = [''.join(bits_tuple) for bits_tuple in itertools.product(bits, repeat=self.n)]
    #     sz_list = [''.join(bits_tuple) for bits_tuple in itertools.product(bits, repeat=self.n)]
    #     phys_errors = {}
    #     if mode == 'eigenvalues':
    #         for sx in sx_list:
    #             for sz in sz_list:
    #                 weight = sum(int(a) | int(b) for a, b in zip(sx, sz))
    #                 phys_errors[(sx, sz)] = ((1-error_rate)**weight) 

    #         return  phys_errors 
            


    #     else:
    #         for sx in sx_list:
    #             for sz in sz_list:
    #                 weight = sum(int(a) | int(b) for a, b in zip(sx, sz))
    #                 phys_errors[(sx, sz)] = ((error_rate/4)**weight) * (1- 3 *error_rate/4)**(self.n - weight) 

    #         return phys_errors
    
    def _get_phys_eigvals(self,
                          pauli: Tuple[str, str]):
        sx, sz = pauli
        weight = sum(int(a) | int(b) for a, b in zip(sx, sz))
        return ((1-self.error_rate)**weight) 
    

    def _logical_convert_2_probability(self,
                                erorr_eignvalues: dict):
        '''
        convert logical pauli eigenvalues channel to erorr rate

        error_eigenvalues: dictionary object whose keys are symplectic representation of k-qubit logical basis and values is the corresponding pauli eigenvalues

        
        Output: 

        error_rate: dict object whose keys are symplectyic representation of k-qubit logical basis and values is the error rate
        '''
        
        logical_error_rate = {}

        bits = ['0', '1']
        # Generate all n-bit strings for s_x and s_z.
        sx_list = [''.join(bits_tuple) for bits_tuple in itertools.product(bits, repeat=self.k)]
        sz_list = [''.join(bits_tuple) for bits_tuple in itertools.product(bits, repeat=self.k)]

        for sx in sx_list:
            for sz in sz_list:
                p = (sx, sz)
                error_rate_p = 0
                # print(erorr_eignvalues)
                for p1, eig1 in erorr_eignvalues.items():
                    p1 = pauli_to_symplectic(p1)
                    error_rate_p += pauli_commutation_sign(p[0], p[1], p1[0], p1[1]) * eig1 
                logical_error_rate[symplectic_to_pauli(p)] = error_rate_p / (4 ** self.k)
        
        return logical_error_rate
    

    def build_commutationtable(self,
                            sample_stabs: List[PauliType],
                            A_paulis: List[PauliType],
                            B_paulis: List[PauliType]):
        '''

        build the commuattion matrix D_M between measurements and pauli errors for both distinct syndrome parts and duplicate parts
        
        '''

        A_mat = np.zeros((len(sample_stabs), len(A_paulis)), dtype=int)
        B_mat = np.zeros((len(sample_stabs), len(B_paulis)), dtype=int)
        for i, stab in enumerate(sample_stabs):
            if isinstance(stab, tuple) and len(stab) == 2:
                stab_pauli = stab
            else:
                stab_pauli = self._get_stabilizer_physreps(stab) 
            for j, pauli_a in enumerate(A_paulis):
                
                comm_val = self._get_commutation(stab_pauli, pauli_a)
                A_mat[i, j] = comm_val
            
            for k, pauli_b in enumerate(B_paulis):
                comm_val = self._get_commutation(stab_pauli, pauli_b)
                B_mat[i, k] = comm_val
        #check that A_mat must be of full column rank
        rank_A = np.linalg.matrix_rank(A_mat)
        if rank_A != A_mat.shape[1]:
            print(f'the matrix Amat and its rank: {A_mat}, {A_mat.shape}, {np.linalg.matrix_rank(A_mat)}')
            raise ValueError
            
        return A_mat, B_mat


class AnalyticLogical(BaseAnalyticLogical):

    def __init__(self,
                 Hx: Union[np.ndarray, sparse.csr_matrix],
                 Lx: Union[np.ndarray, sparse.csr_matrix],
                 Hz: Union[np.ndarray, sparse.csr_matrix],
                 Lz: Union[np.ndarray, sparse.csr_matrix],
                 num_qubits: int,
                 decoder: dict,
                 phys_errors: dict =None,
                 error_rate: float=0.05,
                #  approximation: bool =False,
                 ):
        
        '''
        decoders: dict with key is binary string of syndromes and values are the corresponding corrections (n-bits)
                For example: syndrome "0001" and correction  "XXIIII"
        
        phys_errors: dict with key as simplectic rep of pauli operators and value is the corresponding error
                If none: then we just take as single-qubit depolarizing noise
        '''

        super().__init__(Hx, Lx, Hz, Lz, num_qubits, decoder)
        # Physical channel description
        self.phys_errors = phys_errors
        self.error_rate = error_rate  # used if phys_errors is None
        

    @staticmethod
    def _syndrome_worker(
        args: Tuple["AnalyticLogical", Tuple[str, str], List[str]]
    ) -> float:
        """
        A static method used by the parallel summation. It sums contributions
        for one chunk of the syndrome set for the given logical operator.

        args is a tuple: (instance, pauli_l, sub_syndromes)
          - instance: The AnalyticLogical object (picklable).
          - pauli_l: (sx, sz) for one logical operator.
          - sub_syndromes: a subset of the entire syndrome set to process.
        
        Returns: The partial sum of all contributions from sub_syndromes.
        """
        instance, pauli_l, sub_syndromes = args
        part_sum = 0.0
        for syn in sub_syndromes:
            part_sum += instance._persyndrome_logical_eigvals(pauli_l, syn)
        return part_sum

    def sum_syndromes_in_parallel(
        self, pauli_l: Tuple[str, str], syndrome_sets: List[str], max_workers: int = 4
        ) -> float:
        """
        Sums the contributions for a single logical operator 'pauli_l'
        over the entire syndrome_sets in parallel. We chunk the syndrome list
        and farm out partial sums to multiple workers.

        Args:
            pauli_l: A tuple (sx, sz)
            syndrome_sets: list of all r-bit syndrome strings
            max_workers: number of processes/threads for concurrency

        Returns:
            The sum of _persyndrome_logical_eigvals(pauli_l, syn) over all syn in syndrome_sets.
        """
        # We can partition the syndrome_sets into chunks:
        num_syndromes = len(syndrome_sets)
        chunk_size = max(1, num_syndromes // (max_workers * 2))

        # Build tasks: each is (self, pauli_l, sub_syndromes).
        tasks = []
        start = 0
        while start < num_syndromes:
            end = min(start + chunk_size, num_syndromes)
            sublist = syndrome_sets[start:end]
            tasks.append((self, pauli_l, sublist))
            start = end

        partial_sums = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._syndrome_worker, t) for t in tasks]
            for fut in concurrent.futures.as_completed(futures):
                partial_sums.append(fut.result())

        return sum(partial_sums)

    def get_logical_eigvals_parallel_syndrome(
        self, mode: str = "eigenvalues", max_workers: int = 4
        ) -> Dict[Tuple[str, str], float]:
        """
        Example method that parallelizes across syndrome sets for each logical operator,
        sequentially over all logical operators.

        Potentially, this is nested concurrency if you also parallelize
        across logical operators. So watch out for overhead.

        Args:
            mode: "eigenvalues" or "probability"
            max_workers: concurrency level for summing over syndromes

        Returns:
            Dictionary mapping (sx, sz) -> float (the effective eigenvalue or probability).
        """
        # Build all logical operators of length k
        bits = ["0", "1"]
        sx_list = ["".join(b) for b in itertools.product(bits, repeat=self.k)]
        sz_list = ["".join(b) for b in itertools.product(bits, repeat=self.k)]
        all_logicals = [(sx, sz) for sx in sx_list for sz in sz_list]

        # Build the syndrome set of length r
        syndrome_sets = ["".join(b) for b in itertools.product("01", repeat=self.r)]

        eff_logical_eigvals = {}
        for pauli_l in all_logicals:
            # Sum contributions in parallel across the syndrome set
            total = self.sum_syndromes_in_parallel(pauli_l, syndrome_sets, max_workers)
            # Divide by 2^r
            eff_logical_eigvals[symplectic_to_pauli(pauli_l)] = total / (2 ** self.r)

        if mode == "probability":
            return self._logical_convert_2_probability(eff_logical_eigvals)
        return eff_logical_eigvals


    def get_logical_eigvals(self,
                            mode: str = 'eigenvalues'):

        '''
        get the analytical simulation on the effective noise channel in terms of eigenvalues

        first: generate all the logical operators in (k, k) symplectic representation 
        if self.approximation is True: implement the approximation for weight 3 logical operators
        second: for each logical operator, compute the effective logical eigenvalues by summing over all the syndromes
        treat logical XXXIIII, XIIXXI, have the same effective logical eigenvalues 
        how to implement this efficiently

        mode: "eigenvalues" & "probability"
        '''
        bits = ['0', '1']
        # Generate all n-bit strings for s_x and s_z.
        sx_list = [''.join(bits_tuple) for bits_tuple in itertools.product(bits, repeat=self.k)]
        sz_list = [''.join(bits_tuple) for bits_tuple in itertools.product(bits, repeat=self.k)]
        syndrome_sets = [''.join(bits) for bits in itertools.product('01', repeat=self.r)]

        eff_logical_eigvals = {}
        for sx in sx_list:
            for sz in sz_list:
                pauli_l = (sx, sz)
                eigval =0
                for syndrome in syndrome_sets:
                    eigval += self._persyndrome_logical_eigvals(pauli_l, syndrome)
                eigval = eigval / (2**self.r)
                eff_logical_eigvals[symplectic_to_pauli(pauli_l)] = eigval

        if mode == 'probability':
            return self._logical_convert_2_probability(eff_logical_eigvals)
        
        return eff_logical_eigvals
    

    def _persyndrome_logical_eigvals(self,
                                     pauli_l: Tuple[str, str],
                                     syndrome: str):
        '''
        get the logical vals for syndrome information
        '''
        correction = self.decoder[syndrome]
        # print(f'what is the correction: {correction} and syndrome {syndrome}')
        # print(F'correction: {correction[0]}, {correction[1]}')
        # print(f'physical errors: {self.phys_errors}')
        stab_group_labels = [''.join(bits) for bits in itertools.product('01', repeat=self.r)]
        # using 
       
        
        pauli_physrep = self._get_logical_physreps(pauli_l)
        eigval = 0
        for stab in stab_group_labels:
            stab_physrep = self._get_stabilizer_physreps(stab)
            pauli_ls = self._pauli_addition(stab_physrep, pauli_physrep)
            # ls_x = ''.join(str(bit) for bit in pauli_ls[:self.n])
            # ls_z = ''.join(str(bit) for bit in pauli_ls[self.n:])
            eigval += pauli_commutation_sign(correction[0], correction[1], pauli_ls[0], pauli_ls[1]) * self._get_phys_eigvals(pauli_ls)
        return eigval
    

    # def get_logical_probability_postselect(self,
    #                                        mode: str = 'probability'):
        
    #     '''
    #     Obtain the logical probability by post-selecting the zero syndrome: obtain the effective logical eigenvalues conditioned on the zero syndrome.

    #     # given the channel (defaulted with depolarization channel), compute p0 the probability for all the syndrome classes as a normalization factor, 
    #     then compute P_l by summing over the stabilizer in cosets
    #     '''

    #     # current we only assume for the depolarization channel
    #     bits = ['0', '1']
    #     stab_group_labels = [''.join(bits) for bits in itertools.product('01', repeat=self.r)]
    #     sx_list = [''.join(bits_tuple) for bits_tuple in itertools.product(bits, repeat=self.k)]
    #     sz_list = [''.join(bits_tuple) for bits_tuple in itertools.product(bits, repeat=self.k)]

    #     log_channel = {} # the output logical cahnnel, convert to the strings for Pauli
    #     if mode == 'probability':
    #         p0 = 0
    #         for sx in sx_list:
    #             for sz in sz_list:
    #                 pauli_l = (sx, sz)
    #                 pauli_physrep = self._get_logical_physreps(pauli_l)
    #                 prob_l = 0
    #                 for stab in stab_group_labels:
    #                     stab_physrep = self._get_stabilizer_physreps(stab)
    #                     p_ls = GF2(pauli_physrep) + GF2(stab_physrep)
    #                     ls_x = ''.join(str(bit) for bit in p_ls[:self.n])
    #                     ls_z = ''.join(str(bit) for bit in p_ls[self.n:])
    #                     weight = sum(int(a) | int(b) for a, b in zip(ls_x, ls_z))
    #                     prob_ls = (1-(3 * self.error_rate)/4)**(self.n - weight) * (self.error_rate/4)**weight #for the single qubit depolarization channel
    #                     prob_l += prob_ls
    #                     # print(f'pauli_ls: {p_ls}')
    #                 log_channel[symplectic_to_pauli(pauli_l)] = prob_l 
    #                 p0 += prob_l 

    #         # normalize the logical channel
    #         for key, val in log_channel.items():
    #             log_channel[key] = val / p0
        
    #     if mode == 'eigenvalues':
    #         # p0 = 0
    #         for sx in sx_list:
    #             for sz in sz_list:
    #                 pauli_l = (sx, sz)
    #                 pauli_physrep = self._get_logical_physreps(pauli_l)
    #                 eigval_l = 0
    #                 for stab in stab_group_labels:
    #                     stab_physrep = self._get_stabilizer_physreps(stab)
    #                     p_ls = GF2(pauli_physrep) + GF2(stab_physrep)
    #                     ls_x = ''.join(str(bit) for bit in p_ls[:self.n])
    #                     ls_z = ''.join(str(bit) for bit in p_ls[self.n:])
    #                     eigval_l += self._get_phys_eigvals((ls_x, ls_z))
    #                 log_channel[symplectic_to_pauli(pauli_l)] = eigval_l 
    #                 # p0 += eigval_l 
    #         lambda0 = log_channel['I']
    #         for key, val in log_channel.items():
    #             log_channel[key] = val / lambda0
    #     return log_channel



def get_sydrome_expectations(phys_errors: dict[PauliType, float],
                            A_paulis: List[PauliType],
                            B_paulis: List[PauliType],
                            Amat: np.ndarray,
                            Bmat: np.ndarray):
        '''
         
        Return the syndrome expectation from the known physical noise model. 
        In real experiments (in stim) this should be measured directly and here we simply compute this
        
        '''

        #take the logarithm: here we assume that it is positive so no sign issues
        stab_eigs = [] 
        #First step: combine both A and B matrix
        Dmat = np.hstack((Amat, Bmat))
        # print(f'self.Apaulis: {self.Apaulis}')
        # print(f'self.Bpaulis: {self.Bpaulis}')
        
        for i in range(Dmat.shape[0]):
            eig_val = 1
            for j in range(Dmat.shape[1]):
                if Dmat[i, j] ==1: #we here assume the dtype to be int
                    if j < len(A_paulis): 
                        # print(f'error rate dict: {self.error_rate}')
                        # print(f'the index for self.Apualis: {self.Apaulis[j]}')
                        eig_val = eig_val * phys_errors[A_paulis[j]]
                    else:
                        eig_val = eig_val * phys_errors[B_paulis[j - len(A_paulis)]]
            stab_eigs.append(eig_val)
        
        return stab_eigs

#ToDo: add approximation methods using the fourier modes and perform
class AnalyticLogicalSyndrome(BaseAnalyticLogical):
    '''
    An analtical way to compute the logical noise channel via syndromes 
    '''

    def __init__(self,
                 Hx: Union[np.ndarray, sparse.csr_matrix],
                 Lx: Union[np.ndarray, sparse.csr_matrix],
                 Hz: Union[np.ndarray, sparse.csr_matrix],
                 Lz: Union[np.ndarray, sparse.csr_matrix],
                 num_qubits: int,
                 decoder: dict,
                 sample_stabs_eigs: Optional[dict] =None,
                 phys_errors: Optional[dict] =None,
                 error_rate: Optional[Union[float, None]]=0.05,
                 ):
        
        '''
        decoders: dict with key is binary string of syndromes and values are the corresponding corrections (n-bits)
                For example: syndrome "0001" and correction  "XXIIII"
        
        phys_errors: dict with key as simplectic rep of pauli operators and value is the corresponding error
                If none: then we just take as single-qubit depolarizing noise
        '''

        super().__init__(Hx, Lx, Hz, Lz, num_qubits, decoder)
        

        #if the physical error is not given, we assume it to be depolarization noise on single qubit with fxied error rate. 
        #write it in the character basis 

        if phys_errors is None and error_rate is not None:
            print(f'No physical error channel is provided, default to single-qubit depolarization in chracter basis')
            #create 2n symplectic bitstrings 
            w1_paulis  = [format(1 << i, f'0{self.n}b') for i in range(self.n)]
            error_dict = {}
            for pauli in w1_paulis:
                pauli_x = (pauli, '0' * self.n)
                pauli_z = ('0' * self.n, pauli)
                pauli_y = (pauli, pauli) 
                error_dict[pauli_x] = np.sqrt(1-error_rate)
                error_dict[pauli_y] = np.sqrt(1-error_rate)
                error_dict[pauli_z] = np.sqrt(1-error_rate)
            phys_errors = error_dict
        elif phys_errors is None and error_rate is None and sample_stabs_eigs is not None:
            print(f'No physical error channel is provided but given a subsampled stabilizer eigenvalues. Proceed with learning')
            #create 2n symplectic bitstrings 
            w1_paulis  = [format(1 << i, f'0{self.n}b') for i in range(self.n)]
            error_dict = {}
            for pauli in w1_paulis:
                pauli_x = (pauli, '0' * self.n)
                pauli_z = ('0' * self.n, pauli)
                pauli_y = (pauli, pauli) 
                error_dict[pauli_x] = error_rate
                error_dict[pauli_y] = error_rate
                error_dict[pauli_z] = error_rate
            phys_errors = error_dict
        elif phys_errors is not None and sample_stabs_eigs is not None:
            pass 
        else:
            print('not enough information is provided') 
            raise ValueError(f'physical error model is {phys_errors}, error rate is {error_rate}, and sampled syndrome expectations are {sample_stabs_eigs}')

        print(f'the phsyical error model: {len(phys_errors)}')
        #now we construct the detecton matrix by partitioning it with the syndrome class 
        print('' * 20)
        print('Step 1: Now we partition the error according to the syndromes')
        A_syndromes = []
        B_syndromes = []
        seen = [] #A corresondes to the distinct syndromes

        for pauli in phys_errors.keys():
            pauli_x, pauli_z = pauli
            # print(f'pauli x and z: {pauli}')
            pauli_array = GF2(np.hstack((np.array(list(pauli_z), dtype=int), np.array(list(pauli_x), dtype=int)))) #reversed order. 
            syndrome_pauli = self.H @ pauli_array
            # print(f'syndrome: {syndrome_pauli}')
            syndrome_pauli =''.join(str(bit) for bit in syndrome_pauli)
            if syndrome_pauli not in seen:
                A_syndromes.append(pauli)
            else: 
                B_syndromes.append(pauli)
            seen.append(syndrome_pauli)

        # A_paulis = {}
        # for i, pauli_a in enumerate(A_syndromes):
        #     A_paulis[i] = pauli_a
        # B_paulis = {}
        # for i, pauli_b in enumerate(B_syndromes):
        #     B_paulis[i] = pauli_b
        

        print('' * 20)
        print('Step 2: subsample stabilizers and build the full-rank matrix A')

        if sample_stabs_eigs is None:

            stab_sets = [''.join(b) for b in itertools.product('01', repeat=self.r) if '1' in b] #all the stabilizers in r-bit strings exlcluding the all-zeros

            if 2 ** self.r > 2 * len(A_syndromes):
                #we randomly subsample rows to ensure
                # sample_size = 2 * len(A_syndromes)
                sample_stabs = random.sample(stab_sets, k=2 * len(A_syndromes)) 
            else: 
                sample_stabs = stab_sets
        else:
            sample_stabs, stab_eigs = zip(*sample_stabs_eigs.items())
            sample_stabs = list(sample_stabs)
            stab_eigs = list(stab_eigs)


        # print(f'the error parameters with dinstinct syndrome: {self.Apaulis}')
        # print(f'the error parameters with duplicate syndrome: {self.Bpaulis}')
        Amat, Bmat = self.build_commutationtable(sample_stabs=sample_stabs, A_paulis=A_syndromes,B_paulis=B_syndromes) #Amat as q x K matrix
        # print(f'the matrix Amat and its rank: {self.Amat}, {self.Amat.shape}, {np.linalg.matrix_rank(self.Amat)}')
        # print(f'the Bmatrix: {self.Bmat}' )
        

        # #check whether or not self.A_mat have full column rank. If not, re-run the initialization. ToDo: implement this until-success within a single-run of initialization
        # if np.linalg.matrix_rank(self.Amat) != len(A_syndromes): 
        #     raise ValueError(f'Sampling failed with rank {np.linalg.matrix_rank(self.Amat)}')
        

        #now we compute (measure) the stabilizer eigenvalues
        print('' * 20)
        if sample_stabs_eigs is None:
            print('compute analytically the syndrome expectations if not given explicitly from experiments')
            stab_eigs = get_sydrome_expectations(phys_errors=phys_errors, A_paulis=A_syndromes, B_paulis=B_syndromes, Amat=Amat, Bmat=Bmat)
        else:
            # stab_eigs already extracted above when sample_stabs_eigs not None
            pass

        # sample_stabs_phys = [self._get_stabilizer_physreps(stab) for stab in self.sample_stabs]
        # print(f'the list of sampled stabilziers: {sample_stabs_phys}')
        # print(f' the computed stabilziers eigvals: {self.stab_eigs} ')
        bits = ["0", "1"]
        lx_list = ["".join(b) for b in itertools.product(bits, repeat=self.k)]
        lz_list = ["".join(b) for b in itertools.product(bits, repeat=self.k)]
        all_logical_labels = [(sx, sz) for sx in lx_list for sz in lz_list]
        print('-'*20 + 'initialization completed' + '-'*20)
        self.all_logical_labels = all_logical_labels
        self.sample_stabs = sample_stabs
        self.Apaulis = A_syndromes
        self.Bpaulis = B_syndromes
        self.stab_eigs = stab_eigs
        self.Amat = Amat
        self.Bmat = Bmat
        self.stab_eigs = stab_eigs

    

    
    

    def compute_logical_channel(self,
                                mode: str='probability'):

        '''

        user-provided decoders given the code to compute the logical probability or Pauli eiganvalues
        Default (and currently only supported) on the probability

        
        '''
        
        syndrome_labels = [''.join(bits) for bits in itertools.product('01', repeat=self.r)]

        probability_logis = np.zeros((4**(self.k), ))
        for syndrome in syndrome_labels:
            # correction = self.decoder[syndrome]
            # print(f'per sydrome probability: {self._compute_persyndrome_logical_channel(syndrome, mode)}')
            probability_logis += self._compute_persyndrome_logical_channel(syndrome, mode)
        
        logical_dict = logical_dict = {symplectic_to_pauli(label): val for label, val in zip(self.all_logical_labels, probability_logis)}

        
        return logical_dict






    def _compute_persyndrome_logical_channel(self,
                                             syndrome:str,
                                             mode: str='probability'):

        correction = self.decoder[syndrome]
        

        persyndrome_probs = np.zeros((len(self.all_logical_labels),))

        for i, logi_label in enumerate(self.all_logical_labels):
            logi_pauli = self._get_logical_physreps(logi_label)
            pauli = self._pauli_addition(correction, logi_pauli)
            persyndrome_probs[i] = self._compute_eff_distribution(pauli)
        
        return persyndrome_probs




            




    

    def _compute_eff_distribution(self,
                                    pauli: PauliType, 
                                    precision: Optional[Union[float, None]]=None): 
        '''
        
        compute the |G|p^eff(a) for any given a 
        1. Either it is an exact 
        2. We use sample to some (additive) precision #for gauge group, this is needed

        '''

        #generate all the gauge group: currently only supported stabilizers 
        stab_group_labels = [''.join(bits) for bits in itertools.product('01', repeat=self.r)]
        # Build all logical operators of length k
        bits = ["0", "1"]
        lx_list = ["".join(b) for b in itertools.product(bits, repeat=self.k)]
        lz_list = ["".join(b) for b in itertools.product(bits, repeat=self.k)]
        all_logical_labels = [(sx, sz) for sx in lx_list for sz in lz_list]

        if precision is None:
            #Assume that we have to do the exact way 
            prob = 0
            for stab in stab_group_labels:
                stab_pauli = self._get_stabilizer_physreps(stab)
                for logi in all_logical_labels:
                    logi_pauli = self._get_logical_physreps(logi)
                    pauli_b = self._pauli_addition(logi_pauli, stab_pauli)
                    commutation = self._get_commutation(pauli_b, pauli)
                    if (int(pauli_b[0], 2) | int(pauli_b[1], 2)) == 0:
                        eig_pauli_b = 1
                    else:
                        eig_pauli_b = self._compute_logical_eigs(pauli_b)
                    prob += eig_pauli_b * (-1)**commutation
            return prob / (2**(2 * self.k + self.r)) 
        else:
            pass 

        
    
    def _compute_logical_eigs(self,
                                pauli_l:PauliType,
                                pauli_s:PauliType=None,
                                method: Optional[str]='rip' ):
        
        '''
        method = 'direct' or 'rip'
        compute the logical Pauli eigenvalues

        Note that we use the A(A^TA)^{-1}l = e #change to another method 
        ''' 
        #step 1: determine the vector l
        if pauli_s is not None:
            pauli_log = self._pauli_addition(pauli_l, pauli_s)
        else: 
            pauli_log = pauli_l 
        la = np.zeros((len(self.Apaulis,)), dtype=float)
        for i, pauli_a in enumerate(self.Apaulis):
            if self._get_commutation(pauli_log, pauli_a) ==1:
                la[i] =1.0

        if method == 'direct':
            # step 2: determine e vector from the sampled syndrome vectors using the standard L2 method
            Amat = self.Amat.astype(float)
            # A_evals = np.linalg.eigvalsh(Amat)
            # print(f'the maximum A_evals: {A_evals[-1]}')
            Hmat = Amat.transpose() @ Amat
            Hinv = np.linalg.inv(Hmat)

            #check the condition number for debug
            evals = np.linalg.eigvalsh(Hmat)               # sorted ascending
            lam_min = np.sqrt(float(evals[0]))
            lam_max = np.sqrt(float(evals[-1]))
            cond_H = np.inf if lam_min <= 0 else lam_max / lam_min
            # print(f"λ_min(H)={lam_min:.6e}, λ_max(H)={lam_max:.6e}, cond_2(H)={cond_H:.6e}, shape of the matrix: {Hmat.shape}")

        
            e_vec = Amat @ Hinv @ la # this should be of shape q
        elif method == 'rip':  #to add this and compare with values of e and conditional number
            Amat = self.Amat.astype(float)
            Hada = np.ones(Amat.shape) - 2 * Amat
            ones_col = np.ones((Hada.shape[0], 1), dtype=Hada.dtype)
            Hada = np.hstack([ones_col, Hada])
            ones_row = np.ones((1, Hada.shape[1]), dtype=Hada.dtype)
            Hada = np.vstack([Hada,ones_row])
            # print(f'the dimension of the Hadamard: {Hada.shape}')
            Mmat = Hada.transpose() @ Hada
            #check the condition number for debug
            evals = np.linalg.eigvalsh(Mmat)               # sorted ascending
            lam_min = np.sqrt(float(evals[0]))
            lam_max = np.sqrt(float(evals[-1]))
            cond_M = np.inf if lam_min <= 0 else lam_max / lam_min
            # print(f"λ_min(H)={lam_min:.6e}, λ_max(H)={lam_max:.6e}, cond_2(H)={cond_M:.6e}, shape of the matrix: {Mmat.shape}")
            # print(np.vstack([np.zeros((1)), la]).shape)
            la = np.insert(la, 0, 0)
            tilde_e = 2 * Hada @ np.linalg.inv(Mmat) @ la
            e_vec = tilde_e[:-1] *(-1) 
            # print(f'using the RIP method, return the e_vec: {e_vec}')




             
        # print(f'decompostion vector: {e_vec}')
        # print(f'give the decomposition in fractions: {[Fraction(v).limit_denominator(1<<22) for v in e_vec]}')

        # # 2) convert to rationals (exact exponents you can carry symbolically)
        # def to_frac(v):
        # # snap near-integers exactly; otherwise nearest rational with bounded denominator
        #     r = Fraction(round(v)) if abs(v - round(v)) < 1e-10 else Fraction(v).limit_denominator(1<<20)
        #     return r

        # e_frac = [to_frac(v) for v in e_vec]

        log_eigval = 1
        for i in range(len(e_vec)):
            log_eigval = log_eigval * self.stab_eigs[i] ** e_vec[i] #this needs to be changed to better suited with data type
        
        return log_eigval


    
         
        


                 
         
         
          
         

        




        


        

        
    




            

if __name__ == "__main__":
    system_size = 7
    X_stabilizers = [[0, 1, 2, 3], [0, 1, 4, 5], [0, 2, 4, 6]]
    Z_stabilizers = [[0, 1, 2, 3], [0, 1, 4, 5], [0, 2, 4, 6]]

    X_stabilizers = [[3,4,5,6],[1,2,5,6],[0,2,4,6]]
    Z_stabilizers = [[3,4,5,6],[1,2,5,6],[0,2,4,6]] 

    Hx = get_parity_check_matrix(X_stabilizers, system_size)
    Hz = get_parity_check_matrix(Z_stabilizers, system_size) 

    Lx, Lz = find_logical_operators(Hx, Hz)
    # decoders= load_MLD_decoder('/home/hanzheng/qldpc/sim-LDPC/data/decoders_files/Steane-full-decoder.pickle')
    decoders= load_MLD_decoder('./../demo/files/decoder_files/Steane-full-decoder.pickle') # use this new one
    mle_decoder = {}
    for key, val in decoders.items():
        mle_decoder[key] = pauli_to_symplectic(val)

    # analytical_channel = AnalyticLogical(Hx=Hx, Lx=Lx, Hz=Hz, Lz=Lz, num_qubits=system_size ,decoder=mle_decoder, error_rate=0.15).get_logical_eigvals(mode='probability')
    
    # print(analytical_channel)
    # Create an instance of AnalyticLogical
    analytic_logical = AnalyticLogical(
        Hx=Hx,
        Lx=Lx,
        Hz=Hz,
        Lz=Lz,
        num_qubits=system_size,
        decoder=mle_decoder,
        error_rate=0.1
    )
    # postsel_logical_error_dict = analytic_logical.get_logical_probability_postselect(mode='eigenvalues')
    # print(f"Post-selected logical error probabilities:{postsel_logical_error_dict}")
    start_time = time.perf_counter()
    mle_logical_dict = analytic_logical.get_logical_eigvals_parallel_syndrome(mode='probability', max_workers=14)
    print(f"MLE logical error probabilities: {mle_logical_dict}")
    elapsed = time.perf_counter() - start_time
    print(f'syndrome sampling runtime: {elapsed:.3f} seconds')
    print(''*30)

    syndrome_logical = AnalyticLogicalSyndrome(
        Hx=Hx,
        Lx=Lx,
        Hz=Hz,
        Lz=Lz,
        num_qubits=system_size,
        decoder=mle_decoder,
        error_rate=0.1
    )
    start_time = time.perf_counter()
    syndrome_logical_dict = syndrome_logical.compute_logical_channel()
    print(f"logical error probabilities with syndrome: {syndrome_logical_dict}")
    elapsed = time.perf_counter() - start_time
    print(f'syndrome sampling runtime: {elapsed:.3f} seconds')


    

    # X_stabilizers = [[7,8,9,10,11,12,13,14], [3,4,5,6,11,12,13,14],[1,2,5,6,9,10,13,14], [0,2,4,6,8,10,12,14]]
    # Z_stabilizers = [[7,8,9,10,11,12,13,14], [3,4,5,6,11,12,13,14],[1,2,5,6,9,10,13,14], [0,2,4,6,8,10,12,14]]

    # Hx = get_parity_check_matrix(X_stabilizers, system_size)
    # Hz = get_parity_check_matrix(Z_stabilizers, system_size) 

    # Lx= GF2(np.array([
    #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    #     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    #     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
    #     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    #     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1]
    # ]))

    # Lz = Lx


    # decoders= load_MLD_decoder('/home/hanzheng/qldpc/sim-LDPC/data/decoders_files/15-7-3-full-decoder.pickle')
    # # decoders = {'00': 'II', '10': 'XI', '01': 'IX', '11': 'XX'}


    # mle_decoder = {}
    # for key, val in decoders.items():
    #     mle_decoder[key] = pauli_to_symplectic(val)

    # # Compute effective logical eigenvalues in parallel:
    # start_time = time.perf_counter()
    # print("Parallel effective logical eigenvalues:")
    # # mychannel = AnalyticLogical(Hx=Hx, Lx=Lx, Hz=Hz, Lz=Lz, num_qubits=system_size ,decoder=mle_decoder, error_rate=0.15).get_logical_eigvals_parallel_syndrome(mode='probability', max_workers=14)
    # # convert the representation to string Pauli 
    # # mychannel_str = {}
    # # for key, val in mychannel.items():
    # #     mychannel_str[symplectic_to_pauli(key)] = val 
    # #     # # Compute effective logical eigenvalues using the sequential method:
    # #     # eigenvals_seq = analytic_logical.get_logical_eigvals(mode='probability')
    # #     # print("Sequential effective logical eigenvalues:")
    # #     # print(eigenvals_seq)

    # # Suppose we have Hx, Lx, Hz, Lz, num_qubits, decoder, etc.
    # approx_channel = AnalyticLogical(
    #     Hx=Hx, Lx=Lx, Hz=Hz, Lz=Lz,
    #     num_qubits=system_size,
    #     decoder=mle_decoder,
    #     error_rate=0.05,
    #     # max_weight=7
    # )
    # # Then run:
    # approx_eigens = approx_channel.get_logical_eigvals_approx_with_grouping_parallel(
    #     # mode='probability',
    #     max_workers=8
    # )
    # # print(approx_results)

   
    # end_time = time.perf_counter()
    # print(f"Run time: {end_time - start_time:.2f} seconds")
    # approx_rates  = logical_convert_2_probability_numba(approx_eigens, 7, max_workers=8)
    # approx_rates_str = {}

    # for key, value in itertools.islice(approx_rates.items(), 10):
    #     print(key, value)
    # # print(approx_rates)
    # write_dict_to_file(approx_rates, '/home/hanzheng/qldpc/sim-LDPC/data/analytic_simulations/15-7-3_code_p0.05.txt')

    
