import stim
import copy
import galois
import numpy as np
import random
import multiprocessing
import pickle 
import itertools
from scipy import sparse
import random 
from typing import Any, Callable, Dict, List, Optional, Type, Union, Tuple
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
    extract_physical_errors
)
GF2 = galois.GF(2)






# todo: There seems to be some problem where given z syndromes cannot detect x errors and vice versa. 

# MLE decoder for the LDPC code, which serves as the basic class where we can build upon for other decoders
class MLEDecoder:
    # parallel implementation of the MLE decoder based on Argyris code 
    def __init__(self,
                 Hx: Union[np.ndarray, sparse.csr_matrix],
                 Hz: Union[np.ndarray, sparse.csr_matrix],
                 noise_model: str = "depolarizing",
                 ):
        self.Hx = Hx
        self.Hz = Hz
        print(f'check CSS conditions: {np.all(self.Hx @ self.Hz.T ==0)}')
        # not necessarily needed though, used for stabilizer projections
        self.x_stab_rows = GF2(np.array(find_LI_rows(Hx)))
        self.z_stab_rows = GF2(np.array(find_LI_rows(Hz)))
        
       
        print(f'linearly independent rows in Hx is {len(self.x_stab_rows)}')
        print(f'linearly independent rows in Hz is {len(self.z_stab_rows)}')
        print('----------------------------------------------------------------')
        self.noise_model = noise_model
        self.num_qubits = Hx.shape[1]
        self.num_ancillas = int(len(self.x_stab_rows) + len(self.z_stab_rows))
        print('----------------------------------------------------------------')
        print(f'The number of qubits is {self.num_qubits}')
        self.X_anc = list(range(self.num_qubits, self.num_qubits + len(self.x_stab_rows)))
        self.Z_anc = list(range(self.num_qubits + len(self.x_stab_rows), self.num_qubits + len(self.x_stab_rows) + len(self.z_stab_rows)))
        print(f'The ancillas for X are {self.X_anc}')
        print(f'The ancillas for Z are {self.Z_anc}')
        print('----------------------------------------------------------------')
        # self.circuit = stim.Circuit()
        # The columns of these matrices are the stabilizers, logical operators, and the unit vectors that complete these into a full basis

        X_logical, Z_logical = find_logical_operators(Hx, Hz)
        X_logical = GF2(X_logical) + GF2(Hx[2])
        Z_logical = GF2(Z_logical) + GF2(Hz[2])

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

        print('----------------------Finishing initilization------------------------------------------')


    
    def decode(self,
                circuit: stim.Circuit,
                noise_probability: float,
                shots: int = 1,
                num_processes: int = 1,
                num_samples: int = 1000,
                ):
        
        # stbailizer projections to the code space without noise and measurement
        # circuit = stim.Circuit()
        # full_stabilizer_sequence(circuit, self.X_stabilizers, self.Z_stabilizers, self.X_anc, self.Z_anc, measure=False, noise=False)
        self.full_stabilizer_sequence(circuit, noise_probability, measure=False, noise=False)
        # perfect_dagger = reverse_circuit(circuit)

        # Finish state preparation
        ancillas = self.X_anc + self.Z_anc
        # measure_qubits(circuit, ancillas)
        # reset_ancillas(circuit, ancillas) #so that we assume noiseless ancillas # we will only measure at the end


        # Create the MLE tables
        x_dicts, z_dicts = self._create_decoding_tables()

        for i in range(num_samples):
            noisy_circuit = stim.Circuit()
            # full_stabilizer_sequence(noisy_circuit, self.X_stabilizers, self.Z_stabilizers, self.X_anc, self.Z_anc, measure=False, noise=True)
            self.full_stabilizer_sequence(noisy_circuit, noise_probability, measure=False, noise=True)

            final_circuit = add_circuits(circuit, noisy_circuit)
            # measure_qubits(final_circuit, ancillas)
            


            # adding another new circuit for preparation 

            circuit2 = stim.Circuit()
            # full_stabilizer_sequence(circuit2, self.X_stabilizers, self.Z_stabilizers, self.X_anc, self.Z_anc, measure=False, noise=False)
            self.full_stabilizer_sequence(circuit2, noise_probability, measure=False, noise=False)
            circuit2 = add_circuits(circuit, circuit2)
            perfect_dagger = reverse_circuit(circuit2)

            physical_error_circuits = add_circuits(perfect_dagger, final_circuit)
            
            x_errors, z_errors, y_meas = extract_physical_errors(physical_error_circuits, list(range(self.num_qubits)))
            extended_ancillas = ancillas + [x + len(ancillas) for x in ancillas]
            # print(f' anciallas: {ancillas + ancillas}')
            measure_qubits(final_circuit, extended_ancillas)
            
            # print(f'The X error is {x_errors}')
            # print(f'The Z error is {z_errors}')
            # x_errors, z_errors = check_y_errors(x_errors, z_errors, y_meas)



            # physical_error_circuits = add_circuits(perfect_dagger, final_circuit)
            # add ancilla for each data qubit to measure the physical error

            # measure_qubits(physical_error_circuits, list(range(self.num_qubits)))

            
            

            #sample the circuit
            syndrome_all_sampler = final_circuit.compile_sampler()

            # Extract the syndrome and the physical errors
            syndrome_all = measurement_to_vector(syndrome_all_sampler.sample(shots = shots)[0]) # for all assuming shots=1, todo: increase multiple shots to directly estimate the moment of the syndrome
            # print(f'The syndrome is {syndrome_all}')
            # print(f'the length of the total syndrome is {len(syndrome_all)}')
            syndrome_stab_noiseless = syndrome_all[:self.num_ancillas]
            syndrome_stab_noisy = syndrome_all[self.num_ancillas:2*self.num_ancillas]
            syndrome = (GF2(syndrome_stab_noisy) + GF2(syndrome_stab_noiseless)).tolist()

            # print(f'The stabilziers syndrome is {syndrome_all}')
    
            # x_stab_syndrome = syndrome[:len(self.x_stab_rows)]
            # z_stab_syndrome = syndrome[len(self.x_stab_rows):]
            
            # decompose errors into X and Z errors

            # x_decomposition = (GF2(x_errors) @ self.X_error_basis_inverse).tolist() # maybe fouble check if the slicing is correct
            # z_decomposition = (GF2(z_errors) @ self.Z_error_basis_inverse).tolist()

            x_decomposition = (self.X_error_basis @ GF2(x_errors) ).tolist() # maybe fouble check if the slicing is correct
            z_decomposition = (self.Z_error_basis @ GF2(z_errors) ).tolist()

            x_decomposition = x_decomposition[len(self.x_stab_rows):]
            z_decomposition = z_decomposition[len(self.z_stab_rows):]
            # print(f'The X decomposition is {x_decomposition}')
            # print(f'The Z decomposition is {z_decomposition}')

            x_error_dict, z_error_dict = self._update_error_table(x_dicts, z_dicts, syndrome, x_decomposition, z_decomposition)
        mle_x_error_dict, mle_x_conf_dict = self._get_MLD_solutions(x_error_dict)
        mle_z_error_dict, mle_z_conf_dict = self._get_MLD_solutions(z_error_dict)

        return mle_x_error_dict, mle_z_error_dict, mle_x_conf_dict, mle_z_conf_dict
    
         

    # Stabilizer measurement circuit for both X and Z
    def full_stabilizer_sequence(self,
                                 circuit: stim.Circuit,
                                 noise_probability: float,
                                 measure: int =True,
                                 noise: int =False):

        # Just add noise to the beginning of the circuit for now
       
        if noise:
            self._add_depolarizing_noise(circuit, list(range(self.num_qubits)), noise_probability)   # adding the depolarization noise on the data qubits only    
            
        self._generate_stabilizer_circuits(circuit, mode='Z')
        self._generate_stabilizer_circuits(circuit, mode='X')

        if measure: 
            ancillas = self.X_anc + self.Z_anc
            measure_qubits(circuit ,ancillas)


    # Given the x_error_dict and z_error_dict, compute the overall error_dict, including the Y noise. 
    def get_MLE_error_table(self,
                            x_error_dict: Dict[str, str],
                            z_error_dict: Dict[str, str],
                            ):
        error_dict = {}
        for syndrome_x in x_error_dict.keys():
            x_error = x_error_dict[syndrome_x]
            for syndrome_z in z_error_dict.keys():
                z_error = z_error_dict[syndrome_z]
                syndrome = syndrome_x + syndrome_z
                error = self._get_pauli_error(x_error, z_error)
                error_dict[syndrome] = error

        return error_dict

    def _create_decoding_tables(self,
                           ) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, int]]]:
  
        num_x_errors = self.num_qubits - len(self.x_stab_rows)
        num_z_errors = self.num_qubits - len(self.z_stab_rows)

        # Here X_dict stores the measurement syndrome from Z stabilizers, and the corresponding X error data
        # Similarly for X but flipped
        x_dict = {}
        z_dict = {}
        num_stabilizers = len(self.x_stab_rows) + len(self.z_stab_rows) 

        for syndrome in generate_binary_strings(2**num_stabilizers):
            z_dict[syndrome] = {}

            for error in generate_binary_strings(2**num_z_errors): # this is the number of possible errors, including the logical errors 
                z_dict[syndrome][error] = 0
                

        for syndrome in generate_binary_strings(2**num_stabilizers):
            x_dict[syndrome] = {}

            for error in generate_binary_strings(2**num_x_errors):
                x_dict[syndrome][error] = 0


        return x_dict, z_dict
    

    # Updates the above error table
    def _update_error_table(self,
                           x_dict: Dict[str, Dict[str, int]],
                           z_dict: Dict[str, Dict[str, int]],
                           syndrome: List[int],
                           x_error: List[int],
                           z_error: List[int]) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, int]]]:
        
        syndrome = list_to_string(syndrome)
        

        x_error = list_to_string(x_error)
        z_error = list_to_string(z_error)
        # print(f'x_error: {x_error}')
        # print(f'all kinds of error types: {x_dict[syndrome].keys()}')
        if x_error not in x_dict[syndrome].keys():
            # print(f'currently x errors is {x_dict[syndrome].keys()}, and the corresponding syndrome is {syndrome}')
            # print(f'x_error is {x_error}')
            x_dict[syndrome][x_error] = 1
        else:
            # print(f'currently x errors is {x_dict[syndrome].keys()}, and the corresponding syndrome is {syndrome}')
            # print(f'x_error is {x_error}')
            x_dict[syndrome][x_error] += 1
        
        if z_error not in z_dict[syndrome].keys():
            z_dict[syndrome][z_error] = 1
        else:
            z_dict[syndrome][z_error] += 1

        return x_dict, z_dict
    
    # choose and return the most likely error # todo: modify this to create full syndrome error table
    def _get_MLD_solutions(self,
                          dictionary: Dict[str, Dict[str, int]],
                        ) -> Tuple[Dict[str, str], Dict[str, Tuple[float, int]]]:
        
        new_dictionary = {}
        confidence_dict = {}
        
        for syndrome, error_dict in dictionary.items():
           
            new_dictionary[syndrome] = max(error_dict, key = error_dict.get)
            total_samples = sum(dictionary[syndrome].values())
            all_values = list(dictionary[syndrome].values())
            all_values.sort(reverse=True)

           
            if total_samples == 0:
                confidence_dict[syndrome] = 0 # if no samples, confidence is 0
            elif total_samples == 1:
                confidence_dict[syndrome] = 1
            else:
                gap = (all_values[0]- all_values[1])   
                confidence_dict[syndrome] = gap / total_samples

        
        return new_dictionary, confidence_dict



    # # Initiate a table keeping track of the remaining logical errors after applying corrections
    # def _initiate_logical_error_table(self):
    #     num_log_ops = len(self.X_logical)
    #     dictionary = {}
    #     paulis = ['I', 'X', 'Y', 'Z']
    #     all_errors = itertools.product(paulis, repeat=num_log_ops)
    #     all_errors =[''.join(i) for i in all_errors]

    #     for error in all_errors:
    #         dictionary[error] = 0

    #     return dictionary


    def _get_pauli_error(self,
                        x_error: List,
                        z_error: List):
        # Input is x, z error in binary form
        # Returns product pauli
        num_log_ops = len(x_error)
        final_error = ['I' for _ in range(num_log_ops)]

        for i in range(num_log_ops):
            x = x_error[i]
            z = z_error[i]

            if x == '1' and z=='1':
                final_error[i] = 'Y'

            elif x=='1' and z=='0':
                final_error[i] = 'X'
                
            elif x=='0' and z=='1':
                final_error[i] = 'Z'

        return ''.join(final_error)

    
    def _add_depolarizing_noise(self,
                                circuit: stim.Circuit,
                                qubits: List,
                                probability: float,
                                ):
        for qubit in qubits: 
            # Append a single-qubit depolarizing channel on the given qubit.
            # circuit.append("DEPOLARIZE1", [qubit], probability)
            random_number = random.uniform(0,1)
     
            if random_number < probability/4:
                #print('X', qubit)
                circuit.append_operation("X", [qubit])
                
            elif probability/4 <= random_number < probability/2:
                #print('Y', qubit)
                circuit.append_operation("Y", [qubit])

            elif probability/2 <= random_number < 3*probability/4:
                #print('Z', qubit)
                circuit.append_operation("Z", [qubit])
    
    def _generate_stabilizer_circuits(self,
                                     circuit: stim.Circuit,
                                     mode: str='Z',
                                     noise: bool=False):   # currently does not add noise
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
        # print(f'printing circuit generating stabilizer: {circuit} wit mode {mode}')

    
    

        
if __name__ == "__main__":
    system_size = 7
    X_stabilizers = [[0, 1, 2, 3], [0, 1, 4, 5], [0, 2, 4, 6]]
    Z_stabilizers = [[0, 1, 2, 3], [0, 1, 4, 5], [0, 2, 4, 6]]

    Hx = get_parity_check_matrix(X_stabilizers, system_size)
    Hz = get_parity_check_matrix(Z_stabilizers, system_size)

    MLE = MLEDecoder(Hx, Hz)
    circuit = stim.Circuit()
    x_error_dict, z_error_dict, x_conf_dict, z_conf_dict = MLE.decode(circuit, 0.25, num_samples=5000)

    print(f'-------error dictionary----------')
    print(f'x_error_dict: {x_error_dict}')
    print(f'x_conf_dict: {x_conf_dict}')
    # print(f'--------------------------------')
    # print(f'z_error_dict: {z_error_dict}')
    # print(f'--------------------------------')

    # log_error_table = MLE.logical_channel(circuit, noise_probability=0.1, num_samples=1000, x_error_dict=x_error_dict, z_error_dict=z_error_dict)
    # print(f'logical error chanel', log_error_table)





