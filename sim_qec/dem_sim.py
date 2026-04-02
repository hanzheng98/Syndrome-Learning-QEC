import stim 
import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
import copy
import random
import time
from bposd.css import css_code
from scipy import sparse
from typing import Dict, List, Optional, Union, Tuple
from sim_qec.circuit_scheduling import ColorationCircuit

'''
This is module utilizing the stim detector error model to simulate the logical error rates and get the syndrome expectation values
'''







class BaseDEMSim:

    '''
    
    The base class for DEM simulation of quantum error correcting codes. 


    '''

    def __init__(self,
                 code: css_code,
                 num_cycles: int,
                 circuit_error_params: dict,
                 physical_error_rate: Optional[float]=None,
                 ):
        self.eval_code = code
        self.num_cycles = num_cycles
        if physical_error_rate is not None:
            circuit_error_params = copy.deepcopy(circuit_error_params)
            for key, value in circuit_error_params.items():
                circuit_error_params[key] = value * physical_error_rate
            
        self.circuit_error_params = circuit_error_params
        self.data_indices = list(range(0, code.hx.shape[1]))
        self.n = len(self.data_indices)
        self.num_cycles = num_cycles
        n_Z_ancilla, n_X_ancilla = code.hz.shape[0], code.hx.shape[0]
        self.Z_ancilla_indices = list(np.arange(self.n, self.n + n_Z_ancilla))
        self.X_ancilla_indices = list(np.arange(self.n + n_Z_ancilla, self.n + n_Z_ancilla + n_X_ancilla))

    def _initialize_circuit(self,
                      circuit: stim.Circuit,
                      reset: Union[str, None], # we default to prepare in Z basis 
                      do_syndrome: bool=False, # if we do the syndrome extraction, then we do not need to reset the data qubits
                      mode: str='repetition', # another option would be 'CSS'
                      ) -> stim.Circuit:
        '''
        The initialization round of the syndrome extraction circuit.
        '''
        data_indices = self.data_indices
        Z_ancilla_indices = self.Z_ancilla_indices
        X_ancilla_indices = self.X_ancilla_indices

        if reset is not None:
            circuit.append(reset, data_indices)
            if mode == 'CSS':
                circuit.append("R", X_ancilla_indices + Z_ancilla_indices) # we always init the ancialla in Z basis
            elif mode == 'repetition':
                circuit.append("R", Z_ancilla_indices) # we always init the ancialla in Z basis
        else:
            if do_syndrome is False:
                raise ValueError("If no reset is specified, do_syndrome must be True")
            else: 
                self._ideal_sec_round(circuit, data_indices, pairwise_diff=False)

        return circuit
    
    def _ideal_sec_round(self,
                        circuit: stim.Circuit,
                        data_indices: List[int],
                        pairwise_diff: bool=True,
                        ) -> stim.Circuit:
        '''
        The base stabilizer circuit: the syndrome measurements are peroformed
        without scheduling and assumed that it performed perfectly.
        '''
        hx, hz = self.eval_code.hx, self.eval_code.hz
        num_x_checks = hx.shape[0]
        num_z_checks = hz.shape[0]

        #circuit.append("Tick") # It is not strictly needed
        
        # Terminal perfect Z measuerements
         
        z_terms = []
        for r in range(num_z_checks):
            cols = np.flatnonzero(hz[r])
            # Build Zq1*Zq2*... as a single MPP target
            prod = None
            for c in cols:
                t = stim.target_z(int(data_indices[c]))
                prod = t if prod is None else (prod * t)
            z_terms.append(prod)
        if len(z_terms) > 0:
            circuit.append("MPP", z_terms)

        # Terminal perfect X measurements
        x_terms = []
        for r in range(num_x_checks):
            cols = np.flatnonzero(hx[r])
            prod = None
            for c in cols:
                t = stim.target_x(int(data_indices[c]))
                prod = t if prod is None else (prod * t)
            x_terms.append(prod)
        if len(x_terms) > 0:
            circuit.append("MPP", x_terms)
        
        # ---- Detectors tying the last noisy round to the perfect terminal round ----
        # Let the last ancilla MR block have size (nZ + nX) and come immediately before these MPPs.
        # After both MPP calls, the measurement record gained +nZ (Z) +nX (X) results, in that order.

        if pairwise_diff:

            # Parity Z (terminal perfect) ⊕ Z (previous ancilla round)
            for j in range(num_z_checks):
                circuit.append(
                    "DETECTOR",
                    [
                        stim.target_rec(-(num_x_checks + num_z_checks) + j),          # current terminal Z_j (in the MPP Z block)
                        stim.target_rec(-(2*(num_x_checks + num_z_checks)) + j),      # previous round's Z-ancilla meas j
                    ],
                    0
                )

            # Parity X (terminal perfect) ⊕ X (previous ancilla round)
            for j in range(num_x_checks):
                circuit.append(
                    "DETECTOR",
                    [
                        stim.target_rec(-num_x_checks + j),                 # current terminal X_j (in the MPP X block)
                        stim.target_rec(-(2*(num_x_checks + num_z_checks)) + num_z_checks + j), # previous round's X-ancilla meas j
                    ],
                    0
                )
        else:
            # Add boundary detectors that “anchor” the terminal perfect MPP results themselves.
            # Z-stabilizers (one DETECTOR per Z check):
            for j in range(num_z_checks):
                circuit.append("DETECTOR", [stim.target_rec(-(num_x_checks + num_z_checks) + j)])
            # X-stabilizers (one DETECTOR per X check):
            for j in range(num_x_checks):
                circuit.append("DETECTOR", [stim.target_rec(-num_x_checks + j)])
            circuit.append("TICK")
        return circuit
    

    def _transversal_measurement(self,
                                 circuit: stim.Circuit, 
                                 basis: str='Z',
                                 add_faults: bool=False,
                                 pairwise_diff: bool=True,
                                 ) -> stim.Circuit:
        '''
        final transversal measurement of the data qubits in a given basis

        '''
        # if it is given as sparse matrix, convert to dense array
        # hx = self.eval_code.hx.toarray()
        # hz = self.eval_code.hz.toarray()
        # lz = self.eval_code.lz.toarray()
        # lx = self.eval_code.lx.toarray()

        hx = self.eval_code.hx
        hz = self.eval_code.hz
        lz = self.eval_code.lz
        lx = self.eval_code.lx



        data_indices = list(range(hx.shape[1])) #must be all data qubits 

        if basis == 'Z':
            if add_faults:
                self._add_fault_circuit(circuit, data_indices, fault_location='p_m', fault_type='X_ERROR')
            circuit.append("M", data_indices, float(self.circuit_error_params['p_m']))

            circuit.append("SHIFT_COORDS", [], (1)) #Letting the detector know that this is a new time frame 
            for i in range(hz.shape[0]):
                # print(f'Z parity check: {hz}')
                supported_data_indices = list(np.where(hz[i,:] == 1)[0])
                rec_indices = []
                for j in supported_data_indices:
                    rec_indices.append(- len(data_indices) + j)
                
                if pairwise_diff:
                    rec_indices.append(- hz.shape[0] + i - len(data_indices)) # get the syndrome from last round

                circuit.append("Detector", [stim.target_rec(rec_index) for rec_index in rec_indices], (0))

            # now we proceed with performing logical measurements
            for i in range(lz.shape[0]):
                supported_data_indices = list(np.where(lz[i,:] == 1)[0])
                rec_indices = []
                for j in supported_data_indices:
                    rec_indices.append(- len(data_indices) + j)
                
                circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(rec_index) for rec_index in rec_indices], (i))
        
        elif basis == 'X':
            if add_faults:
                self._add_fault_circuit(circuit, data_indices, fault_location='p_m', fault_type='DEPOLARIZE1')
            circuit.append("MX", data_indices, float(self.circuit_error_params['p_m']))
            circuit.append("SHIFT_COORDS", [], (1)) #Letting the detector know that this is a new time frame 
            for i in range(hx.shape[0]):
                # print(f'Z parity check: {hz}')
                supported_data_indices = list(np.where(hx[i,:] == 1)[0])
                rec_indices = []
                for j in supported_data_indices:
                    rec_indices.append(- len(data_indices) + j)
                
                if pairwise_diff:
                    rec_indices.append(- hx.shape[0] + i - len(data_indices)) # get the syndrome from last round

                circuit.append("Detector", [stim.target_rec(rec_index) for rec_index in rec_indices], (0))

            # now we proceed with performing logical measurements
            for i in range(lx.shape[0]):
                supported_data_indices = list(np.where(lx[i,:] == 1)[0])
                rec_indices = []
                for j in supported_data_indices:
                    rec_indices.append(- len(data_indices) + j)
                
                circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(rec_index) for rec_index in rec_indices], (i))
                

    #ToDO: modulize the noise process from noise_model.py for better flexibility and added 2-qubit noise as well. 
    def _add_fault_circuit(self,
                           circuit: stim.Circuit,
                           indices: List[int],
                           fault_location: str,
                           fault_type: Optional[str]="X_ERROR", # "X_ERROR" or"DEPOLARIZE1"
                           ) -> stim.Circuit:
        '''

         Add the circuit faults based on the circuit error parameters
         indices are given either as data qubits or ancilla qubits
         fault_location is given as the key of the self.circuit_error_params, indicating where to add the fault
         fault_type is either 'X' or depoalrization. No other types are supported currently
        
         ToDo: support more fault types including the correlated errors
        '''
        # add random number to randomize the errors 
        circuit.append(fault_type, indices, float(self.circuit_error_params[fault_location]) )




class DEMSyndromeExtraction(BaseDEMSim):
    '''

    A class to perform syndrome extraction using stim detector error model, which is the memory experiment circuit

    '''

    def __init__(self,
                 code: css_code,
                 num_cycles: int,
                 circuit_error_params: dict,
                 physical_error_rate: Optional[float]=None,
                 ):
        super().__init__(code, num_cycles, circuit_error_params, physical_error_rate)

        # add attributes to the data and ancilla indices
        
        scheduling_X = ColorationCircuit(code.hx)
        scheduling_Z = ColorationCircuit(code.hz)
        print("Syndrome extraction circuit scheduling for X stabilizers:", scheduling_X)
        print("Syndrome extraction circuit scheduling for Z stabilizers:", scheduling_Z)
        self.scheduling = {
            "X": scheduling_X,
            "Z": scheduling_Z
        }


    def build_circuit(self,
                       fault_type: str= 'DEPOLARIZE1') -> stim.Circuit:
        
        '''
        Build the stim circuit for syndrome extraction.

        Returns:
            stim.Circuit: The constructed stim circuit.

            fault_type: str='DEPOLARIZE1' , assumed to be depolarization error 


        '''



      

        # Set the noise model
        # error_params = {"p_i": circuit_error_params['p_i']*p, "p_state_p": circuit_error_params['p_state_p']*p, 
        # "p_m": circuit_error_params['p_m']*p, "p_CX":circuit_error_params['p_CX']*p, "p_idling_gate": circuit_error_params['p_idling_gate']*p}
        # error_params = self.circuit_error_params
        # Set the key params 
        # hx = self.hx
        # hz = self.hz
        # hx = eval_code.hx
        # hz = eval_code.hz
        # lx = eval_code.lx
        
        # set the params for indices 
        data_indices = self.data_indices
        # n = self.n
        Z_ancilla_indices = self.Z_ancilla_indices
        X_ancilla_indices = self.X_ancilla_indices


        ## Initialization layer
        circuit_init = stim.Circuit()
        self._initialize_circuit(circuit_init, reset="RX") # initialize in X basis for now

        # add the state preparation errors, which only on the data qubits
        self._add_fault_circuit(circuit_init, data_indices, fault_location='p_state_p', fault_type=fault_type)


        circuit_rep1 = stim.Circuit()
        # we now perform the first round of stabilizer measurements
        self._noisy_sec_round(circuit_rep1, data_indices, fault_type=fault_type, mode='CSS')
        

        # # add the measurement noise 
        # self._add_fault_circuit(circuit_rep1, Z_ancilla_indices, fault_location='p_m', fault_type=fault_type)

        circuit_rep1.append("H", X_ancilla_indices)
        # add the measurement noise after the Hadamard gates
        self._add_fault_circuit(circuit_rep1, X_ancilla_indices, fault_location='p_m', fault_type=fault_type)
        

        circuit_rep1.append("MR", Z_ancilla_indices +  X_ancilla_indices) # the order matters for the detectors later
        circuit_rep1.append("SHIFT_COORDS", [], (1))

        for i in range(len(X_ancilla_indices)):
            circuit_rep1.append("DETECTOR", [stim.target_rec(- len(X_ancilla_indices) + i)], (0))
        circuit_rep1.append("TICK")

        # we now build the subsequent repeated rounds
        circuit_rep2 = stim.Circuit()
        self._noisy_sec_round(circuit_rep2, data_indices, fault_type=fault_type, mode='CSS')

        circuit_rep2.append("H", X_ancilla_indices)
        self._add_fault_circuit(circuit_rep2, X_ancilla_indices, fault_location='p_m', fault_type=fault_type)

        circuit_rep2.append("MR", Z_ancilla_indices +  X_ancilla_indices) # the order matters for the detectors later
        circuit_rep2.append("SHIFT_COORDS", [], (1))

        for i in range(len(X_ancilla_indices)):
            circuit_rep2.append("DETECTOR", [stim.target_rec(- len(X_ancilla_indices) + i), 
                                        stim.target_rec(- len(X_ancilla_indices) + i - len(Z_ancilla_indices) - len(X_ancilla_indices))], (0))
        circuit_rep2.append("TICK")


        # We now add the final layer of perfect measurements (noise strictly act on the terminal time but not on the measurements)

        final_circuit = stim.Circuit()
        self._transversal_measurement(final_circuit, basis='X', add_faults=True, pairwise_diff=True)


        circuit_syn_meas = circuit_init + circuit_rep1 + (self.num_cycles-1) * circuit_rep2 + final_circuit

        return circuit_syn_meas


    

       
       




        

    def build_repetition_circuit(self,
                                 fault_type: str= 'X_ERROR',
                                 pairwise_diff: bool = True) -> stim.Circuit:
        '''
        We first build the repetition code circuit for syndrome extraction without worrying about the X stabilizers

        This is used as demo for learnability and unlearnability 

        pairwise_diff: bool = True indicates whether we do the pairwise difference between the last round and the terminal perfect measurement round
        '''

      
        
        data_indices = self.data_indices
        Z_ancilla_indices = self.Z_ancilla_indices

        ## Initialization layer
        circuit_init = stim.Circuit()
        self._initialize_circuit(circuit_init, reset="R", mode='repetition') # initialize in Z basis for now
        self._add_fault_circuit(circuit_init, data_indices, fault_location='p_state_p', fault_type=fault_type)
       
        # we now perform the first round of stabilizer measurements
        circuit_rep1 = stim.Circuit()
        
        self._noisy_sec_round(circuit_rep1, data_indices, fault_type=fault_type, mode='repetition')

        # add the measurement noise
        # self._add_fault_circuit(circuit_rep1, Z_ancilla_indices, fault_location='p_m', fault_type=fault_type)
        # circuit_stab_meas_rep2.append("DEPOLARIZE1", data_indices, (error_params['p_i'])) # Add the idling errors on the data qubits during the measurement of X ancillas
        circuit_rep1.append("MR", Z_ancilla_indices)
        circuit_rep1.append("SHIFT_COORDS", [], (1))
        for i in range(len(Z_ancilla_indices)):
            circuit_rep1.append("DETECTOR", [stim.target_rec(- len(Z_ancilla_indices) + i)], (0))
        circuit_rep1.append("TICK")

        circuit_rep2 = stim.Circuit()
        self._noisy_sec_round(circuit_rep2, data_indices, fault_type=fault_type, mode='repetition')

        circuit_rep2.append("MR", Z_ancilla_indices)
        circuit_rep2.append("SHIFT_COORDS", [], (1)) # we shift to another time frame
        # add the measurement noise
        # self._add_fault_circuit(circuit_rep2, Z_ancilla_indices, fault_location='p_m', fault_type=fault_type)
        
        for i in range(len(Z_ancilla_indices)):
            circuit_rep2.append("DETECTOR", [stim.target_rec(- len(Z_ancilla_indices) + i), 
                                        stim.target_rec(- 2 * len(Z_ancilla_indices) + i)], (0))
        circuit_rep2.append("TICK")

        # Now perform the terminal perfect measurements

        final_circuit = stim.Circuit()
        self._transversal_measurement(final_circuit, basis='Z', add_faults=True, pairwise_diff=pairwise_diff) # assuming the last 
        # self._noisy_sec_round(final_circuit, data_indices, fault_type=fault_type, mode='repetition')

        # add the measurement noise
        # self._add_fault_circuit(circuit_rep1, Z_ancilla_indices, fault_location='p_m', fault_type=fault_type)
        # circuit_stab_meas_rep2.append("DEPOLARIZE1", data_indices, (error_params['p_i'])) # Add the idling errors on the data qubits during the measurement of X ancillas
        # final_circuit.append("M", Z_ancilla_indices)


        # for i in range(len(Z_ancilla_indices)):
        #     final_circuit.append("DETECTOR", [stim.target_rec(- len(Z_ancilla_indices) + i), 
        #                                 stim.target_rec(- 2 * len(Z_ancilla_indices) + i)], (0))
        # final_circuit.append("TICK")

       
         # combine the middle syndrome measurement circuit 

        circuit_syn_meas = circuit_init + circuit_rep1 + (self.num_cycles-1) * circuit_rep2 + final_circuit


        #implement the logical measurement here using the stim and detector error model


        return circuit_syn_meas
    

    def build_demo_rep_circuit(self,
                                 fault_type: str= 'X_ERROR', 
                                 type: str='midcircuit') -> stim.Circuit:
        '''
        
        A demo on the unlearnable and learnable logical noise from the syndrome noise 

        type : str = 'midcircuit' demonstration on how the improper mid-circuit syndrome measuremment can lead to unlearnable logical noise
               str = 'logical' demonstrate on how two learns can lead to unlearnable logical noise
        '''


        if type == 'midcircuit':
            # build the midcircuit demo; assume we have the measurement error for a single round of syndrome extraction, which is unlearnable
            data_indices = self.data_indices
            Z_ancilla_indices = self.Z_ancilla_indices

            circuit_init = stim.Circuit()
            self._initialize_circuit(circuit_init, reset="R", mode='repetition') # initialize in Z basis for now
            # we do not add the state preparation errors here for simplicity
            # self._add_fault_circuit(circuit_init, data_indices, fault_location='p_state_p', fault_type=fault_type)

            # We only do one round of noise syndrome extraction here 
            circuit_rep1 = stim.Circuit()
            self._noisy_sec_round(circuit_rep1, data_indices, fault_type=fault_type, mode='repetition')

            circuit_rep1.append("MR", Z_ancilla_indices)
            circuit_rep1.append("SHIFT_COORDS", [], (1))
            for i in range(len(Z_ancilla_indices)):
                circuit_rep1.append("DETECTOR", [stim.target_rec(- len(Z_ancilla_indices) + i)], (0))
            circuit_rep1.append("TICK")

            # we now do the final transversal measurement

            final_circuit = stim.Circuit()

            # now we proceed with performing logical measurements

            # should we only do the final logical measurement right? 
            lz = self.eval_code.lz
            for i in range(lz.shape[0]):
                supported_data_indices = list(np.where(lz[i,:] == 1)[0])
                rec_indices = []
                for j in supported_data_indices:
                    rec_indices.append(- len(data_indices) + j)

                final_circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(rec_index) for rec_index in rec_indices], (i))

            circuit_syn_meas = circuit_init + circuit_rep1 + final_circuit

            return circuit_syn_meas



        elif type == 'logical':
            # build the logical demo
            circuit = self.build_logical_demo(fault_type=fault_type)

        return circuit  

    def _noisy_sec_round(self,
                        circuit: stim.Circuit,
                        data_indices: List[int],
                        fault_type: str= 'X_ERROR',
                        mode: str='repetition',
                        ) -> stim.Circuit:
        
        '''
        The noisy stabilizer measurement round of the syndrome extraction circuit.
        '''

        Z_ancilla_indices = self.Z_ancilla_indices
        X_ancilla_indices = self.X_ancilla_indices


        if mode == 'repetition':
            # Now we measure the Z type stabilizers, every time step we perform a single Z stabilizer measurement
            hz = self.eval_code.hz
            for time_step in range(hz.shape[0]):
                # we first set the idling qubits to be everything then we gradually remove the indices where the gates are acted
                idling_qubits = data_indices + Z_ancilla_indices
                self._add_fault_circuit(circuit, idling_qubits, fault_location='p_i', fault_type=fault_type)

                for j in range(hz.shape[1]): # iterate through all the data qubits to implement the CX gates controled on the data qubits
                    circuit.append("CX", [j , Z_ancilla_indices[time_step]]) if hz[time_step, j] == 1 else None

                circuit.append("TICK")
            
            # we add the measurement noise only at the end of all the CX gates
            self._add_fault_circuit(circuit, Z_ancilla_indices, fault_location='p_m', fault_type=fault_type)
                    
        
        elif mode =='CSS':
            #First, we perform the X stabililizer measurements
            circuit.append("H", X_ancilla_indices)

            self._add_fault_circuit(circuit, X_ancilla_indices, fault_location='p_state_p', fault_type=fault_type)

            circuit.append("TICK")

            for time_step in range(len(self.scheduling['X'])):
                # add idling errors for all the qubits during the ancilla shuffling
                idling_qubits = data_indices + X_ancilla_indices
                idling_data_indices = list(copy.deepcopy(data_indices))
                self._add_fault_circuit(circuit, idling_qubits, fault_location='p_idling_gate', fault_type=fault_type)
                for j in self.scheduling['X'][time_step]:
            #                 supported_data_qubits = list(np.where(hx[X_ancilla_index - n - n_Z_ancilla,:] == 1)[0])
                    X_ancilla_index = X_ancilla_indices[j] #what the hell does this do; check
                    data_index = self.scheduling['X'][time_step][j]
                    # data_index = supported_data_qubits[i]
                    circuit.append("CX", [X_ancilla_index, data_index])
                    if data_index in idling_data_indices:
                        idling_data_indices.pop(idling_data_indices.index(data_index))
               
                self._add_fault_circuit(circuit, idling_data_indices, fault_location='p_i', fault_type=fault_type)
                circuit.append("TICK")
            
            # Now we measure the Z type stabilizers
            for time_step in range(len(self.scheduling['Z'])):
                # we first set the idling qubits to be everything then we gradually remove the indices where the gates are acted
                idling_qubits = data_indices + Z_ancilla_indices
                self._add_fault_circuit(circuit, idling_qubits, fault_location='p_idling_gate', fault_type=fault_type)
                idling_data_indices = list(copy.deepcopy(data_indices))

                # for each time step, we implement the CX gates according to the scheduling 
                for j in self.scheduling['Z'][time_step]:
                    Z_ancilla_index_sch = Z_ancilla_indices[j]
                    data_index_sch = self.scheduling['Z'][time_step][j]
                    circuit.append("CX", [data_index_sch, Z_ancilla_index_sch])
                    if data_index_sch in idling_data_indices:
                        idling_data_indices.pop(idling_data_indices.index(data_index_sch))
                self._add_fault_circuit(circuit, idling_data_indices, fault_location='p_i', fault_type=fault_type)
                circuit.append("TICK")


class DEMSyndromeExtractionNonCSS(BaseDEMSim):
    '''

    A class to perform syndrome extraction using stim detector error model for non-CSS codes, which is the memory experiment circuit

    '''

    def __init__(self,
                 code: css_code,
                 num_cycles: int,
                 circuit_error_params: dict,
                 physical_error_rate: Optional[float]=None,
                 ):
        super().__init__(code, num_cycles, circuit_error_params, physical_error_rate)

        # add attributes to the data and ancilla indices
        pass 


        


           



    


       



if __name__ == "__main__":

    # this is our errors
    '''
    p_i is the single qubit idling error prob 
    p_state_p is the state preparation error prob
    p_m is the measurement error prob
    p_CX is the two qubit gate error prob (setting to be 0.0)
    p_idling_gate is the idling error during the gate operation also two qubits (setting to be 0.0)
    '''
    CIRCUIT_ERROR_PARAMS = {"p_i": 0.01, "p_state_p": 0.01, "p_m": 0.01, "p_CX":0.0, "p_idling_gate": 0.0}

    # we now build a a three-qubit repetition code for demonstration

    '''
    The repetition code stabilizers are given as
    '''
    # Hz = np.array([[1,1,0],
    #                [0,1,1]])
    # Hx = np.array([[0,0,0]])

    '''
    The [[7, 1, 3]] Steane code stabilizers are given as
    
    '''
    # X_stabilizers = [[0, 1, 2, 3], [0, 1, 4, 5], [0, 2, 4, 6]]
    # Z_stabilizers = [[0, 1, 2, 3], [0, 1, 4, 5], [0, 2, 4, 6]]

    X_stabilizers = [[3,4,5,6],[1,2,5,6],[0,2,4,6]]
    Z_stabilizers = [[3,4,5,6],[1,2,5,6],[0,2,4,6]]

    from sim_qec.utils import get_parity_check_matrix 

    Hx = get_parity_check_matrix(X_stabilizers, 7)
    Hz = get_parity_check_matrix(Z_stabilizers, 7) 
    rep_code = css_code(Hx, Hz)
    # print("Code parameters:", rep_code.hz)
    # print('the X parity check', rep_code.hx)
    # print(f'the Lz logical operators: {rep_code.lz}')
    # print(f'the Lx logical operators: {rep_code.lx}')

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

    # Build the repetition-code circuit; fault_type "X_ERROR" injects X noise where configured
    # circ = dem.build_repetition_circuit(fault_type="X_ERROR")

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

 

