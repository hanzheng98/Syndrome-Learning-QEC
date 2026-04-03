import stim
import copy
import galois
import numpy as np
import random
import multiprocessing
import pickle 
import itertools
from typing import Any, Callable, Dict, List, Optional, Type, Union
import scipy  
from scipy import sparse
import os
import warnings

# utility functions for finding the logical operators given by the parity checks matrices

def indices_to_vector(indices, system_size):
    # Takes in an operator with the above description and returns a corresponding vector
    vec = [0] * system_size
    for i in indices:
        vec[i] = 1
    return vec

GF2 = galois.GF(2)

def get_parity_check_matrix(stabilizers, num_qubits):
    empty_stab = [0 for i in range(num_qubits)]
    empty_matrix = [copy.deepcopy(empty_stab) for j in range(len(stabilizers))]

    for i in range(len(stabilizers)):
        stab = stabilizers[i]
        for ind in stab:
            empty_matrix[i][ind] = 1
    return GF2(empty_matrix)



# Finds vectors that are in the kernel of one matrix and not in the span of another
# Helper for finding logical operators
def vecs_not_in_span(kernel, matrix):
    rank = np.linalg.matrix_rank(matrix)
    ops = []
    
    for vec in kernel:
        new_matrix = copy.deepcopy(matrix)
        new_matrix = np.vstack((new_matrix, vec))
        new_rank = np.linalg.matrix_rank(new_matrix)
        
        if new_rank == rank+1:
            ops.append(vec)
            matrix = new_matrix
            rank = new_rank
    return GF2(ops)


# finds logical operators given the parity check matrices
def find_logical_operators(x_matrix, z_matrix):
    ker_x = x_matrix.null_space()
    ker_z = z_matrix.null_space()

    Z_logical = vecs_not_in_span(ker_x, z_matrix)
    X_logical = vecs_not_in_span(ker_z, x_matrix)

    return X_logical, Z_logical

        

def measurement_to_vector(true_false_mat):
    # Takes in stim measurement result, maps true to 1 and false to 0
    return np.asarray(true_false_mat, dtype=bool).astype(int).tolist()
    # return [[1 if val else 0 for val in row] for row in true_false_matrix]



def unit_vectors_not_in_span(vectors: List[np.array]) -> Union[np.array, scipy.sparse.csr_matrix]:
    # Finds unit vectors that aren't in the span of given vectors to form a complete basis
    # Used for cosets of decoder
    # assume for binary operation 
    rank  = len(vectors)
    system_size = len(vectors[0])
    
    for i in range(system_size):
        unit_vector = GF2(indices_to_vector([i], system_size))

        new_matrix = GF2(copy.deepcopy(vectors))
        new_matrix = np.vstack((new_matrix, unit_vector))
     
        new_rank = np.linalg.matrix_rank(new_matrix)

        if new_rank > rank:
            vectors = new_matrix
            rank = new_rank

    return vectors



def find_LI_rows(matrix: Union[np.ndarray, sparse.csr_matrix]) -> List[List[int]]:
    """
    Given a binary matrix (over GF(2)) provided as a dense NumPy array or a sparse CSR matrix,
    this function returns a list of the original rows (each as a list of 0s and 1s) that are linearly 
    independent over GF(2). The number of returned rows equals the dimension of the row space.
    """
    # Convert sparse matrix to dense, if necessary, and ensure we work mod 2.
    if hasattr(matrix, "toarray"):
        A = matrix.toarray().copy() % 2
    else:
        A = np.array(matrix, dtype=int) % 2

    # Make a copy of the original matrix rows for later retrieval.
    original = A.copy()

    m, n = A.shape
    pivot_indices = []  # To store indices of pivot rows in A (and in original)
    row_idx = 0       # Current pivot row index

    # Process each column from 0 to n-1 using Gaussian elimination mod 2.
    for col in range(n):
        # Find a pivot row in the current column starting from row_idx.
        pivot = None
        for r in range(row_idx, m):
            if A[r, col] == 1:
                pivot = r
                break
        if pivot is None:
            continue

        # Swap pivot row with the row at row_idx if needed.
        if pivot != row_idx:
            A[[row_idx, pivot]] = A[[pivot, row_idx]]
            original[[row_idx, pivot]] = original[[pivot, row_idx]]
        
        # Record the pivot row index.
        pivot_indices.append(row_idx)
        
        # Eliminate all 1's in the current column from other rows.
        for r in range(m):
            if r != row_idx and A[r, col] == 1:
                A[r] = (A[r] + A[row_idx]) % 2

        row_idx += 1
        if row_idx >= m:
            break

    # Return the original rows corresponding to the pivot indices.
    return [original[i].tolist() for i in pivot_indices]


def pauli_to_symplectic(pauli_str: str) -> tuple:
    """
    Convert a Pauli operator string (e.g., "XYZXII") into its symplectic representation.
    
    Each Pauli is represented as a tuple (s_x, s_z), where s_x and s_z are binary strings.
    
    Conventions:
        I -> (0, 0)
        X -> (1, 0)
        Y -> (1, 1)
        Z -> (0, 1)
    
    Args:
        pauli_str (str): A string representing an n-qubit Pauli operator.
        
    Returns:
        tuple[str, str]: A tuple (s_x, s_z) of binary strings of length n.
    """
    mapping = {
        'I': (0, 0),
        'X': (1, 0),
        'Y': (1, 1),
        'Z': (0, 1)
    }
    
    s_x_bits = []
    s_z_bits = []
    
    # Process each character in the input string.
    for ch in pauli_str.upper():
        if ch not in mapping:
            raise ValueError(f"Invalid Pauli operator: {ch}")
        sx_bit, sz_bit = mapping[ch]
        s_x_bits.append(str(sx_bit))
        s_z_bits.append(str(sz_bit))
    
    # Concatenate bits into strings
    s_x = ''.join(s_x_bits)
    s_z = ''.join(s_z_bits)
    
    return (s_x, s_z)

def symplectic_to_pauli(symplectic: tuple) -> str:
    """
    Convert a Pauli operator from its symplectic representation back to a standard Pauli string.
    
    The symplectic representation is a tuple (s_x, s_z) of two binary strings of equal length.
    Each bit position corresponds to one qubit:
      - (0, 0) -> 'I'
      - (1, 0) -> 'X'
      - (1, 1) -> 'Y'
      - (0, 1) -> 'Z'
    
    Args:
        symplectic (tuple[str, str]): A tuple (s_x, s_z) where both s_x and s_z are binary strings.
    
    Returns:
        str: The corresponding Pauli operator string.
    """
    s_x, s_z = symplectic
    if len(s_x) != len(s_z):
        raise ValueError("s_x and s_z must have the same length")
    
    mapping = {
        ('0', '0'): 'I',
        ('1', '0'): 'X',
        ('1', '1'): 'Y',
        ('0', '1'): 'Z'
    }
    
    # Construct the Pauli string by processing each bit position.
    pauli_str = ''.join(mapping[(s_x[i], s_z[i])] for i in range(len(s_x)))
    return pauli_str

# -----------------------------
# stabilizer preparation circuits 
# -----------------------------

# Reverses a clifford circuit
# this is used to extract the logical errors 
def reverse_circuit(circ_obj):
    reversed_circuit = stim.Circuit()


    for instruction in reversed(circ_obj):
        reversed_circuit.append_operation(instruction)

    return reversed_circuit


# appends circ2 to the end of circ1, in a new circuit object
def add_circuits(circ1, circ2):
    circ3 = copy.deepcopy(circ1)
    for instruction in circ2:
        circ3.append(instruction)

    return circ3




def measure_qubits(circ_obj, qubits):
    for qubit in qubits:
        circ_obj.append('M', qubit)

def reset_ancillas(circ_obj, ancillas):
    for anc in ancillas:
        circ_obj.append("R", [anc])




def list_to_string(lst):
    string = ''.join([str(i) for i in lst])
    return string

def str_to_vector(string):
    return GF2([int(i) for i in list(string)])


def generate_binary_strings(n):
    # Generate all binary string representations of numbers from 0 to n-1
    width = len(f"{n-1:b}")
    return [f"{i:b}".zfill(width) for i in range(n)]



# -----------------------------
# Extract physical errors 
# -----------------------------
def extract_physical_x_errors(circ_obj, data_qubits, shots: int = 1):
    circuit = copy.deepcopy(circ_obj)

    measure_qubits(circuit, data_qubits)
    physical_sampler = circuit.compile_sampler()
    samples = measurement_to_vector(physical_sampler.sample(shots))
    return samples if shots != 1 else samples[0]


def extract_physical_z_errors(circ_obj, data_qubits, shots: int = 1):
    circuit = copy.deepcopy(circ_obj)

    new_circ = stim.Circuit()
    for qubit in data_qubits:
        new_circ.append('H', [qubit])

    circuit = add_circuits(new_circ, circuit)

    for qubit in data_qubits:
        circuit.append('H', [qubit])

    measure_qubits(circuit, data_qubits)
    physical_sampler = circuit.compile_sampler()
    samples = measurement_to_vector(physical_sampler.sample(shots))
    return samples if shots != 1 else samples[0]

def extract_y_meas(circ_obj, data_qubits, shots: int = 1):
    circuit = copy.deepcopy(circ_obj)

    new_circ = stim.Circuit()
    for qubit in data_qubits:
        new_circ.append('H', [qubit])
        new_circ.append('S', [qubit])

    circuit = add_circuits(new_circ, circuit)

    for qubit in data_qubits:
        circuit.append('S_DAG', [qubit])
        circuit.append('H', [qubit])

    measure_qubits(circuit, data_qubits)
    physical_sampler = circuit.compile_sampler()
    samples = measurement_to_vector(physical_sampler.sample(shots))
    return samples if shots != 1 else samples[0]

def extract_physical_errors(circ_obj, data_qubits, shots: int = 1):
    return (
        extract_physical_x_errors(circ_obj, data_qubits, shots=shots),
        extract_physical_z_errors(circ_obj, data_qubits, shots=shots),
        extract_y_meas(circ_obj, data_qubits, shots=shots),
    )


def check_y_errors(x_errors, z_errors, y_meas):
    """
    For each index i, if both x_errors[i] and z_errors[i] are 1, then the
    y measurement (y_meas[i]) should be 0, and if either x_errors[i] or z_errors[i]
    is 1 (but not both) then y_meas[i] should be 1. If the condition is violated,
    a warning is issued.
    
    Parameters:
        x_errors (list or array-like): Binary array indicating X errors.
        z_errors (list or array-like): Binary array indicating Z errors.
        y_meas (list or array-like): Binary array for Y measurements.
    """
    n = len(x_errors)
    if len(z_errors) != n or len(y_meas) != n:
        raise ValueError("All input arrays must have the same length.")
    
    for i in range(n):
        if x_errors[i] == 1 and z_errors[i] == 1:
            if y_meas[i] != 0:
                # x_errors[i] = 0
                # z_errors[i] = 0
                # print(f"x_errors: {x_errors}")
                # print(f"z_errors: {z_errors}")
                # print(f"y_meas: {y_meas}")
                warnings.warn(f"There should be a Y error at index {i}", UserWarning)
    return x_errors, z_errors

# -----------------------------
# Write the data to a file and extract it 
# -----------------------------

def write_pauli_data(filename, data_dict):
    """
    Writes a dictionary to a text file where each line contains:
      PauliOperator: Value

    The function attempts to create the file in "../data/decoders" using the provided filename.
    If writing the file there fails, it falls back to creating a new file with the given filename
    in the current working directory.

    If a file with the provided filename already exists in the target location, a FileExistsError is raised.

    Parameters:
        filename (str): The filename to use (should be a simple filename, not a full path).
        data_dict (dict): A dictionary with Pauli operator strings as keys and floats as values.
    """
    # Define default directory and path
    default_dir = os.path.join("..", "data", "decoders")
    default_path = os.path.join(default_dir, filename)
    
    try:
        # Ensure the default directory exists
        os.makedirs(default_dir, exist_ok=True)
        # Check if file already exists in the default location
        if os.path.exists(default_path):
            raise FileExistsError(f"File already exists: {default_path}")
        # Attempt to write the file at the default location
        with open(default_path, 'w') as f:
            for pauli_operator, value in data_dict.items():
                f.write(f"{pauli_operator}: {value}\n")
        print(f"Data successfully written to {default_path}")
    except OSError as e:
        print(f"Could not write to {default_path} due to: {e}. Falling back to current directory.")
        fallback_path = filename
        # Check if file already exists in the fallback location (current directory)
        if os.path.exists(fallback_path):
            raise FileExistsError(f"File already exists: {fallback_path}")
        # Write the file in the current directory
        with open(fallback_path, 'w') as f:
            for pauli_operator, value in data_dict.items():
                f.write(f"{pauli_operator}: {value}\n")
        print(f"Data successfully written to {os.path.abspath(fallback_path)}")


def read_pauli_data(filename):
    # Thanks ChatGPT! 
    """
    Reads a text file and returns a dictionary where:
      - The key is the Pauli operator string (e.g., 'IXIIIX')
      - The value is a numeric value (float) on the same line, after a colon.

    Each line in the file should be of the form:
        PauliOperator: Value
    For example:
        IXIIIX: 0.5
        XIXXXI: 0.123
    
    Parameters:
        filename (str): The path to the .txt file to read.

    Returns:
        dict: A dictionary with the Pauli operator string as keys and floats as values.
    """
    data_dict = {}
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty or whitespace-only lines
            if not line:
                continue
            
            # Split the line on the first occurrence of ':'.
            # This way, even if there's other whitespace, we safely get Pauli and the numeric part.
            pauli_str, val_str = line.split(':', 1)
            
            # Remove extra whitespace
            pauli_operator = pauli_str.strip()
            value = float(val_str.strip())
            
            # Store in the dictionary
            data_dict[pauli_operator] = value

    return data_dict

# -----------------------------
# TEST CASES
# -----------------------------
# print("Test Case 1:")
# # Example 1: Start with a single vector in GF(2)^3.
# vectors1 = [np.array([1, 0, 0])]
# print("Input vectors:")
# for v in vectors1:
#     print(v)
# basis1 = unit_vectors_not_in_span(vectors1)
# print("Complete basis (columns):")
# print(basis1)
# print("Shape:", basis1.shape)  # Should be (3,3)

# print("\nTest Case 2:")
# # Example 2: Two independent vectors in GF(2)^3.
# vectors2 = [np.array([1, 1, 0]), np.array([0, 1, 1])]
# print("Input vectors:")
# for v in vectors2:
#     print(v)
# basis2 = unit_vectors_not_in_span(vectors2)
# print("Complete basis (columns):")
# print(basis2)
# print("Shape:", basis2.shape)  # Should be (3,3)

# print("\nTest Case 3:")
# # Example 3: Two independent vectors in GF(2)^4.
# vectors3 = [np.array([1, 0, 1, 0]), np.array([0, 1, 1, 0])]
# print("Input vectors:")
# for v in vectors3:
#     print(v)
# basis3 = unit_vectors_not_in_span(vectors3)
# print("Complete basis (columns):")
# print(basis3)
# print("Shape:", basis3.shape)  # Should be (4,4)

# print("n = 4:", generate_binary_strings(3))
