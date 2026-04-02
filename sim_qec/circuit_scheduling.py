import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
import copy
import random


def CoorToMat(coors, mat_size):
    l,m = list(mat_size.values())
    mat = np.zeros([l,m])
    for coor in coors:
        alpha, beta = coor
        mat[alpha, beta] = 1
    return mat

def CoorToPauli(coors, mat_size, sec):
    l,m = list(mat_size.values())
    vec = np.reshape(CoorToMat(coors, mat_size), (l,m))
    if sec == 'L':
        return np.hstack([vec, np.zeros(l*m)])
    elif sec == 'R':
        return np.hstack([np.zeros(l*m), vec])


def permute_columns(matrix, permutation):
    """
    Permutes the columns of a given matrix according to the permutation list.

    Parameters:
    matrix (2D array): The input matrix to permute.
    permutation (list): A list specifying the new order of columns.

    Returns:
    2D array: The matrix with permuted columns.
    """
    # Ensure permutation is valid
    if sorted(permutation) != list(range(matrix.shape[1])):
        raise ValueError("Invalid permutation: indices must match the number of columns.")
    
    # Permute the columns
    permuted_matrix = matrix[:, permutation]
    
    return permuted_matrix
    
def max_degree(graph):
    return max(list(dict(graph.degree).values()))

def BipartitieGraphFromCheckMat(H):
    num_checks, num_bits = H.shape
    C_nodes = list(-np.arange(1, num_checks + 1))
    V_nodes = list(np.arange(1, num_bits + 1))
    edges = [(-(i + 1), j + 1) for i in range(num_checks) for j in range(num_bits) if H[i, j] == 1]
    
    G = nx.Graph()
    G.add_nodes_from(C_nodes, bipartite=0)
    G.add_nodes_from(V_nodes, bipartite=1)
    G.add_edges_from(edges)
    
    return G

def best_match(graph):
    C_nodes = list({n for n, d in graph.nodes(data=True) if d["bipartite"] == 0})
    V_nodes = list(set(graph) - set(C_nodes))

    return bipartite.matching.hopcroft_karp_matching(graph, C_nodes)

# Coloration circuit
def TransformBipartiteGraph(G):
    # transform any bipartite graph to a symmetric one by adding dummy vertices and edges
    G_s = copy.deepcopy(G)
    C_nodes = list({n for n, d in G.nodes(data=True) if d["bipartite"] == 0})
    V_nodes = list(set(G) - set(C_nodes))
    
    # Suppose C_nodes all have degree Delta_c, and # V_nodes > # C_nodes
    # Add dummy vertices to C_nodes
    C_nodes_dummy = list(-np.arange((len(C_nodes) + 1), len(V_nodes) + 1))
    G_s.add_nodes_from(C_nodes_dummy, bipartite=0)
    
    # Add dummy edges between edges with degree < Delta_c
    Delta = max_degree(G_s)
#     print('max degree:', Delta)
    open_degree_nodes = copy.deepcopy(dict((node, degree) for node, degree in dict(G_s.degree()).items() if degree < Delta))
            
    while len(open_degree_nodes) > 0:       
        for node1 in list(open_degree_nodes.keys()):
            if node1 < 0:
                c_node = node1
                for node2 in list(open_degree_nodes.keys()):
                    if node2 > 0:
                        v_node = node2
                        if not G_s.has_edge(c_node, v_node):
                            G_s.add_edge(c_node, v_node)
                            
                            if open_degree_nodes[c_node] + 1 == Delta:
                                open_degree_nodes.pop(c_node)
                            else:
                                open_degree_nodes[c_node] = open_degree_nodes[c_node] + 1

                            if open_degree_nodes[v_node] + 1 == Delta:
                                open_degree_nodes.pop(v_node)
                            else:
                                open_degree_nodes[v_node] = open_degree_nodes[v_node] + 1
                            
                            break            
                        
            
    return G_s


def edge_corloring(graph):
    matches_list = []
    g = copy.deepcopy(graph)
    g_s = TransformBipartiteGraph(g)
    
    number_colors = max_degree(g_s)
    
    C_nodes = list({n for n, d in g.nodes(data=True) if d["bipartite"] == 0})
    V_nodes = list(set(g) - set(C_nodes))
    
    C_nodes_s = list({n for n, d in g_s.nodes(data=True) if d["bipartite"] == 0})
    V_nodes_s = list(set(g_s) - set(C_nodes_s))

    while len(g_s.edges()) > 0:
#         print('NEXT COLOR')
        bm=best_match(g_s)
#         print(bm)
#         matches_list.append(bm)
        
        # find the uniqe edges
        unique_match = dict((c_node, bm[c_node]) for c_node in bm if c_node in C_nodes)
        edges_list = [(c_node, bm[c_node]) for c_node in bm if c_node in C_nodes_s]
        matches_list.append(unique_match)
        
        g_s.remove_edges_from(edges_list)
#     assert len(g.edges()) == 0
        
    return matches_list

def ColorationCircuit(H):
    G = BipartitieGraphFromCheckMat(H)
    matches_list = edge_corloring(G)
    
    scheduling_list = []
    for match in matches_list:
        # Only keep edges that exist in H; use sparse-friendly indexing
        scheduling = {}
        for c_node, v_neighbor in match.items():
            row_idx = -c_node - 1
            col_idx = v_neighbor - 1
            if H[row_idx, col_idx] == 1:
                scheduling[row_idx] = col_idx
        scheduling_list.append(scheduling)
    
    return scheduling_list

# special circuit for the weight-8 self-dual BB code
# def BB_SD_circuit(H):
#     sequence = [0, 4, 2, 6, 1, 5, 3, 7]
#     BB_scheduling = []

#     for t in range(8):
#         scheduling = {}
#         s_t = sequence[t]
#         for i in range(H.shape[0]):
#             H_ixs = np.where(H[i] == 1)[0]
#             scheduling[i] = int(H_ixs[s_t])
#         BB_scheduling.append(scheduling)
#     return BB_scheduling

def IxsToCoors(ixs, mat_size):
    l,m = list(mat_size.values())
    
    coors = []
    for ix in ixs:
        if ix < l*m:
            vec = np.zeros(l*m)
            vec[ix] = 1
            mat = np.reshape(vec, (l,m))
            r_ixs, c_ixs = np.where(mat == 1)
            coors.append([np.array([r_ixs[0], c_ixs[0]]), 'L'])
        else:
            vec = np.zeros(l*m)
            vec[ix - l*m] = 1
            mat = np.reshape(vec, (l, m))
            r_ixs, c_ixs = np.where(mat == 1)
            coors.append([np.array([r_ixs[0], c_ixs[0]]), 'R'])
    return coors

def CoorsToIxs(coors, mat_size):
    l,m = list(mat_size.values())
    
    ixs = []
    for coor in coors:
        coor, block = coor
        pauli = CoorToPauli([coor], mat_size, block)
        ixs.append(np.where(pauli == 1)[0][0])
    return ixs

def ShiftCoors(coors, shift, mat_size):
    shifted_coors = []
    for coor in coors:
        coor, block = coor
        shifted_coor = Mod(coor + shift, mat_size)
        shifted_coors.append([shifted_coor, block])
    return shifted_coors

def BB_SD_circuit(H, sequence):
    # sequence = [0, 4, 2, 6, 1, 5, 3, 7] # d = 5
    # sequence = [0, 1, 2, 3, 4, 5, 6, 7]
    # sequence = [0, 1, 4, 5, 2, 3, 6, 7]
    # sequence = [0, 1, 6, 7, 4, 5, 2, 3]
    
    # sequence = [0, 3, 4, 7, 1, 2, 5, 6] # d = 7
    # sequence = [0, 3, 5, 6, 1, 2, 4, 7] # d = 6
    # sequence = [0, 4, 3, 7, 1, 5, 2, 6] # d = 6
    # sequence = [0, 1, 2, 3, 4, 5, 6, 7]
    # random.shuffle(sequence)
    # print('sequence:', sequence)

    # sequence = [4, 5, 7, 2, 0, 6, 1, 3]
    # sequence = [3, 4, 1, 2, 5, 7, 6, 0]
    print('sequence:', sequence)

    l, m = 3, int(H.shape[0]/3)
    mat_size = {'l':l, 'm':m}
    
    check = H[0]
    ixs = np.where(check == 1)[0][sequence]

    coors = IxsToCoors(ixs, mat_size)

    BB_scheduling = []

    for t in range(8):
        scheduling = {}
        for i in range(H.shape[0]):
            shift = np.array([i//m, i%m])
            check_schedulings = CoorsToIxs(ShiftCoors(coors, shift, mat_size), mat_size)
            
            scheduling[i] = int(check_schedulings[t])
        BB_scheduling.append(scheduling)
    return BB_scheduling


# Random circuit
def RandomCircuit(H):
    # Obtain a random scheduling 
    rand_scheduling_seed = 30000
    num_checks, num_bits = H.shape
    max_stab_w = max([int(np.sum(H[i,:])) for i in range(num_checks)])
    scheduling_list = [list(np.where(H[ancilla_index,:] == 1)[0]) for ancilla_index in range(num_checks)]
    [random.Random(i + rand_scheduling_seed).shuffle(scheduling_list[i]) for i in range(len(scheduling_list))]
    
    schedulings = []
    for time_step in range(max_stab_w):
        scheduling = {}
        for ancilla_index in range(num_checks):
            if len(scheduling_list[ancilla_index]) >= time_step + 1:
                scheduling[ancilla_index] = scheduling_list[ancilla_index][time_step]
        schedulings.append(scheduling)
    return schedulings



# ColorProductCircuit
def QubitIndexToPos(q_index, n_C, n_V):
    if q_index <= n_V**2 - 1:
        i = q_index//n_V
        j = q_index - i*n_V
        return i, j + n_C
    else:
        q_index -= n_V**2
        i = q_index//n_C
        j = q_index - i*n_C
        return i + n_V, j   
    
def XcheckIndexToPos(X_index, n_C, n_V):
    i = X_index//n_V
    j = X_index - i*n_V
    
    return n_V + i, n_C + j

def ZcheckIndexToPos(Z_index, n_C, n_V):
    i = Z_index//n_C
    j = Z_index - i*n_C
    
    return i, j

def GetPosToQubitIndexMap(n_C, n_V):
    n_q = (n_C)**2 + (n_V)**2
    map = {}
    for i in range(n_q):
        q_pos = QubitIndexToPos(i, n_C, n_V)
        map[q_pos] = i
    return map

def GetPosToZCheckIndexMap(n_C, n_V):
    n_Z = n_V*n_C
    map = {}
    for i in range(n_Z):
        q_pos = ZcheckIndexToPos(i, n_C, n_V)
        map[q_pos] = i
    return map

def GetPosToXCheckIndexMap(n_C, n_V):
    n_X = n_C*n_V
    map = {}
    for i in range(n_X):
        q_pos = XcheckIndexToPos(i, n_C, n_V)
        map[q_pos] = i
    return map


def ClassicalCheckFromQuantumCheck(h, check_type):
    n = h.shape[1]
    n0 = int(np.sqrt(n/25))
    n_C, n_V = int(3*n0), int(4*n0)
    
    q_pos_index_map = GetPosToQubitIndexMap(n_C, n_V)
    Z_pos_index_map = GetPosToZCheckIndexMap(n_C, n_V)
    X_pos_index_map = GetPosToXCheckIndexMap(n_C, n_V)
        
    if check_type == 'Z':
        row_Zchecks = [Z_pos_index_map[0, i] for i in range(n_C)]
        row_qubits = [q_pos_index_map[0, i + n_C] for i in range(n_V)]

        H = np.zeros([len(row_Zchecks), len(row_qubits)])

        for i in range(len(row_Zchecks)):
            for j in range(len(row_qubits)):
                H[i,j] = h[row_Zchecks[i], row_qubits[j]]
    else:
        column_Xchecks = [X_pos_index_map[i + n_V, n_C] for i in range(n_C)]
        columm_qubits = [q_pos_index_map[i, n_C] for i in range(n_V)]

        H = np.zeros([len(column_Xchecks), len(columm_qubits)])

        for i in range(len(column_Xchecks)):
            for j in range(len(columm_qubits)):
                H[i,j] = h[column_Xchecks[i], columm_qubits[j]]
    return H

def ColorProductCircuit(q_h, check_type):
    assert check_type in ['X', 'Z'], f'check_type should be either X or Z'
    n = q_h.shape[1]
    n0 = np.sqrt(n/25)
    n_C, n_V = int(3*n0), int(4*n0)
    
    q_pos_index_map = GetPosToQubitIndexMap(n_C, n_V)
    Z_pos_index_map = GetPosToZCheckIndexMap(n_C, n_V)
    X_pos_index_map = GetPosToXCheckIndexMap(n_C, n_V)
    
    c_h = ClassicalCheckFromQuantumCheck(q_h, check_type)
    classical_coloration_scheduling = ColorationCircuit(c_h)
    
    scheduling = []
    
    if check_type == 'Z':
        # add the vertical connections
        for c in classical_coloration_scheduling:
            v_c = {}
            for c_index in list(c.keys()):
                q_y = c_index + n_V
                Z_y = c[c_index]
                for q_x, Z_x in zip(range(n_C), range(n_C)):
                    q_pos = q_y, q_x
                    q_index = q_pos_index_map[q_pos]
                    Z_pos = Z_y, Z_x
                    Z_index = Z_pos_index_map[Z_pos]
                    v_c[Z_index] = q_index
            scheduling.append(v_c)

            h_c = {}
            for c_index in list(c.keys()):
                Z_x = c_index 
                q_x = c[c_index] + n_C
                for q_y, Z_y in zip(range(n_V), range(n_V)):
                    q_pos = q_y, q_x 
                    q_index = q_pos_index_map[q_pos]
                    Z_pos = Z_y, Z_x
                    Z_index = Z_pos_index_map[Z_pos]
                    h_c[Z_index] = q_index
            scheduling.append(h_c)
            
    else:
        for c in classical_coloration_scheduling:
            v_c = {}
            for c_index in list(c.keys()):
                X_y = c_index + n_V
                q_y = c[c_index]
                for q_x, X_x in zip(n_C + np.arange(n_V), n_C + np.arange(n_V)):
                    q_pos = q_y, q_x
                    q_index = q_pos_index_map[q_pos]
                    X_pos = X_y, X_x
                    X_index = X_pos_index_map[X_pos]
                    v_c[X_index] = q_index
            scheduling.append(v_c)

            h_c = {}
            for c_index in list(c.keys()):
                q_x = c_index 
                X_x = c[c_index] + n_C
                for q_y, X_y in zip(n_V + np.arange(n_C), n_V + np.arange(n_C)):
                    q_pos = q_y, q_x
                    q_index = q_pos_index_map[q_pos]
                    X_pos = X_y, X_x
                    X_index = X_pos_index_map[X_pos]
                    h_c[X_index] = q_index
            scheduling.append(h_c)
    return scheduling   
