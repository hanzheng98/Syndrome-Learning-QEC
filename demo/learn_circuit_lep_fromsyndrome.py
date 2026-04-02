'''
Learn the logical channel from the syndrome information with circuit level demonstration

Written by Han Zheng

Demo code to run and use the package 
'''

import numpy as np
import time
from bposd.css import css_code
from beliefmatching import detector_error_model_to_check_matrices
from sim_qec.codes_family.hpc_lp import rotated_surface_code_checks
from sim_qec.dem_sim import DEMSyndromeExtraction
from sim_qec.circuit_lep_prediction import PredictPriors
from sim_qec.circuit_decoders import BPLSD_Decoder


'''

running the simple repetition code circuit with syndrome extraction

'''

'''
p_i is the single qubit idling error prob 
p_state_p is the state preparation error prob
p_m is the measurement error prob
p_CX is the two qubit gate error prob (setting to be 0.0)
p_idling_gate is the idling error during the gate operation also two qubits (setting to be 0.0)
'''


# CIRCUIT_ERROR_PARAMS = {"p_i": 0.01, "p_state_p": 0.01, "p_m": 0.01, "p_CX":0.0, "p_idling_gate": 0.0}

# we now build a a three-qubit repetition code for demonstration
# Hz = np.array([[1,1,0],
#                 [0,1,1]])
# Hx = np.array([[0,0,0]])
# rep_code = css_code(Hx, Hz)



# If we try to use the general CSS code, Example: [[7, 1, 3]] Steane code

# X_stabilizers = [[3,4,5,6],[1,2,5,6],[0,2,4,6]]
# Z_stabilizers = [[3,4,5,6],[1,2,5,6],[0,2,4,6]]
# Hx = get_parity_check_matrix(X_stabilizers, 7)
# Hz = get_parity_check_matrix(Z_stabilizers, 7)


# steane_code = css_code(Hx, Hz)
# # print(f'parity check for steane code Hx shape: {Hx.shape}, Hz shape: {Hz.shape}')


'''
General implementation: with a code family: we choose to be the surface code here
'''

d = 3
Hx, Hz = rotated_surface_code_checks(d)

print(f'Shape of Hx: {Hx.shape}, Shape of Hz: {Hz.shape}')

print(f'check the CSS condition Hx*Hz^T = 0: {np.all((Hx @ Hz.T) % 2 == 0)}')

surface_code = css_code(Hx, Hz)

print(f'Shape of surface_code.hx: {surface_code.hx.shape}, Shape of surface_code.hz: {surface_code.hz.shape}')


'''
Just in case the num qubits get too large: use the folloiwng codes

[[7, 1, 3]] #color code 
[[17, 1, 5]] #color code 
[[23, 1, 7]] #code
'''


# H7 = np.array([
#     [1,0,0,1,0,1,1],
#     [0,1,0,1,1,0,1],
#     [0,0,1,0,1,1,1],
# ], dtype=int)
# HX_7 = H7.copy()
# HZ_7 = H7.copy()

# ============================================================
# [[17, 1, 5]]  (planar 4.8.8 color code)
# Choose 8 Z-type face checks on qubits {1..17}; X-type checks use the same supports.
# These faces are taken from Table 7 (17-qubit color code) in Chamberland & Beverland (2018). :contentReference[oaicite:2]{index=2}
# faces_17 = [
#     {1,2,3,4},
#     {1,3,5,6},
#     {5,6,9,10},
#     {7,8,11,12},
#     {9,10,13,14},
#     {11,12,15,16},
#     {8,12,16,17},
#     {3,4,6,7,10,11,14,15},   # weight-8 plaquette
# ]
# def row_from_face(S, n=17):
#     v = np.zeros(n, dtype=int)
#     for i in S:
#         v[i-1] = 1  # 1-based -> 0-based
#     return v
# H17 = np.vstack([row_from_face(F) for F in faces_17])
# HX_17 = H17.copy()
# HZ_17 = H17.copy()
# print(f'checking the CSS condition: {np.all(HX_17 @ HZ_17.T % 2 == 0 )}')
# print(f'checking the maximum weights: {np.sum(HX_17, axis=1), np.sum(HZ_17, axis=1)}')
# color_code = css_code(HX_7, HZ_7)


# ============================================================
# [[23, 1, 7]]  (quantum Golay)
# Use the standard 11×23 parity-check matrix H = [ M | I_11 ] for the classical [23,12,7] Golay code.
# Because the Golay code is weakly self-dual (its dual [23,11,8] even subcode is contained in it),
# we may set HX = HZ = H, which indeed obeys HX @ HZ.T == 0 over F2.
# Matrix M below is from MathWorld; see "Golay Code" (parity-check H = (M, I_11)). :contentReference[oaicite:3]{index=3}
# M = np.array([
#     [1,0,0,1,1,1,0,0,0,1,1,1],
#     [1,0,1,0,1,1,0,1,1,0,0,1],
#     [1,0,1,1,0,1,1,0,1,0,1,0],
#     [1,0,1,1,1,0,1,1,0,1,0,0],
#     [1,1,0,0,1,1,1,0,1,1,0,0],
#     [1,1,0,1,0,1,1,1,0,0,0,1],
#     [1,1,0,1,1,0,0,1,1,0,1,0],
#     [1,1,1,0,0,1,0,1,0,1,1,0],
#     [1,1,1,0,1,0,1,0,0,0,1,1],
#     [1,1,1,1,0,0,0,0,1,1,0,1],
#     [0,1,1,1,1,1,1,1,1,1,1,1],
# ], dtype=int)
# H23 = np.hstack([M, np.eye(11, dtype=int)])
# HX_23 = H23.copy()
# HZ_23 = H23.copy()


# color_code = css_code(HX_23, HZ_23)
# print(f'checking the CSS condition: {np.all(HX_23 @ HZ_23.T % 2 == 0 )}')
# print(f'number of logical: {color_code.lx.shape}')

# print("Code parameters:", surface_code.hz)
# print('the X parity check', surface_code.hx)
# print(f'the Lz logical operators: {surface_code.lz}')
# print(f'the Lx logical operators: {surface_code.lx}')



# Circuit error knobs (per-location probabilities).
# A physical_error_rate "p" will uniformly scale these inside DEMSyndromeExtraction.
CIRCUIT_ERROR_PARAMS = {
    "p_i": 1.0,          # idling single-qubit error
    "p_state_p": 0.8,    # state prep error
    "p_m": 0.9,          # measurement error
    "p_CX": 0.0,         # 2-qubit depolarizing error (not used in 'repetition' mode below)
    "p_idling_gate": 0.0 # idling during gates
}

num_cycles = 1
p = 5e-4  # overall physical error scale; you can start with 0.01 and tweak

dem = DEMSyndromeExtraction(
    code=surface_code,
    num_cycles=num_cycles,
    circuit_error_params=CIRCUIT_ERROR_PARAMS,
    physical_error_rate=p,
)

# Build the repetition-code circuit; fault_type "X_ERROR" injects X noise where configured
# circ = dem.build_repetition_circuit(fault_type="X_ERROR")
circ = dem.build_circuit(fault_type='DEPOLARIZE1')
# print(circ)  # helpful when debugging
# print(circ.diagram("timeline"))

# Get the detector error model (DEM) and a sampler
det_model = circ.detector_error_model(flatten_loops=True)
print("Detectors:", det_model.num_detectors)
print("Observables:", det_model.num_observables)

sampler = circ.compile_detector_sampler()
# Each shot returns: (detector bits, obs bits) if separate_observables=True
det_vals, log_vals = sampler.sample(
    shots=5000000, separate_observables=True
)
print("detector bit array shape:", det_vals.shape)
print("observable bit array shape:", log_vals.shape)
# print("First 10 detector results:\n", det_vals[:10,0])
det_vals = det_vals.astype(int) #convert from bool to int
log_vals = log_vals.astype(int)
# print("First 10 detector results:\n", det_vals[:10,0])
q = det_vals.mean(axis=0)
parity_expectation = 1.0 - 2.0 * q
print("Syndrome expectation values:", parity_expectation)

#now we compute the syndrome expectation values



dem = circ.detector_error_model(flatten_loops=True)
dem_matrix = detector_error_model_to_check_matrices(dem, allow_undecomposed_hyperedges=True)
h = dem_matrix.check_matrix.toarray()

# check very carefully on the logical error channel defined here. what does it mean for a logical error to take place. 
l = dem_matrix.observables_matrix.toarray()
channel_probs = dem_matrix.priors
print(f'Check matrix shape: {h.shape}, number of faults: {len(channel_probs)}')
# print(f'Channel error probabilities: {channel_probs}')
# print(f'sum of priors" {sum(p)}')
space_time_code_params = {'H': h, 'L': l, 'channel_probs': channel_probs}





dem_samples = PredictPriors(
    dectector_samples=det_vals,
    check_matrix=h,
    subsample=True,
)
A_syndrome, sample_stabs = dem_samples._build_A_matrix_syndromes()
print(f'A syndrome matrix shape: {A_syndrome.shape}')
print(f'checking the rank of this matrix: {np.linalg.matrix_rank(A_syndrome)}')
sample_stab_eigs = dem_samples._get_syndrome_expectations(sample_stabs=sample_stabs)
print(f'sample stabilizer eigenvalues: {sample_stab_eigs}')
# we now solve the linear system to get the priors
predicted_priors = dem_samples.predict_priors(A_syndrome, sample_stab_eigs, mode='rip')
print('-' * 20 )
print(f'testing the priors predictions here')
print('-' * 20 )
print(f'predicted priors (first 10): {predicted_priors[:10]}')
print(f'true priors (first 10): {channel_probs[:10]}')

print(f'comparing the predicted priors vs true priors')
print(f' the L2 norm between predicted priors and true priors: {np.linalg.norm(np.array(predicted_priors) - np.array(channel_probs))}')
print(f' the maximum absolute difference between predicted priors and true priors: {np.max(np.abs(np.array(predicted_priors) - np.array(channel_probs)))}')
print(f' the maximum relative difference between predicted priors and true priors: {np.max(np.abs((np.array(predicted_priors) - np.array(channel_probs)) / np.array(channel_probs)))}')

print('-' * 20 )
print(f'comparing the predicted logical error rate vs sampled logical error rate')
print('-' * 20 )

# decoder = relay_decoder

# setting up the BPLSD decoder
BPLSD_PARAMS = {'max_iter':5, 'bp_method':'min_sum', 'ms_scaling_factor':0.5, 'schedule':'parallel', 'lsd_method': 'lsd_e', 'lsd_order':3}
bplsd_decoder = BPLSD_Decoder(BPLSD_params=BPLSD_PARAMS)
decoder = bplsd_decoder
decoder.set_decoder(space_time_code_params)

# sample the logical error rate
lep_sampled_timer = time.perf_counter()
corrections = decoder.decode(det_vals)
les = 1 * ((log_vals + corrections @ l.T % 2) % 2).any(axis=1)
lep_sampled = np.average(les)
lep_sampled_runtime = time.perf_counter() - lep_sampled_timer

print('-' * 20 )

print(f'Logical observables matrix shape: {l.shape}')
print(f'detector shape {h.shape}')
print(f'correction shape {corrections.shape}')
print(f'log_vals shape {log_vals.shape}')

print('-' * 20)

print('Sampled logical error probability:', lep_sampled)
print(f'Runtime for sampled logical error probability: {lep_sampled_runtime:.6f} s')

# sampled the logical error rate with the predicted priors

lep_predicted_timer = time.perf_counter()
lep_predicted  = dem_samples.predict_logical_error_efficient(decoder=decoder,
                                                   observables_matrix=l,
                                                   priors=predicted_priors,
                                                   max_order=4)
lep_predicted_runtime = time.perf_counter() - lep_predicted_timer

print('Predicted Logical error probability:', lep_predicted)
print(f'Runtime for predicted logical error probability: {lep_predicted_runtime:.6f} s')


lep_true_timer = time.perf_counter()
lep_true = dem_samples.predict_logical_error_efficient(decoder=decoder,
                                                    observables_matrix=l,
                                                    priors=channel_probs,
                                                    max_order=4)
lep_true_runtime = time.perf_counter() - lep_true_timer

print('True Logical error probability:', lep_true)
print(f'Runtime for true logical error probability: {lep_true_runtime:.6f} s')
















