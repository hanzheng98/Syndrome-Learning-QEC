import numpy as np
import matplotlib.pyplot as plt
from graph_tools import Graph
import networkx as nx
import random
import copy
import time
import json

import ldpc
import bposd

# from bposd.css_decode_sim import css_decode_sim
# from bposd.hgp import hgp
import pickle

import multiprocessing as mp
import random

from scipy.optimize import curve_fit
from abc import ABC, abstractmethod

# BPOSD decoder
from bposd import bposd_decoder

import gurobipy as gp
from gurobipy import GRB
import scipy
import pymatching
import relay_bp
from ldpc.bplsd_decoder import BpLsdDecoder
from ldpc.bposd_decoder import BpOsdDecoder




class BaseCircuitDecoder:

    def __init__(self,
                 decoder_params: dict):
        self.decoder_params = decoder_params
        

    @abstractmethod
    def set_decoder(self, space_time_code_params):
        pass

    @abstractmethod
    def decode(self, detector_shots:np.ndarray):
        pass



class ToyDecoder(BaseCircuitDecoder):
    """
        A trivial decoder for DEMs:
      Given detector bits d (shape S×D) and H (D×E), find ONE solution e (S×E)
      to H e = d (mod 2). It ignores priors and chooses the free variables = 0.

        Works well as a baseline/demo decoder, especially for small D like repetition-3.
    """

    def __init__(self, decoder_params: dict = None):
        super().__init__(decoder_params or {})
        self.H = None
        self.L = None
        self.priors = None
        self.U = None            # row-operation matrix so that R = U @ H (mod 2) is in RREF
        self.pivot_cols = None   # indices of pivot columns in original ordering
        self.rank = 0

    def set_decoder(self,
                    space_time_code_params):
        H = np.asarray(space_time_code_params['H'], dtype=np.uint8) & 1
        L = np.asarray(space_time_code_params.get('L', np.zeros((0, H.shape[1]), dtype=np.uint8)), dtype=np.uint8) & 1
        priors = np.asarray(space_time_code_params.get('channel_probs', np.zeros(H.shape[1], dtype=float)), dtype=float)

        self.H, self.L, self.priors = H, L, priors
        self._precompute_rref_operators()

    def _precompute_rref_operators(self):
        """
        Compute U and pivot columns so that R = (U @ H) % 2 is in reduced row echelon form.
        In R, the first `rank` rows each have a pivot '1' in column pivot_cols[k], zeros elsewhere in that column.
        """
        A = self.H.copy()
        m, n = A.shape
        U = np.eye(m, dtype=np.uint8)
        pivot_cols = []
        row = 0
        for col in range(n):
            if row >= m:
                break
            # find a pivot row with A[r, col] == 1 at or below current row
            pivot = None
            for r in range(row, m):
                if A[r, col]:
                    pivot = r
                    break
            if pivot is None:
                continue
            # swap pivot row up to `row`
            if pivot != row:
                A[[row, pivot]] = A[[pivot, row]]
                U[[row, pivot]] = U[[pivot, row]]
            # eliminate this column in all other rows
            for r in range(m):
                if r != row and A[r, col]:
                    A[r, :] ^= A[row, :]
                    U[r, :] ^= U[row, :]
            pivot_cols.append(col)
            row += 1

        self.U = U                 # R = (U @ H) % 2
        self.pivot_cols = np.asarray(pivot_cols, dtype=int)
        self.rank = len(pivot_cols)

    def decode(self, detector_shots: np.ndarray):
        """Return corrections as an (S×E) array e s.t. H e = d (mod 2), choosing free vars = 0."""
        d = (np.asarray(detector_shots, dtype=np.uint8) & 1)
        S, D = d.shape
        assert D == self.H.shape[0], f"detector_shots has {D} bits but H has {self.H.shape[0]} rows"

        # Apply the same row operations to d:  b' = U @ d  (for row-vectors: d' = d @ U^T)
        bprime = (d @ self.U.T) % 2  # shape (S, D)

        # Build a solution with free variables = 0: x[pivot_col[k]] = b'_k  (for k in 0..rank-1)
        E = self.H.shape[1]
        e = np.zeros((S, E), dtype=np.uint8)
        if self.rank > 0:
            e[:, self.pivot_cols] = bprime[:, :self.rank]

        # (Optional) sanity: verify H e == d
        # resid = (d + (e @ self.H.T) % 2) % 2
        # assert resid.sum() == 0

        return e


class BPOSD_Decoder():
    def __init__(self, BPOSD_params):
        self.BPOSD_params = BPOSD_params
    
    def set_decoder(self, space_time_code_params):
        h, channel_probs = space_time_code_params['H'], space_time_code_params['channel_probs']
        
        max_iter=self.BPOSD_params['max_iter']
        bp_method=self.BPOSD_params['bp_method']
        ms_scaling_factor=self.BPOSD_params['ms_scaling_factor']
        osd_method=self.BPOSD_params['osd_method']
        osd_order=self.BPOSD_params['osd_order']
        
        self.decoder = bposd_decoder(
                h,
                channel_probs=channel_probs,
                max_iter=max_iter,
                bp_method=bp_method,
                ms_scaling_factor=ms_scaling_factor,
                osd_method=osd_method,
                osd_order=7)
        
        # self.h = h

    def decode(self, detector_shots:np.ndarray):
        error_corrections = []
        for detector_shot in detector_shots:
            self.decoder.decode(detector_shot)
            error_corrections.append(self.decoder.osdw_decoding)
        return np.array(error_corrections)
  

class BPLSD_Decoder():
    def __init__(self, BPLSD_params):
        self.BPOSD_params = BPLSD_params
    
    def set_decoder(self, space_time_code_params):
        h, channel_probs = space_time_code_params['H'], space_time_code_params['channel_probs']
        
        max_iter=self.BPOSD_params['max_iter']
        bp_method=self.BPOSD_params['bp_method']
        lsd_order=self.BPOSD_params['lsd_order']
        schedule=self.BPOSD_params['schedule']
        ms_scaling_factor=self.BPOSD_params['ms_scaling_factor']
        lsd_method=self.BPOSD_params['lsd_method'] 
        self.decoder = BpLsdDecoder(h, error_channel=channel_probs, bp_method = bp_method, max_iter = max_iter, ms_scaling_factor=ms_scaling_factor, schedule = schedule, lsd_method = lsd_method, lsd_order = lsd_order)
        

    def decode(self, detector_shots:np.ndarray):
        error_corrections = []
        for detector_shot in detector_shots:
            # self.decoder.decode(detector_shot)
            error_corrections.append(self.decoder.decode(detector_shot))
        return np.array(error_corrections)



class BPOSD_Decoder_V2():
    def __init__(self, BPOSD_params):
        self.BPOSD_params = BPOSD_params
    
    def set_decoder(self, space_time_code_params):
        h, channel_probs = space_time_code_params['H'], space_time_code_params['channel_probs']
        
        max_iter=self.BPOSD_params['max_iter']
        bp_method=self.BPOSD_params['bp_method']
        ms_scaling_factor=self.BPOSD_params['ms_scaling_factor']
        osd_method=self.BPOSD_params['osd_method']
        osd_order=self.BPOSD_params['osd_order']
        self.decoder = BpOsdDecoder(h, error_channel=list(channel_probs), max_iter= max_iter, bp_method=bp_method, ms_scaling_factor= ms_scaling_factor, schedule='parallel', omp_thread_count=4, osd_method=osd_method, osd_order=osd_order)
        

    def decode(self, detector_shots:np.ndarray):
        error_corrections = []
        for detector_shot in detector_shots:
            # self.decoder.decode(detector_shot)
            error_corrections.append(self.decoder.decode(detector_shot))
        return np.array(error_corrections)


class MLE_Decoder():
    def __init__(self, env):
        self.env = env
    
    def set_decoder(self, space_time_code_params):
        H, channel_probs = space_time_code_params['H'], space_time_code_params['channel_probs']
        self.weights = list(np.log(np.array(channel_probs) / (1 - np.array(channel_probs))))
        hyperedges_matrix = scipy.sparse.lil_matrix(H.T, dtype=bool)
        # Set hyperedges
        self.hyperedges = []
        for row in hyperedges_matrix.T:
            self.hyperedges.append(list(np.argwhere(row)[:, 1].flatten()))


    def decode(self, detector_shots:np.ndarray):
        error_corrections = []
        
        for (d, detector_shot) in enumerate(detector_shots):
            # print(d)
            # Create a new model
            m = gp.Model('mip1', env=self.env)

            # Create variables
            error_variables = []
            hyperedge_variables = []
            objective = 0

            # Set objective function
            for i in range(len(self.weights)):
                error_variables.append(m.addVar(vtype=GRB.BINARY, name='e' + str(i)))
                objective += self.weights[i] * error_variables[i]
            m.setObjective(objective, GRB.MAXIMIZE)

            # Set constraints
            for i in range(len(self.hyperedges)):
                hyperedge_variables.append(m.addVar(vtype=GRB.INTEGER, name='h' + str(i),
                                                    ub=len(self.hyperedges[i]), lb=0))
                constraint = 0
                for j in self.hyperedges[i]:
                    constraint += error_variables[j]
                constraint -= 2 * hyperedge_variables[i]
                m.addConstr(constraint == detector_shot[i], name='c' + str(i))

            # Optimize model
            m.optimize()
            if m.status != 2 and verbose:  # Print error code if optimal solution not found
                print('Did not find optimal solution', m.status)
            error = np.round(np.array([e.X for e in error_variables]), decimals=0).astype(int)
            # print(error)

            error_corrections.append(error)
        
        return np.array(error_corrections)


class ReplayBP_Decoder():
    def __init__(self, ReplayBP_params):
        self.ReplayBP_params = ReplayBP_params
    
    def set_decoder(self, space_time_code_params):
        h, channel_probs = space_time_code_params['H'], space_time_code_params['channel_probs']
        
        gamma0=self.ReplayBP_params['gamma0'] # Uniform memory weight for the first ensemble
        pre_iter=self.ReplayBP_params['pre_iter'] # Max BP iterations for the first ensemble
        num_sets=self.ReplayBP_params['num_sets'] # Number of relay ensemble elements
        set_max_iter=self.ReplayBP_params['set_max_iter'] # Max BP iterations per relay ensemble
        gamma_dist_interval=self.ReplayBP_params['gamma_dist_interval'] # Set the uniform distribution range for disordered memory weight selection
        stop_nconv=self.ReplayBP_params['stop_nconv']
        
        self.decoder = relay_bp.RelayDecoderF32(
            h,
            error_priors=channel_probs, # Set the priors probability for each error
            gamma0=gamma0, # Uniform memory weight for the first ensemble
            pre_iter=pre_iter, # Max BP iterations for the first ensemble
            num_sets=num_sets, # Number of relay ensemble elements
            set_max_iter=set_max_iter, # Max BP iterations per relay ensemble
            gamma_dist_interval=gamma_dist_interval, # Set the uniform distribution range for disordered memory weight selection
            stop_nconv=stop_nconv, # Number of relay solutions to find before stopping (the best will be selected)
        )

    def decode(self, detector_shots:np.ndarray):
        return self.decoder.decode_batch(detector_shots.astype(np.uint8))

# class MatchingDecoder():
#     def __init__(self):
#         # self.env = env
    
#     def set_decoder(self, space_time_code_params):
#         H, channel_probs = space_time_code_params['H'], space_time_code_params['channel_probs']
#         weights = list(np.log(np.array(channel_probs) / (1 - np.array(channel_probs))))
#         self.decoder = pymatching.Matching(H, weights=weights)


#     def decode(self, detector_shots:np.ndarray):
#         correction = self.decoder.decode(np.array([0, 1, 0, 1]))
        
#         return correction

