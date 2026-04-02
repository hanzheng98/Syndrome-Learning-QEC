import numpy as np
import bposd
from ldpc.codes import ring_code
from bposd.hgp import hgp
from bposd import bposd_decoder
import multiprocessing as mp
from bposd.css import css_code


# Computing the code dimenson 
def BinaryRepMat(mat):
    rows = [int(''.join(map(str, list(row))), 2) for row in mat]
    return rows
def gf2_rank(rows):
    """
    Find rank of a matrix over GF2.

    The rows of the matrix are given as nonnegative integers, thought
    of as bit-strings.

    This function modifies the input list. Use gf2_rank(rows.copy())
    instead of gf2_rank(rows) to avoid modifying rows.
    """
    rows = BinaryRepMat(rows)
    rank = 0
    while rows:
        pivot_row = rows.pop()
        if pivot_row:
            rank += 1
            lsb = pivot_row & -pivot_row
            for index, row in enumerate(rows):
                if row & lsb:
                    rows[index] = row ^ pivot_row
    return rank
def local_check(m, delta):
    check = np.random.randint(low=0, high=2, size=(m, delta))
    flag = 0 
    while gf2_rank(check) != m or flag < 10000: 
        check = np.random.randint(low=0, high=2, size=(m, delta))
        flag = 1
    return check
def code_rate(hx, hz):
    rank_hx = gf2_rank(hx)
    rank_hz = gf2_rank(hz)
    n = hx.shape[1]
    k = n - (rank_hx + rank_hz)
    return k, np.round(k/n, 5)





# Modify the multiprocessing functions
def fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))


def parmap(f, X, nprocs=mp.cpu_count()):
    q_in = mp.Queue(1)
    q_out = mp.Queue()

    proc = [mp.Process(target=fun, args=(f, q_in, q_out))
            for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res)]



def DistanceEst_BPOSD(H, L, num_trials=1):    
    num_qubits = np.shape(H)[1]
    num_checks = np.shape(H)[0]
    num_logicals = np.shape(L)[0]
    
    # setup the decoder parameters
    decoder_params = {'channel_probs':0.1*np.ones(num_qubits), 'max_iter':int(num_qubits/20),
                     'bp_method':'min_sum', 'ms_scaling_factor':0.9, 'osd_method':'osd_e',
                     'osd_order':6}
    
    def SingleEst():
        # generate random logical operators to anticommute with
        logical = np.zeros(num_logicals)
        while np.sum(logical) == 0:
            logical = np.random.choice([0, 1], size=(num_logicals))@L%2

        combined_check = np.vstack([H, logical])
        combined_syndrome = np.zeros(num_checks + 1)
        combined_syndrome[-1] = 1

        # set up the decoder
        decoder = bposd_decoder(combined_check,
                    channel_probs=decoder_params['channel_probs'],
                    max_iter=decoder_params['max_iter'],
                    bp_method=decoder_params['bp_method'],
                    ms_scaling_factor=decoder_params['ms_scaling_factor'],
                    osd_method=decoder_params['osd_method'],
                    osd_order=decoder_params['osd_order'], )

        corr = decoder.decode(combined_syndrome)
        return np.sum(corr)
    
    # perform SingleEst in parallel
    eval_func = lambda _: SingleEst()      
    distances = parmap(eval_func, [0]*num_trials, nprocs = mp.cpu_count())
    
    return np.min(distances)