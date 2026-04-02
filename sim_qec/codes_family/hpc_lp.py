import numpy as np 
from scipy import sparse
import random
import galois 
from typing import Any, Callable, Dict, List, Optional, Type, Union, Tuple
import itertools
from sim_qec.codes_family.classical_codes import generate_random_binary_matrix, generate_even_ones_matrix, generate_even_support_matrix
from bposd.css import css_code
from sim_qec.codes_family.est_distance import DistanceEst_BPOSD, code_rate
import time
import collections
#         # Generate a random binary vector of length n
#         # with an even number of 1s
'''
Implementation for the homological product codes and lifted product codes. 

ToDO: added a parent class for the codes.
'''

GF2 = galois.GF(2)

Index  = Tuple[int, int]         # (row, col)
Values = List[int]               # list of ints at that entry


def randomise_lifts(
    lifts: Dict[Index, Values],
    m: int,
    n: int,
    l: int,
    *,
    seed: Optional[int] = None
) -> Dict[Index, Values]:
    """
    Randomise a 'lifts' dict.

    • If the original list is *exactly* [l], leave it unchanged.
    • If it has length 2  → replace by two distinct random ints in [0,l) or [l] randomly 
    • Otherwise (length 1 but not [l])  → replace by one random int in [0,l).
    """

    if seed is not None:
        random.seed(seed)

    out: Dict[Index, Values] = {}

    for (i, j), vals in lifts.items():
        # --------- rule 1: keep [l] unchanged ----------
        if len(vals) == 1 and vals[0] == l:
            out[(i, j)] = vals[:]          # copy as-is
            continue

        # --------- rule 2: produce 2 distinct ints ----------
        if len(vals) == 2:
            if l >= 2:
                 # Both options ("two distinct" and "[l]") are possible. Choose one randomly.
                if random.random() < 0.1:  # 50% chance for "two distinct random ints"
                    new_vals = random.sample(range(l), 2)  # distinct ints from [0, l-1]
                else:  # 50% chance for "[l]"
                    new_vals = [l]
            else:                                       # not enough symbols – fallback
                new_vals = [0, 0]
            out[(i, j)] = new_vals
            continue

        # --------- rule 3: ordinary single-value entry ----------
        out[(i, j)] = [random.randint(0, l - 1)]

    return out

def sample_lifts(
    base_lifts: Dict[Index, Values],
    m: int,
    n: int,
    l: int,
    num_samples: int,
    seed: Optional[int] = None
) -> List[Dict[Index, Values]]:
    """
    Generate `num_samples` randomised versions of `base_lifts`.
    """
    samples = []
    # Optionally seed once for reproducibility
    if seed is not None:
        random.seed(seed)
    for _ in range(num_samples):
        # no seed passed here so each is different
        samples.append(randomise_lifts(base_lifts, m, n, l))
    return samples


def rotated_surface_code_checks(d: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotated planar surface code of odd distance d, with n=d^2 data qubits.

    Qubits:
      - Coordinates (x,y) with x,y in {0,...,d-1}
      - Column index q(x,y) = x + d*y  (row-major)

    Generators:
      - Bulk: for each 2x2 cell with lower-left corner (x,y), include
          X-type if (x+y) even, Z-type if (x+y) odd.
      - Boundary: include (d-1) additional checks of each type, placed on
        alternating edge segments in a staggered way:
          * X-type on left edge for y odd, and right edge for y even.
          * Z-type on bottom edge for x even, and top edge for x odd.

    Output shapes:
      Hx, Hz have shape ((d^2-1)//2, d^2).
    """
    if d < 3 or (d % 2) == 0:
        raise ValueError("d must be an odd integer >= 3.")

    n = d * d

    def q(x: int, y: int) -> int:
        return x + d * y

    hx_rows = []
    hz_rows = []

    # -------------------------
    # Bulk 2x2 plaquette checks
    # -------------------------
    for y in range(d - 1):
        for x in range(d - 1):
            cols = [q(x, y), q(x + 1, y), q(x, y + 1), q(x + 1, y + 1)]
            if (x + y) % 2 == 0:
                hx_rows.append(cols)
            else:
                hz_rows.append(cols)

    # -------------------------
    # Boundary weight-2 checks
    # (staggered alternating segments)
    # -------------------------

    # Left boundary X checks at y odd: (0,y)-(0,y+1)
    for y in range(1, d - 1, 2):
        hx_rows.append([q(0, y), q(0, y + 1)])

    # Right boundary X checks at y even: (d-1,y)-(d-1,y+1)
    for y in range(0, d - 1, 2):
        hx_rows.append([q(d - 1, y), q(d - 1, y + 1)])

    # Bottom boundary Z checks at x even: (x,0)-(x+1,0)
    for x in range(0, d - 1, 2):
        hz_rows.append([q(x, 0), q(x + 1, 0)])

    # Top boundary Z checks at x odd: (x,d-1)-(x+1,d-1)
    for x in range(1, d - 1, 2):
        hz_rows.append([q(x, d - 1), q(x + 1, d - 1)])

    # Build binary matrices
    Hx = np.zeros((len(hx_rows), n), dtype=np.uint8)
    Hz = np.zeros((len(hz_rows), n), dtype=np.uint8)
    for i, cols in enumerate(hx_rows):
        Hx[i, cols] = 1
    for i, cols in enumerate(hz_rows):
        Hz[i, cols] = 1

    expected = (d * d - 1) // 2
    if Hx.shape != (expected, n) or Hz.shape != (expected, n):
        raise RuntimeError(f"Unexpected shapes: Hx={Hx.shape}, Hz={Hz.shape}, expected ({expected},{n}).")

    return Hx, Hz





#  parent class 



class CubicalCode:
    def __init__(self,
                 base_codes: List[Union[np.ndarray, sparse.csr_matrix]],
                ) -> None:
        pass 




class HGP: 
    def __init__(self, codes_lst: list) -> None:
        # Ensure each code is stored as an integer-valued numpy array.
        self.d = len(codes_lst)
        self.codes_lst = [np.array(code, dtype=int) for code in codes_lst]

    def build_totalcomplexes(self) -> dict: 
        # Build the total complexes from tensor products
        base_boundary = {1: self.codes_lst[1]}
        # Base case: combine the 0th and 1st classical codes.
        prod_boundaries = self._build_kunneth(self.codes_lst[0], base_boundary)
        if self.d == 2: 
            Hz = prod_boundaries[1]
            Hx = prod_boundaries[0]
            return Hz, Hx
        
        for i in range(2, self.d):
            Hb = self.codes_lst[i]
            prod_boundaries = self._build_kunneth(Hb, prod_boundaries)

        return prod_boundaries

    def _build_2dhgp(self, HA: np.array, HB: np.array):
        # Build 2-dimensional HGP
        ma, na = HA.shape
        mb, nb = HB.shape
        
        Hz2 = np.kron(HA, np.eye(nb, dtype=int))
        Hz1 = np.kron(np.eye(na, dtype=int), HB)
        partial2 = np.vstack((Hz1, Hz2))
        
        Hx2 = np.kron(np.eye(ma, dtype=int), HB)
        Hx1 = np.kron(HA, np.eye(mb, dtype=int))
        partial1 = np.hstack((Hx1, Hx2))

        return partial2, partial1

    def _build_kunneth(self, Hb: np.array, boundaries: dict) -> dict:
        # Build the total complexes given a long chain complex with a classical code.
        Hb = np.array(Hb, dtype=int)  # Ensure Hb is integer-valued.
        mb, nb = Hb.shape
        ell = len(boundaries)
        if ell < 1: 
            raise ValueError('Not enough classical codes to build from')
            
        prod_boundaries = dict()
        if ell == 1: 
            partial2, partial1 = self._build_2dhgp(Hb, boundaries[1])
            prod_boundaries[1] = partial2
            prod_boundaries[0] = partial1
            Hz = partial2.T 
            Hx = partial1
            return Hz, Hx
        else:
            for i in range(1, ell):
                mi, ni = boundaries[i].shape 
                # Ensure boundaries[i+1] is integer-valued.
                boundary_ip1 = np.array(boundaries[i+1], dtype=int)
                prod_boundary_1 = np.hstack((
                    np.kron(boundary_ip1, np.eye(mb, dtype=int)),
                    np.kron(np.eye(ni, dtype=int), Hb)
                ))
                prod_boundary_2 = np.hstack((
                    np.zeros((mi * nb, boundaries[i+1].shape[1] * mb), dtype=int),
                    np.kron(boundaries[i], np.eye(nb, dtype=int))
                ))
                prod_boundary = np.vstack((prod_boundary_1, prod_boundary_2))
                prod_boundaries[i+1] = prod_boundary
            
            n_ell = boundaries[ell].shape[1]
            m0 = boundaries[1].shape[0]
            prod_boundary_highest = np.vstack((
                np.kron(np.eye(n_ell, dtype=int), Hb),
                np.kron(boundaries[ell], np.eye(nb, dtype=int))
            ))
            prod_boundary_lowest = np.hstack((
                np.kron(boundaries[1], np.eye(mb, dtype=int)),
                np.kron(np.eye(m0, dtype=int), Hb)
            ))
            prod_boundaries[ell+1] = prod_boundary_highest
            prod_boundaries[1] = prod_boundary_lowest
            return prod_boundaries
        


# ToDo: implement high-dimensional LP codes

# ToDo: implement the basis for LP logical operator: using the brute-force search and analytical result


'''
Implemetation of lifted product codes.
'''

class LP:
    # ──────────────────────────────────────────────────────────────
    # Helper: accept either ((m, n), dict)  OR  dict
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def _normalize_mat(
        mat: Tuple[Tuple[int, int], Dict[Tuple[int, int], List[int]]] | Dict[
            Tuple[int, int], List[int]
        ],
    ) -> Tuple[Tuple[int, int], Dict[Tuple[int, int], List[int]]]:
        """
        Convert *mat* to canonical form  ``((m, n), dict)``.
        Raises ``TypeError`` if the object is neither format.
        """
        # Already canonical
        if isinstance(mat, tuple) and isinstance(mat[0], tuple):
            return mat

        # Bare dict → infer size
        if isinstance(mat, dict):
            rows = [r for r, _ in mat.keys()]
            cols = [c for _, c in mat.keys()]
            m = max(rows) + 1
            n = max(cols) + 1
            return (m, n), mat

        raise TypeError("Each lifted matrix must be dict or ((m,n), dict).")

    # ──────────────────────────────────────────────────────────────
    # Constructor (only part that changed)
    # ──────────────────────────────────────────────────────────────
    def __init__(
        self,
        lift_mats: List[
            Tuple[Tuple[int, int], Dict[Tuple[int, int], List[int]]]
            | Dict[Tuple[int, int], List[int]]
        ],
        lift_size: int,
        b: Optional[List[int]] = None,
    ):
        self.l = lift_size
        self.num_mats = len(lift_mats)

        if self.num_mats == 1:
            # Normalise A
            self.A = self._normalize_mat(lift_mats[0])
            self.b = b
            self.bbar = self._involution(b)

        elif self.num_mats == 2:
            # Normalise A and B
            self.A = self._normalize_mat(lift_mats[0])
            self.B = self._normalize_mat(lift_mats[1])
            self.b = b
        else:
            raise NotImplementedError("Lifted product codes are not implemented yet")

    # ──────────────────────────────────────────────────────────────
    # All methods below are **identical** to your original code
    # ──────────────────────────────────────────────────────────────
    def _to_regularreps(self, f: List[int]):
        f_vec = self._cyclic_maps(f)

        rep_matrix = GF2(np.zeros((self.l, self.l), dtype=int))
        for j in range(self.l):
            column_vec = np.roll(f_vec, shift=j)
            rep_matrix[:, j] = column_vec

        return rep_matrix

    def _involution(self, f: List[int]):
        inv_f = []
        for degree in f:
            inv_degree = self.l - degree
            if inv_degree == self.l:
                inv_degree = 0
            inv_f.append(inv_degree)

        inv_f = sorted(inv_f)
        return inv_f

    def lift_matrix(self, A: Tuple[Tuple[int, int], Dict[Tuple[int, int], List[int]]]):

        '''
        convert the lifted matrix A to a binary matrix
        A: ((m,n), {(i,j): [list of exponents]})
        return: binary matrix of size (m*l, n*l)
        0 <= exponents < l, [l] represents the zero polynomial
        '''
        m, n = A[0]
        binary_A = GF2(np.zeros((self.l * m, self.l * n), dtype=int))

        for i in range(m):
            for j in range(n):
                poly_exponent = A[1][(i, j)]
                reg_rep = self._to_regularreps(poly_exponent)
                row_start, row_end = i * self.l, (i + 1) * self.l
                col_start, col_end = j * self.l, (j + 1) * self.l
                binary_A[row_start:row_end, col_start:col_end] = reg_rep
        return binary_A

    def lift_kron(
        self,
        A: Tuple[Tuple[int, int], Dict[Tuple[int, int], List[int]]],
        B: Tuple[Tuple[int, int], Dict[Tuple[int, int], List[int]]],
    ):
        ma, na = A[0]
        mb, nb = B[0]
        C: Dict[Tuple[int, int], List[int]] = {}
        for (ra, ca), pa in A[1].items():
            for (rb, cb), pb in B[1].items():
                r = ra * mb + rb
                c = ca * nb + cb
                poly = self._polynomial_multiplication(pa, pb)
                C[(r, c)] = poly
        return C

    def _polynomial_multiplication(self, pa: List[int], pb: List[int]) -> List[int]:
        if pa == [self.l] or pb == [self.l]:
            return [self.l]
        poly = [(ka + kb) % self.l for ka in pa for kb in pb]
        counts = collections.Counter(poly)
        poly = [k for k, v in counts.items() if v % 2 != 0]
        poly.sort()
        return poly

    def build_LP_parity_checks(self, debug: bool = False):
        if self.num_mats == 1:
            m, n = self.A[0]
            b_reg = self._to_regularreps(self.b)
            bbar_reg = self._to_regularreps(self.bbar)
            binary_A = GF2(np.zeros((self.l * m, self.l * n), dtype=int))
            binary_b = GF2(np.zeros((self.l * m, self.l * m), dtype=int))
            binary_bbar = GF2(np.zeros((self.l * n, self.l * n), dtype=int))

            for i in range(m):
                for j in range(n):
                    # print('checking (i,j)', self.A[1][(i, j)])
                    poly_exponent = self.A[1][(i, j)]
                    reg_rep = self._to_regularreps(poly_exponent)
                    row_start, row_end = i * self.l, (i + 1) * self.l
                    col_start, col_end = j * self.l, (j + 1) * self.l
                    binary_A[row_start:row_end, col_start:col_end] = reg_rep

            for k in range(m):
                row_start, row_end = k * self.l, (k + 1) * self.l
                col_start, col_end = k * self.l, (k + 1) * self.l
                binary_b[row_start:row_end, col_start:col_end] = b_reg
            for k in range(n):
                row_start, row_end = k * self.l, (k + 1) * self.l
                col_start, col_end = k * self.l, (k + 1) * self.l
                binary_bbar[row_start:row_end, col_start:col_end] = bbar_reg

            Hx = np.hstack((binary_A, binary_b))
            Hz = np.hstack((binary_bbar, binary_A.T))
            return Hx, Hz

        elif self.num_mats == 2 and self.b is None:
            ma, na = self.A[0]
            A = self.A
            B = self.B
            mb, nb = self.B[0]
            Ina = self._build_identity_matrix(na)
            Inb = self._build_identity_matrix(nb)
            Ima = self._build_identity_matrix(ma)
            Imb = self._build_identity_matrix(mb)

            Bkron_na = self.lift_kron(((na, na), Ina), B)
            Bkron_ma = self.lift_kron(((ma, ma), Ima), B)
            Akron_nb = self.lift_kron(A, ((nb, nb), Inb))
            Akron_mb = self.lift_kron(A, ((mb, mb), Imb))

            binary_Bkron_na = self.lift_matrix(((na * mb, na * nb), Bkron_na))
            binary_Bkron_ma = self.lift_matrix(((ma * mb, ma * nb), Bkron_ma))
            binary_Akron_nb = self.lift_matrix(((nb * ma, nb * na), Akron_nb))
            binary_Akron_mb = self.lift_matrix(((mb * ma, mb * na), Akron_mb))

            if debug is True:
                Bkron_na_conjt = self.lift_kron(Ina, self._build_transpose_lifts(B))
                Akron_nb_conjt = self.lift_kron(self._build_transpose_lifts(A), Inb)
                binary_Bkron_na_conjt = self.lift_matrix(Bkron_na_conjt)
                binary_Akron_nb_conjt = self.lift_matrix(Akron_nb_conjt)

            Hx = GF2(np.hstack((binary_Akron_mb, binary_Bkron_ma)))
            if debug is True:
                Hz = GF2(np.hstack((binary_Bkron_na_conjt, binary_Akron_nb_conjt)))
            Hz = GF2(np.hstack((binary_Bkron_na.T, binary_Akron_nb.T)))

            return Hx, Hz
        elif self.num_mats == 2 and self.b is not None:
            pass 

    def _build_identity_matrix(self, n: int):
        identity_matrix: Dict[Tuple[int, int], List[int]] = {}
        for i in range(n):
            for j in range(n):
                if i == j:
                    identity_matrix[(i, j)] = [0]
                else:
                    identity_matrix[(i, j)] = [self.l]
        return identity_matrix

    def _build_transpose_lifts(
        self, A: Tuple[Tuple[int, int], Dict[Tuple[int, int], List[int]]]
    ) -> Tuple[Tuple[int, int], Dict[Tuple[int, int], List[int]]]:
        m, n = A[0]
        conj_transpose_A: Dict[Tuple[int, int], List[int]] = {}
        for (i, j), poly in A[1].items():
            conj_transpose_A[(j, i)] = self._involution(poly)
        return ((n, m), conj_transpose_A)

    def _cyclic_maps(self, f: List[int]):
        if f == [self.l] or f == []:
            return GF2(np.zeros(self.l, dtype=int))
        return GF2(np.isin(np.arange(self.l), f).astype(int))
    





if __name__ == '__main__':
    # Example with LP(A, 1+x)

    # m, n = 2, 3
    # A_lifts = {(0, 0): [1], (0, 1): [2], (0, 2): [4], (1, 0): [6], (1, 1):[5], (1, 2):[3]}
    # lift_mats = [((m, n), A_lifts)]

    m, n = 4, 5
    A_lifts = {(0,0):[0], (0,1):[0], (0,2):[0], (0,3):[0], (0,4):[0],
                (1,0):[0], (1,1):[1], (1,2):[11], (1,3):[8], (1,4):[9],
                (2,0):[0], (2,1):[4], (2,2):[5], (2,3):[6], (2,4):[10],
                (3,0):[0], (3,1):[10], (3,2):[6], (3,3):[2], (3,4):[12]}
    
    lift_mats = [((m, n), A_lifts)]

    QLP = LP(lift_mats, lift_size=13)

    # punctured code 
    # m,n = 4, 4
    # A_lifts = {(0,0):[0], (0,1):[0], (0,2):[0], (0,3):[0], 
    #             (1,0):[0], (1,1):[1], (1,2):[11], (1,3):[8], 
    #             (2,0):[0], (2,1):[4], (2,2):[5], (2,3):[6], 
    #             (3,0):[0], (3,1):[10], (3,2):[6], (3,3):[2]}
    # lift_mats = [((m, n), A_lifts)]

    
    # m, n = 4, 5
    # A_lifts = {(0,0):[0], (0,1):[0], (0,2):[0], (0,3):[0], (0,4):[0],
    #             (1,0):[0], (1,1):[1], (1,2):[11], (1,3):[8], (1,4):[9],
    #             (2,0):[0], (2,1):[4], (2,2):[5], (2,3):[6], (2,4):[10],
    #             (3,0):[0], (3,1):[10], (3,2):[6], (3,3):[2], (3,4):[12]}
    # lift_mats = [((m, n), A_lifts)]

    # m, n = 2, 3
    # A_lifts = {(0,0):[0], (0,1):[0], (0,2):[0],
    #             (1,0):[0], (1,1):[1], (1,2):[11],}
    # lift_mats = [((m, n), A_lifts)]

    # QLP = LP(lift_mats, lift_size=13)

    

    
    Hx, Hz = QLP._build_LP_parity_checks()

    print(f'Hx shape:', Hx.shape)
    print(f'Hz shape:', Hz.shape)
    print(f'checking the CSS condition: Hx @ Hz.T')
    print(np.all(Hx @ Hz.T == 0))
#    print(f'checking conjuagte tranpose before and after binarization: {np.all(Hz ==Hz2)}')
    
    print('-----estimating the distance for the lifted product code-----')
    Hx = np.array(Hx)
    Hz = np.array(Hz)
    print(f'the number of logical qubits: {code_rate(Hx, Hz)}')

    qlp_code = css_code(Hx, Hz)
    hx = qlp_code.hx
    lx = qlp_code.lx
    # num_logicals = L.shape[0]

    start_time = time.time()
    circuit_distance = DistanceEst_BPOSD(hx, lx, num_trials=20000)
    end_time = time.time()
    print('time elapsed:', end_time - start_time)
    print('X distance:', circuit_distance)

    hz = qlp_code.hz
    lz = qlp_code.lz
    # num_logicals = L.shape[0]

    start_time = time.time()
    circuit_distance = DistanceEst_BPOSD(hz, lz, num_trials=20000)
    end_time = time.time()
    print('time elapsed:', end_time - start_time)
    print('z distance:', circuit_distance)
   