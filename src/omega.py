# The following function is used to take the place of 
# omega_matr(vor, points, neighbors) so that the FWHT
# can be used to improve the effeciency
# 2026, 01, 24
import math
import numpy as np
from src.hamiltonian import laplacian, coulombic_potential

def omega_matr_fwht(vor, points, neighbors, sigma, volumes):
    length = len(points)
    num_bits = math.ceil(math.log2(length))
    dim = 1 << num_bits

    # 1. 构造扩展后的 t_bar
    t_bar_small = t_bar_matr(points, neighbors, sigma, volumes)
    t_bar = [[0.0]*dim for _ in range(dim)]
    for i in range(length):
        for j in range(length):
            t_bar[i][j] = t_bar_small[i][j]

    # 2. 构造 omega
    omega = [[0.0]*dim for _ in range(dim)]

    # 对每个 m 做一次 1D FWHT
    for m in range(dim):
        # Step 1: 构造 f_m(x)
        a = [0.0]*dim
        for x in range(dim):
            label = m ^ x
            a[x] = t_bar[label][x]

        # Step 2: 对 a 做 FWHT
        # fwht_inplace(a)
        # a = np.array(a, dtype=float)
        fwht(a)

        # Step 3: 写入 omega[m][n]
        for n in range(dim):
            omega[m][n] = a[n] / dim

    return omega

# The following code is a new version for the eqn. (21) 
# 2026-02-20
def t_bar_matr(points, neighbors, sigma, volumes):
    # Laplacian (already symmetric)
    L_sym = laplacian(points, neighbors, sigma, volumes)

    # electron–nucleus attraction
    ele_nu, _ = coulombic_potential(points)

    length = len(points)
    Tbar = np.zeros((length, length))

    for m in range(length):
        for n in range(length):
            Tbar[m, n] = -0.5 * L_sym[m, n]

        # add nuclear attraction to diagonal
        Tbar[m, m] += ele_nu[m] 

    return Tbar   


"""In-place Fast Walsh–Hadamard Transform on last axis."""
def fwht(a):
    h = 1
    n = len(a)
    while h < n:
        step = h * 2
        for i in range(0, n, step):
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        h *= 2