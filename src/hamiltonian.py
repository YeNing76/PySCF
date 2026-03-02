import numpy as np
import math

"""
    Construct the symmetric Laplacian matrix following:
    (15)  L_mn = Ω_mn / (U_m * r_mn)
    (19)  L̃_mn = L_mn / sqrt(U_m U_n)
"""
def laplacian(points, neighbors, sigma, volumes):

    length = len(points)
    L = np.zeros((length, length))

    # --- Step 1: build the non-symmetric Laplacian (15) ---
    for m in range(length):
        for n in neighbors[m]:
            if n not in sigma[m]:
                continue

            dist = np.linalg.norm(points[m] - points[n])
            L[m, n] = sigma[m][n] / ( dist)

        # diagonal term
        L[m, m] = -np.sum(L[m, :])

    # --- Step 2: symmetric Laplacian (19) --- 
    L_sym = np.zeros_like(L)   
    for m in range(length):
        for n in range(length):
            if L[m, n] != 0:
                L_sym[m, n] = L[m, n] / np.sqrt(volumes[m] * volumes[n])

    return L_sym

# Coulombic electron-nuclei attraction
# For hydrogen atom, we set the nuclear at the origin
# Therefore, \vec{R}_{\alpha}=\vec{0} and Z_{\alpha}=1
def coulombic_potential(points):
    length = len(points)
    num_bits = math.ceil(math.log2(length))
    dim = 1 << num_bits
    ele_nu = [0.0] * dim
    ele_ele = [[0.0]*dim for _ in range(dim)]

    Z_alpha = 1.0
    R_alpha = np.array([0.0, 0.0, 0.0])

    # electron–nucleus attraction
    for m in range(length):
        dist_nucl = np.linalg.norm(points[m] - R_alpha)
        if dist_nucl == 0:
            ele_nu[m] = np.inf
        else:
            ele_nu[m] = -Z_alpha / dist_nucl   # negative sign is important

    # electron–electron repulsion
    for m in range(length):
        for p in range(m, length):
            if m == p:
                ele_ele[m][p] = 0.0
            else:
                dist = np.linalg.norm(points[m] - points[p])
                ele_ele[m][p] = 1.0 / dist
                ele_ele[p][m] = ele_ele[m][p]  # symmetry

    return ele_nu, ele_ele

"""
    Hydrogen Hamiltonian via eq. (14)/(20):
    H = -1/2 * L̃ + V_nuc
"""
def hamilton_14(points, neighbors, sigma, volumes):
    length = len(points)

    # 对称化 Laplacian（公式 (19) 的结果)
    L_sym = laplacian(points, neighbors, sigma, volumes)

    # 电子–核吸引势 V_nuc(r_m) = -1 / |r_m|
    ele_nu, _ = coulombic_potential(points) # 第二个返回值暂时不用

    # 初始化 Hamiltonian
    H = np.zeros((length, length))

    # 动能部分：-1/2 * L̃
    H -= 0.5 * L_sym

    # 加上势能对角项
    for m in range(length):
        H[m, m] += ele_nu[m]
    
    return H