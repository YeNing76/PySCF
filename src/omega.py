# The following function is used to take the place of 
# omega_matr(vor, points, neighbors) so that the FWHT
# can be used to improve the effeciency
# 2026, 01, 24
def omega_matr_fwht(vor, points, neighbors):
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
