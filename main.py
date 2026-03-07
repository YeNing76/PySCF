import numpy as np
import src.omega as om
import src.Pauli_LCU as pl
import src.geometry as gm
import src.hamiltonian as hm

def main():
    # 6, 14, 26, 50, 74, 86
    points, neighbors, vor = gm.hydrogen_adaptive_grid_lebedev(num_r=20, order = 50)

    # Check the Hamiltonian from first quantization
    sigma, volumes = gm.build_geometry(points, neighbors, vor)
    ham = hm.hamilton_14(points, neighbors, sigma, volumes)
    H = np.array(ham)
    vals, vecs = np.linalg.eigh(H)
    print(vals[0])

    # test pauli_lcu_one_body_from_omega and t_bar
    eta = 1 # for single electron system
    omega_pg = om.omega_matr_fwht(vor, points, neighbors, sigma, volumes)
    omega_pg = np.array(omega_pg)
    H_pauli = pl.pauli_lcu_one_body_from_omega(omega_pg, eta)
    H_from_pauli = H_pauli.to_matrix()

    num_qubits = 10
    dim = 2**num_qubits
    length = len(points)
    t_bar_small = om.t_bar_matr(points, neighbors, sigma, volumes)
    t_bar = [[0.0]*dim for _ in range(dim)]
    for i in range(length):
        for j in range(length):
            t_bar[i][j] = t_bar_small[i][j]
    err = np.linalg.norm(H_from_pauli - t_bar)
    print("err =", err)

if __name__=="__main__":
    main()