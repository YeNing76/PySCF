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

if __name__=="__main__":
    main()