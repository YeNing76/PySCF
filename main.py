import src.omega as om
import src.pauli_lcu as pl

def main():
    # 6, 14, 26, 50, 74, 86
    points, neighbors, vor = hydrogen_adaptive_grid_lebedev(num_r=20, order = 50)