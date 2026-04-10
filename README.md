# Real-space Chemistry on Quantum Computers
This repository is for coding based on the paper https://arxiv.org/pdf/2507.20583v1. In this paper, a first-quantized, real-space formulations of quantum chemistry on quantum computers are studied. In this paper, Voronoi partition is employed instead of uniform grid. 

For this repository, it contains quantum chemistry/ quantum computing — specifically building a real-space molecular Hamiltonian on a Voronoi grid and then mapping it to qubits for simulation with Qiskit.

# Main Sections:
1. **Radial & Angular Grids (Section 1-2)**
   * becke_radial_grid: Generates a radially streched grid using the Becke mapping (logarithmic streching towward the nucleus).
   * Lebedev grids (lebedev_6, lebedev_14, )
