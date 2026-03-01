import math
import qiskit
import itertools
import numpy as np
from scipy.spatial import Voronoi
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import eigvalsh
from collections import defaultdict
from itertools import product

#from qiskit import QuantumCircuit, execute
from qiskit_aer import Aer
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector
from qiskit.circuit.library import QFT
from qiskit.circuit.library import PauliEvolutionGate
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP
from qiskit.circuit.library import TwoLocal
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit.primitives import StatevectorEstimator as Estimator
#from qiskit.opflow import PauliTrotterEvolution