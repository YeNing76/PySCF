import math
import numpy as np
import qiskit
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector

# This function is to realize bitwise dot
# 2025-12-29
def bitwise_dot(m, n, num_bits):
    """Return the bitwise dot product (mod 2) of m and n."""
    s = 0
    for b in range(num_bits):
        s ^= ((m >> b) & 1) & ((n >> b) & 1)
    return s


# This function is for bitwise xor
# 2025-12-29
def bitwise_xor(m, n, num_bits):
    """Return the bitwise XOR of m and n, restricted to num_bits bits."""
    mask = (1 << num_bits) - 1
    return (m ^ n) & mask

def int_to_bits(x, num_bits):
    return [(x >> b) & 1 for b in range(num_bits)]


"""
    对应文章1 式 (22) / 文章3 式 (II.10) 的一体项 Pauli LCU 构造：
    
    输入：
        omega : 形状为 (D, D) 的复数矩阵，对应 ω_{pq}
        eta   : 电子数（文章1 的 η，文章3 的 N）
        tol   : 忽略绝对值小于 tol 的系数
    
    输出：
        SparsePauliOp，作用在 eta * log2(D) 个 qubit 上
"""
def pauli_lcu_one_body_from_omega(omega, eta, tol=1e-12):
    D = omega.shape[0]
    assert omega.shape == (D, D)
    M = int(np.log2(D))
    assert 2**M == D, "D 必须是 2 的整数次幂"

    n_qubits = eta * M

    paulis = []
    coeffs = []

    for p in range(D):
        for q in range(D):
            coef_pq = omega[p, q]
            if abs(coef_pq) < tol:
                continue

            # p, q 的二进制展开（低位在右）
            p_bits = [(p >> k) & 1 for k in range(M)]
            q_bits = [(q >> k) & 1 for k in range(M)]

            # 对每个电子 i，构造对应的 Pauli 串
            for i in range(eta):
                coef = coef_pq
                chars = ["I"] * n_qubits

                # 该电子对应的 qubit 区间：[i*M, (i+1)*M)
                for k in range(M):
                    z = q_bits[k]
                    x = p_bits[k]
                    qubit = i * M + k

                    if z == 0 and x == 0:
                        # I
                        continue
                    elif z == 1 and x == 0:
                        chars[qubit] = "Z"
                    elif z == 0 and x == 1:
                        chars[qubit] = "X"
                    else:
                        # z=1, x=1 → Y，带 i 相位
                        chars[qubit] = "Y"
                        coef *= 1j

                # Qiskit 使用从左到右高位到低位，这里反转一下  
                pauli_str = "".join(chars[::-1])  
                if abs(coef) > tol:
                    paulis.append(pauli_str)
                    coeffs.append(coef)
                    
    return SparsePauliOp(paulis, coeffs)

def potential_pauli_from_diag(V_diag):
    length = len(V_diag)
    num_bits = math.ceil(math.log2(length))
    dim = 1 << num_bits

    # padding
    V_pad = np.zeros(dim)
    V_pad[:length] = V_diag

    pauli_dict = {}

    for i in range(dim):
        coeff = V_pad[i]
        if coeff == 0.0:
            continue

        # 初始只有一个空字符串
        paulis = {'': 1.0}

        # 逐比特扩展
        for k in range(num_bits):
            bit = (i >> k) & 1
            sign = -1.0 if bit == 1 else 1.0

            new_paulis = {}
            for pstr, c in paulis.items():
                # I 分支
                p_I = pstr + 'I'
                new_paulis[p_I] = new_paulis.get(p_I, 0.0) + c * 0.5

                # Z 分支
                p_Z = pstr + 'Z'
                new_paulis[p_Z] = new_paulis.get(p_Z, 0.0) + c * 0.5 * sign

            paulis = new_paulis

        # 累加到全局字典   
        for pstr, c in paulis.items():
            pauli_dict[pstr] = pauli_dict.get(pstr, 0.0) + coeff * c

    labels = list(pauli_dict.keys())   
    coeffs = np.array([pauli_dict[l] for l in labels], dtype=complex)    
    return SparsePauliOp(labels, coeffs)
