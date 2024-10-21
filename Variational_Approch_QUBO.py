'''
Quantum Variational Approach

Use a Quantum Variational approach(like VQE) to solve the QUBO.
Create multiple Ansatz for the tests
'''

import numpy as np
from dwave.system import DWaveSampler, EmbeddingComposite
from dimod import BinaryQuadraticModel
from dwave.samplers import SimulatedAnnealingSampler

# Define your QUBO matrix
Q = np.array([[1, -2, 0], 
              [-2, 4, -1], 
              [0, -1, 2]])

# Convert QUBO matrix into a dictionary form that D-Wave accepts
def convert_qubo_to_dict(Q):
    n = len(Q)
    qubo_dict = {}
    for i in range(n):
        for j in range(i, n):
            if Q[i, j] != 0:
                qubo_dict[(i, j)] = Q[i, j]
    return qubo_dict

# Convert the QUBO matrix to a dictionary
qubo_dict = convert_qubo_to_dict(Q)

# Set up the Binary Quadratic Model (BQM)
bqm = BinaryQuadraticModel.from_qubo(qubo_dict)

# Initialize the D-Wave Sampler
sampler = SimulatedAnnealingSampler()

# Solve the problem using the sampler
response = sampler.sample(bqm, num_reads=100)

# Print the results
print("Best solution:", response.first.sample)
print("Energy:", response.first.energy)


import pennylane as qml
import numpy as np
from pennylane import numpy as np

# Number of qubits based on the problem
num_qubits = 3

# Create a quantum device (simulator)
dev = qml.device("default.qubit", wires=num_qubits)

# QUBO matrix
Q = np.array([[1, -2, 0], 
              [-2, 4, -1], 
              [0, -1, 2]])

# Convert QUBO matrix into a Hamiltonian
def qubo_to_hamiltonian(Q):
    terms = []
    coeffs = []
    num_qubits = len(Q)
    
    for i in range(num_qubits):
        terms.append(qml.PauliZ(i))
        coeffs.append(Q[i, i])
        
        for j in range(i + 1, num_qubits):
            terms.append(qml.PauliZ(i) @ qml.PauliZ(j))
            coeffs.append(Q[i, j])
    
    H = qml.Hamiltonian(coeffs, terms)
    return H

H = qubo_to_hamiltonian(Q)

# Define various Ansatz circuits
def ansatz_circuit_1(params):
    for i in range(num_qubits):
        qml.Hadamard(wires=i)  # Initial state
    for i in range(num_qubits):
        qml.RY(params[i], wires=i)  # Apply rotations
    for i in range(num_qubits - 1):
        qml.CNOT(wires=[i, i + 1])  # Entangling gates

def ansatz_circuit_2(params):
    for i in range(num_qubits):
        qml.Hadamard(wires=i)  # Initial state
    for i in range(num_qubits):
        qml.RX(params[i], wires=i)  # Different rotation gates
    for i in range(num_qubits - 1):
        qml.CNOT(wires=[i, i + 1])  # Entangling gates

def ansatz_circuit_3(params):
    for i in range(num_qubits):
        qml.Hadamard(wires=i)  # Initial state
    for i in range(num_qubits):
        qml.RZ(params[i], wires=i)  # RZ rotations
    for (i, j) in np.ndindex(num_qubits):
        if i < j:
            qml.CNOT(wires=[i, j])  # CNOT for entangling

# Define the VQE function
def vqe(Q, ansatz, p):
    # Number of parameters based on ansatz
    params = np.random.uniform(0, 2 * np.pi, p)
    
    # Define the QNode
    @qml.qnode(dev)
    def circuit(params):
        ansatz(params)  # Apply the chosen ansatz
        return qml.expval(H)  # Return expectation value
    
    # Define the cost function
    def cost_function(params):
        return circuit(params)

    # Use PennyLane's optimizer (Gradient Descent)
    opt = qml.GradientDescentOptimizer(stepsize=0.1)

    # Perform the optimization loop
    steps = 100
    for step in range(steps):
        params = opt.step(cost_function, params)
        energy = cost_function(params)
        print(f"Ansatz: {ansatz.__name__}, Step {step + 1}: Energy = {energy}")

    # Final optimal parameters
    print(f"Optimal Parameters for {ansatz.__name__}:", params)
    return energy, params

# Number of layers (p) based on ansatz parameter count
p1 = num_qubits  # For ansatz 1
p2 = num_qubits  # For ansatz 2
p3 = num_qubits  # For ansatz 3

# Run VQE with different Ansatz
print("Running VQE with Ansatz 1:")
final_energy_1, optimal_params_1 = vqe(Q, ansatz_circuit_1, p1)

print("\nRunning VQE with Ansatz 2:")
final_energy_2, optimal_params_2 = vqe(Q, ansatz_circuit_2, p2)

print("\nRunning VQE with Ansatz 3:")
final_energy_3, optimal_params_3 = vqe(Q, ansatz_circuit_3, p3)

print(f"\nFinal Energies:\nAnsatz 1: {final_energy_1}\nAnsatz 2: {final_energy_2}\nAnsatz 3: {final_energy_3}")
