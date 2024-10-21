from qiskit import transpile, QuantumCircuit
from qiskit_aer import Aer
from qiskit.primitives import Sampler
from scipy.optimize import minimize
import numpy as np
from qiskit.circuit import Parameter

def create_qaoa_ansatz(num_qubits: int, p: int) -> tuple:
    """
    Creates a QAOA ansatz quantum circuit for solving QUBO problems.
    
    The circuit alternates between applying the cost Hamiltonian (related to the problem we want to solve)
    and the mixing Hamiltonian (to explore the solution space).

    Parameters:
    - num_qubits (int): Number of qubits (equal to the size of the problem)
    - p (int): Number of layers (or depth) in the QAOA ansatz. Higher values of p typically result in better approximations.

    Returns:
    - qc (QuantumCircuit): The quantum circuit that implements the QAOA ansatz.
    - gamma (list of Parameter): List of gamma parameters for the cost Hamiltonian layers.
    - beta (list of Parameter): List of beta parameters for the mixing Hamiltonian layers.
    """

    # Initialize the quantum circuit with num_qubits qubits
    qc = QuantumCircuit(num_qubits)

    # Create parameterized angles for gamma (for the cost Hamiltonian) and beta (for the mixing Hamiltonian)
    gamma = [Parameter(f'gamma_{i}') for i in range(p)]
    beta = [Parameter(f'beta_{i}') for i in range(p)]

    # Construct the QAOA circuit with alternating layers of the cost and mixing Hamiltonians
    for layer in range(p):
        # Apply the cost Hamiltonian (using RZZ gates to encode the problem's interactions)
        for i in range(num_qubits):
            for j in range(num_qubits):
                if i < j:
                    qc.rzz(2 * gamma[layer], i, j)  # RZZ gate for pairwise interactions

        # Apply the mixing Hamiltonian (using RX gates to explore solution space)
        for i in range(num_qubits):
            qc.rx(2 * beta[layer], i)  # RX gate for qubit rotation

    return qc, gamma, beta

def qaoa_solver(Q: np.ndarray, p: int) -> tuple:
    """
    Solves a QUBO problem using the Quantum Approximate Optimization Algorithm (QAOA).
    
    QAOA is a hybrid quantum-classical algorithm that alternates between quantum operations and classical optimization.
    
    Parameters:
    - Q (np.ndarray): QUBO matrix (must be a square matrix).
    - p (int): Number of layers in the QAOA ansatz.

    Returns:
    - optimal_params (np.ndarray): The optimized values for the gamma and beta parameters.
    - min_energy (float): The minimum energy (objective value) found.
    """
    
    # Validate the input matrix Q (must be square) and p (must be positive)
    if not isinstance(Q, np.ndarray) or Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be a square matrix")
    if not isinstance(p, int) or p <= 0:
        raise ValueError("p must be a positive integer")

    num_qubits = Q.shape[0]  # The number of qubits is the size of the QUBO problem

    # Create the QAOA ansatz circuit for the given number of qubits and layers
    qc, gamma, beta = create_qaoa_ansatz(num_qubits, p)

    def objective_function(params):
        """
        Objective function for classical optimization.

        The function assigns the current values of gamma and beta to the QAOA circuit, executes it,
        and measures the resulting energy (the expected value of the cost function).
        
        Parameters:
        - params (np.ndarray): A list of parameter values for gamma and beta.

        Returns:
        - energy (float): The average energy (cost function value) for the current parameters.
        """

        # Map the current parameters to the gamma and beta variables
        param_dict = {gamma[i]: params[i] for i in range(p)}
        param_dict.update({beta[i]: params[p + i] for i in range(p)})

        # Assign the parameters to the QAOA circuit
        bound_circ = qc.assign_parameters(param_dict, inplace=False)
        
        # Simulate the quantum circuit using a quantum simulator backend
        backend = Aer.get_backend('aer_simulator')
        bound_circ = transpile(bound_circ, backend)  # Transpile the circuit for the simulator
        sampler = Sampler(backend)  # Create a sampler to run the circuit
        
        # Execute the circuit and obtain the measurement results
        counts = sampler.run(bound_circ).result().get_counts()

        # Calculate the energy (cost function value) based on the measurement results
        energy = 0.0
        for state, count in counts.items():
            state_vector = [int(bit) for bit in state]
            # Sum over all pairwise interactions in the QUBO matrix
            energy += count * (-np.sum(Q[state_vector[i], state_vector[j]] 
                                       for i in range(num_qubits) 
                                       for j in range(num_qubits) if i != j))

        # Return the average energy (normalized by the number of measurements)
        return energy / sum(counts.values())

    # Initialize random parameters for gamma and beta
    initial_params = np.random.rand(2 * p)

    # Use a classical optimizer (COBYLA) to minimize the objective function
    result = minimize(objective_function, initial_params, method='COBYLA')

    # Retrieve the optimized parameters and the corresponding minimum energy
    optimal_params = result.x
    min_energy = result.fun
    
    return optimal_params, min_energy

# Example usage: Define a small QUBO matrix and solve it using QAOA with p=1 layer
Q = np.array([[1, -1, -1],
              [-1, 1, -1],
              [-1, -1, 1]])

optimal_params, min_energy = qaoa_solver(Q, p=1)
print("Optimal parameters:", optimal_params)
print("Minimum energy found:", min_energy)
