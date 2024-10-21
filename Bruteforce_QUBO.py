import numpy as np
import itertools

# Brute force solver for QUBO (Quadratic Unconstrained Binary Optimization)
def brute_force_qubo_solver(Q):
    """
    Solves the QUBO problem using a brute force approach by evaluating all possible binary combinations.
    
    Parameters:
    - Q (numpy.ndarray): A symmetric matrix representing the QUBO problem (n x n matrix for n binary variables).
    
    Returns:
    - best_solution (tuple): The binary configuration that minimizes the QUBO energy.
    - best_energy (float): The minimum energy value corresponding to the best solution.
    """
    n = len(Q)  # Number of variables (binary variables), Q is an n x n matrix
    best_energy = float('inf')  # Initialize with a very large value to find the minimum energy
    best_solution = None  # Placeholder to store the configuration that gives the minimum energy
    
    # Generate all possible binary configurations (there are 2^n configurations for n variables)
    for config in itertools.product([0, 1], repeat=n):
        # Calculate the energy for the current binary configuration
        energy = calculate_qubo_energy(Q, config)
        
        # Update the best solution if the current one has a lower energy
        if energy < best_energy:
            best_energy = energy
            best_solution = config

    return best_solution, best_energy

# Function to calculate the QUBO energy for a given binary configuration
def calculate_qubo_energy(Q, config):
    """
    Computes the QUBO energy for a given binary configuration.
    
    Parameters:
    - Q (numpy.ndarray): The QUBO matrix representing the quadratic coefficients.
    - config (tuple): A binary configuration (0s and 1s) representing a candidate solution.
    
    Returns:
    - energy (float): The computed energy value for the given configuration.
    """
    config = np.array(config)  # Convert the configuration (tuple) to a numpy array
    # Reshape the config to a column vector for matrix multiplication (n x 1 shape)
    config = config.reshape(-1, 1)
    # Calculate the QUBO energy using the formula: x^T * Q * x
    energy = np.dot(config.T, np.dot(Q, config))[0, 0]
    return energy

# Example QUBO matrix (you can replace this with your specific QUBO problem)
Q = np.array([[1, -2, 0],  # Q[i, j] represents the interaction between binary variables x_i and x_j
              [-2, 4, -1],  # A symmetric matrix is used to define the problem
              [0, -1, 2]])

# Solve the QUBO problem using brute force
solution, energy = brute_force_qubo_solver(Q)

# Output the best solution found and its corresponding minimum energy
print("Best solution:", solution)
print("Minimum energy:", energy)
