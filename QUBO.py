import numpy as np

def create_qubo_bpp(items, bin_capacity):
    """
    Creates a QUBO matrix for solving the Bin Packing Problem (BPP).
    
    The Bin Packing Problem aims to assign a set of items with given weights into a minimum number of bins,
    without exceeding the bin capacity.

    Parameters:
    - items (list or array): A list of item weights to be packed into bins.
    - bin_capacity (int): The capacity of each bin.

    Returns:
    - Q (np.ndarray): The QUBO matrix for the BPP.
    """

    n = len(items)  # Number of items to be packed
    m = n  # Assume we need at most n bins (worst case scenario: each item in a separate bin)
    w = items  # List of item weights
    
    # Penalty coefficients to control the weight of different constraints
    lambda_1 = 1.0  # Penalty for multiple assignments of the same item to different bins
    lambda_2 = 1.0  # Penalty for exceeding bin capacity
    lambda_3 = 1.0  # Penalty to minimize the number of bins used
    
    # Initialize the QUBO matrix with zeros (size n*m by n*m)
    Q = np.zeros((n * m, n * m))
    
    # Constraint 1: Each item must be assigned to exactly one bin
    for i in range(n):
        for j in range(m):
            # Diagonal terms: Penalize each item for not being placed in exactly one bin
            Q[i * m + j, i * m + j] += lambda_1 * (1 - 2 * 1)  # (1 - sum(x_ij))^2 expanded
        
        for j1 in range(m):
            for j2 in range(m):
                if j1 != j2:
                    # Cross-terms: Penalize placing the same item in more than one bin
                    Q[i * m + j1, i * m + j2] += lambda_1 * 2
    
    # Constraint 2: Total weight in each bin should not exceed the bin's capacity
    for j in range(m):
        for i1 in range(n):
            for i2 in range(n):
                # Penalize if the total weight of items in bin j exceeds the bin's capacity
                Q[i1 * m + j, i2 * m + j] += lambda_2 * w[i1] * w[i2]

    # Constraint 3: Minimize the number of bins used
    for j in range(m):
        for i in range(n):
            # Penalize the use of more bins to encourage bin minimization
            Q[i * m + j, i * m + j] += lambda_3
    
    return Q

# Testing the QUBO function

"""
We will test the QUBO function using small, medium, and large instances of the bin packing problem:
- Small instance: 3-5 items with a small bin capacity (e.g., 10-20 units)
- Medium instance: 6-10 items with a moderate bin capacity (e.g., 30-50 units)
- Large instance: 15-20 items with a large bin capacity (e.g., 50-100 units)
"""

# Example small, medium, and large instances
small_instance = [3, 4, 5]  # Item weights for the small instance
medium_instance = [10, 15, 7, 8, 9, 12]  # Item weights for the medium instance
large_instance = [20, 15, 10, 8, 5, 12, 18, 25, 7, 6, 17, 9, 14]  # Large instance weights

# Bin capacities for each test instance
small_capacity = 10
medium_capacity = 30
large_capacity = 50

# Function to run QUBO test for each problem instance
def run_qubo_test(instance, capacity):
    """
    Runs the QUBO test for a given instance and bin capacity.
    
    Parameters:
    - instance (list): List of item weights for the bin packing problem.
    - capacity (int): The capacity of the bins.
    """
    
    print(f"\nTesting QUBO for instance with items: {instance} and bin capacity: {capacity}")
    
    # Generate the QUBO matrix using the create_qubo_bpp function
    Q = create_qubo_bpp(instance, capacity)
    
    # Output the resulting QUBO matrix
    print("QUBO matrix created:")
    print(Q)

# Run QUBO tests for small, medium, and large problem instances
run_qubo_test(small_instance, small_capacity)
run_qubo_test(medium_instance, medium_capacity)
run_qubo_test(large_instance, large_capacity)
