import numpy as np

def create_qubo_bpp(items, bin_capacity):
    n = len(items)  # Number of items
    m = n  # Assume we need at most n bins
    w = items  # Weights of items
    
    # Penalty coefficients (tune them for proper scaling)
    lambda_1 = 1.0  # Penalty for multiple assignments to bins
    lambda_2 = 1.0  # Penalty for exceeding bin capacity
    lambda_3 = 1.0  # Penalty for minimizing the number of bins
    
    # Initialize the QUBO matrix
    Q = np.zeros((n * m, n * m))
    
    # Constraint 1: Each item is assigned to exactly one bin
    for i in range(n):
        for j in range(m):
            Q[i * m + j, i * m + j] += lambda_1 * (1 - 2 * 1)  # (1 - sum(x_ij))^2 expanded
        
        for j1 in range(m):
            for j2 in range(m):
                if j1 != j2:
                    Q[i * m + j1, i * m + j2] += lambda_1 * 2  # Cross-term
    
    # Constraint 2: Total weight in each bin should not exceed capacity
    for j in range(m):
        for i1 in range(n):
            for i2 in range(n):
                Q[i1 * m + j, i2 * m + j] += lambda_2 * w[i1] * w[i2]  # Penalty term for bin capacity

    # Constraint 3: Minimize the number of bins used
    for j in range(m):
        for i in range(n):
            Q[i * m + j, i * m + j] += lambda_3  # Bin usage penalty
    
    return Q
'''
Testing the QUBO Function

We created small, medium, and large problem instances, each with a set of items(weight) and a bin capacity. Here is how it's structured
Small instances: 3-5 items , small bin capacity(e.g., 10-20 units)
Medium instance: 6-10 items, moderate bin capacirty(e.g 30-50 units)
Large instances: 15-20 items, larger bins capacity(e.g., 50-100 units)

'''

import numpy as np

# Example small, medium, and large instances
small_instance = [3, 4, 5]  # Item weights for the small instance
medium_instance = [10, 15, 7, 8, 9, 12]  # Item weights for the medium instance
large_instance = [20, 15, 10, 8, 5, 12, 18, 25, 7, 6, 17, 9, 14]  # Large instance weights

# Bin capacities
small_capacity = 10
medium_capacity = 30
large_capacity = 50

# Function to run QUBO test for each instance
def run_qubo_test(instance, capacity):
    print(f"\nTesting QUBO for instance with items: {instance} and bin capacity: {capacity}")
    Q = create_qubo_bpp(instance, capacity)
    print("QUBO matrix created:")
    print(Q)

# Run QUBO tests for small, medium, and large instances
run_qubo_test(small_instance, small_capacity)
run_qubo_test(medium_instance, medium_capacity)
run_qubo_test(large_instance, large_capacity)
