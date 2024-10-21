from docplex.mp.model import Model

def create_ilp_bpp(items, bin_capacity):
    """
    Creates an Integer Linear Programming (ILP) model for solving the Bin Packing Problem (BPP) using Docplex.
    
    The goal is to pack a set of items into the minimum number of bins without exceeding the bin capacity.

    Parameters:
    - items (list): A list of item weights that need to be packed.
    - bin_capacity (int): The maximum capacity that each bin can hold.

    Returns:
    - model (docplex.mp.model.Model): The ILP model representing the bin packing problem.
    """

    # Initialize the model
    model = Model(name='BinPacking')

    # Define the number of bins, which is at most equal to the number of items
    bins = range(len(items))  # We assume the worst-case scenario where each item gets its own bin

    # Define the binary decision variables:
    # x[i, j] = 1 if item i is placed in bin j, 0 otherwise
    # y[j] = 1 if bin j is used, 0 otherwise
    x = {(i, j): model.binary_var(name=f'x_{i}_{j}') for i in range(len(items)) for j in bins}
    y = {j: model.binary_var(name=f'y_{j}') for j in bins}

    # Constraint 1: Each item must be placed in exactly one bin
    for i in range(len(items)):
        model.add_constraint(sum(x[i, j] for j in bins) == 1, ctname=f'item_{i}_in_one_bin')

    # Constraint 2: The total weight of items in a bin must not exceed the bin's capacity
    for j in bins:
        model.add_constraint(sum(items[i] * x[i, j] for i in range(len(items))) <= bin_capacity * y[j], 
                             ctname=f'bin_capacity_{j}')

    # Objective: Minimize the number of bins used (i.e., minimize the sum of y[j])
    model.minimize(model.sum(y[j] for j in bins))

    return model
