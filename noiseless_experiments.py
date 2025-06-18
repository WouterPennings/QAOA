# Quantumsim imports
from quantumsim_performante.quantumsim_performante import Circuit
from quantumsim_performante.dirac import Dirac

# Importing minimize functions
from scipy.optimize import minimize as minimize_cobyla
from spsa import minimize as minimize_spsa

# Additional imports
import numpy as np
import matplotlib.pyplot as plt

# Constants
NUM_OPTIMIZATION_ITERATIONS_SPSA = 10
DATA_PATH = "data_noiseless/data.toml"
USE_GPU = False
USE_LAZY = True
USE_CACHE = False


# Graph data
graphs_low = [
    [[0, 1, 2, 3], [(0, 1), (1, 2), (1, 3)]], 
    [[0, 1, 2, 3, 4], [(0, 3), (1, 2), (1, 4), (2, 3), (2, 4)]], 
    [[0, 1, 2, 3, 4, 5], [(0, 2), (1, 3), (1, 4), (2, 4), (2, 5), (3, 5)]], 
    [[0, 1, 2, 3, 4, 5, 6], [(0, 4), (0, 6), (1, 3), (2, 5), (3, 6), (5, 6)]], 
    [[0, 1, 2, 3, 4, 5, 6, 7], [(0, 1), (0, 2), (0, 3), (0, 4), (0, 7), (1, 2), (1, 7), (2, 3), (2, 5), (3, 4), (4, 6), (5, 7)]], 
    [[0, 1, 2, 3, 4, 5, 6, 7, 8], [(0, 2), (0, 3), (0, 4), (0, 6), (0, 7), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (2, 4), (3, 4), (3, 5), (4, 6), (5, 7), (6, 8)]], 
    [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [(0, 6), (0, 9), (1, 2), (1, 7), (2, 3), (2, 5), (2, 9), (3, 5), (3, 6), (3, 8), (3, 9), (4, 5), (4, 6), (6, 8), (7, 8), (8, 9)]],
    [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [(0, 4), (0, 6), (0, 8), (0, 9), (0, 10), (0, 12), (0, 13), (0, 14), (1, 2), (1, 4), (1, 5), (1, 7), (1, 8), (1, 9), (1, 10), (1, 13), (2, 3), (2, 5), (2, 10), (2, 12), (2, 13), (2, 14), (3, 5), (3, 8), (3, 11), (3, 12), (4, 5), (4, 6), (4, 7), (4, 11), (4, 14), (5, 9), (5, 10), (5, 13), (6, 7), (6, 10), (6, 11), (6, 13), (7, 14), (8, 10), (8, 11), (8, 13), (8, 14), (9, 10), (10, 13), (10, 14), (11, 13), (12, 13), (13, 14)]],
    # [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], [(0, 1), (0, 6), (0, 10), (0, 12), (0, 13), (0, 15), (0, 16), (0, 19), (1, 7), (1, 8), (1, 12), (1, 16), (2, 6), (2, 10), (2, 13), (2, 16), (2, 17), (3, 4), (3, 5), (3, 10), (3, 19), (4, 5), (4, 6), (4, 7), (4, 8), (4, 10), (4, 16), (4, 18), (5, 7), (5, 9), (5, 10), (5, 11), (5, 13), (5, 14), (5, 15), (5, 16), (6, 7), (6, 8), (6, 12), (6, 13), (6, 15), (6, 17), (6, 19), (7, 14), (7, 17), (8, 10), (8, 11), (8, 13), (8, 15), (8, 17), (10, 11), (10, 12), (10, 15), (10, 19), (11, 12), (11, 13), (11, 14), (11, 16), (11, 18), (11, 19), (12, 15), (12, 16), (13, 15), (13, 16), (14, 15), (14, 19), (15, 17), (16, 17), (16, 18), (16, 19), (17, 18), (18, 19)]]
]

graphs_high = [
    [[0, 1, 2, 3], [(0, 1), (0, 2), (0, 3), (1, 3), (2, 3)]], 
    [[0, 1, 2, 3, 4], [(0, 1), (0, 3), (0, 4), (1, 4), (2, 3), (2, 4), (3, 4)]], 
    [[0, 1, 2, 3, 4, 5], [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5)]], 
    [[0, 1, 2, 3, 4, 5, 6], [(0, 1), (0, 2), (0, 4), (0, 5), (0, 6), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (2, 3), (2, 5), (2, 6), (3, 4), (3, 5), (3, 6), (4, 6), (5, 6)]], 
    [[0, 1, 2, 3, 4, 5, 6, 7], [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (3, 4), (3, 7), (4, 5), (5, 7), (6, 7)]], 
    [[0, 1, 2, 3, 4, 5, 6, 7, 8], [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (1, 2), (1, 3), (1, 4), (1, 5), (1, 7), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (3, 5), (3, 6), (3, 7), (3, 8), (4, 5), (4, 6), (4, 7), (4, 8), (5, 6), (5, 7), (6, 7), (6, 8), (7, 8)]], 
    [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [(0, 1), (0, 2), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (5, 6), (5, 7), (5, 8), (5, 9), (6, 7), (6, 9), (7, 8), (7, 9), (8, 9)]], 
    [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [(0, 1), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 13), (0, 14), (1, 3), (1, 5), (1, 6), (1, 8), (1, 9), (1, 10), (1, 12), (1, 13), (1, 14), (2, 5), (2, 6), (2, 7), (2, 8), (2, 10), (2, 12), (2, 13), (2, 14), (3, 4), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (3, 12), (3, 13), (3, 14), (4, 5), (4, 6), (4, 8), (4, 9), (4, 10), (4, 11), (4, 12), (4, 13), (4, 14), (5, 6), (5, 7), (5, 9), (5, 10), (5, 11), (5, 12), (5, 13), (6, 7), (6, 8), (6, 9), (6, 10), (6, 11), (7, 8), (7, 9), (7, 10), (7, 11), (7, 12), (7, 13), (7, 14), (8, 9), (8, 10), (8, 11), (8, 13), (9, 10), (9, 11), (9, 12), (9, 13), (9, 14), (10, 11), (10, 12), (10, 13), (10, 14), (11, 13), (11, 14), (13, 14)]],
    # [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 13), (0, 14), (0, 15), (0, 17), (0, 18), (0, 19), (1, 2), (1, 3), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (1, 15), (1, 16), (1, 17), (1, 18), (2, 3), (2, 4), (2, 7), (2, 10), (2, 11), (2, 13), (2, 14), (2, 15), (2, 16), (2, 18), (2, 19), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 11), (3, 12), (3, 14), (3, 16), (3, 18), (4, 5), (4, 7), (4, 8), (4, 9), (4, 11), (4, 12), (4, 13), (4, 14), (4, 16), (4, 17), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10), (5, 11), (5, 12), (5, 13), (5, 14), (5, 15), (5, 16), (5, 17), (5, 18), (6, 7), (6, 8), (6, 10), (6, 11), (6, 12), (6, 13), (6, 14), (6, 16), (6, 17), (6, 18), (6, 19), (7, 8), (7, 9), (7, 10), (7, 11), (7, 13), (7, 14), (7, 16), (7, 17), (7, 18), (7, 19), (8, 9), (8, 10), (8, 11), (8, 12), (8, 13), (8, 14), (8, 15), (8, 16), (8, 18), (8, 19), (9, 11), (9, 15), (9, 16), (9, 17), (9, 18), (9, 19), (10, 11), (10, 14), (10, 16), (11, 13), (11, 14), (11, 15), (11, 16), (11, 17), (11, 18), (11, 19), (12, 13), (12, 14), (12, 17), (12, 18), (13, 16), (13, 17), (13, 19), (14, 16), (14, 17), (14, 18), (14, 19), (15, 16), (15, 17), (15, 18), (15, 19), (16, 17), (16, 18), (16, 19), (17, 19), (18, 19)]]
]

graphs = {
    "low": graphs_low,
    "high": graphs_high
}

def maxcut_bruteforce(nodes, edges):
    nr_nodes = len(nodes)
    nr_partitions = 2**nr_nodes
    
    all_partitions_data = []
    max_cut_size = -1  # Initialize with a value lower than any possible cut size
    max_cut_index = -1

    # We iterate from 0 to nr_partitions - 1 to include all possible partitions.
    # The problem statement for max-cut usually includes trivial partitions (all nodes in one set).
    # If you specifically want to exclude trivial partitions (all nodes in one set),
    # you can iterate from 1 to nr_partitions - 2 as in your original code.
    # For a comprehensive list, iterating from 0 to nr_partitions - 1 is generally better.
    for i in range(nr_partitions):
        binary_digits = format(i, f'0{nr_nodes}b')
        partition = {}
        for n_idx, n in enumerate(nodes):
            partition[n] = 0 if binary_digits[n_idx] == '0' else 1 # Corrected index access

        current_cut_size = cut_size(edges, partition)
        
        # Store the current partition and its cut size
        all_partitions_data.append((partition.copy(), current_cut_size))
        
        # Check if this is the new maximum cut
        if current_cut_size > max_cut_size:
            max_cut_size = current_cut_size
            max_cut_index = len(all_partitions_data) - 1 # Store the index of the current partition
            
    return all_partitions_data, max_cut_index

def cut_size(edges, partition):
    return sum(1 for u, v in edges if partition[u] != partition[v])

def compute_average_cut_size_from_probabilities(nodes: list, edges: list, state_vector: list, top_n:int=-1) -> float:
    state_vector = np.asarray(state_vector).flatten()

    # Determine the number of qubits
    nr_qubits = int(np.log2(len(state_vector)))

    # Determine the probabilities
    probabilities = np.square(np.abs(state_vector))

    # Normalize probabilities
    probabilities /= np.sum(probabilities)

    # If top_n is not specified, use all states
    if top_n == -1:
        top_n = len(probabilities)

    # Get indices of the top_n most probable states
    top_indices = np.argsort(probabilities)[-top_n:]

    # Compute average cut size from the top_n probabilities
    sum_cut_size = 0
    for i in top_indices:
        string = Dirac.state_as_string(i, nr_qubits)
        bit_string = string[1:-1]  # Assuming this strips off unwanted characters
        partition = {node: int(bit_string[node]) for node in nodes}
        partition_cut_size = cut_size(edges, partition)
        sum_cut_size += partition_cut_size * probabilities[i]

    # Normalize the sum by the sum of the selected probabilities
    average_cut_size = sum_cut_size / np.sum(probabilities[top_indices])
    return average_cut_size

def compute_max_cut_from_probabilities(nodes: list, edges: list, state_vector: list) -> int:
    state_vector = np.asarray(state_vector).flatten()

    # Determine the number of qubits
    nr_qubits = int(np.log2(len(state_vector)))

    # Determine the probabilities
    probabilities = np.square(np.abs(state_vector))

    # Normalize probabilities
    probabilities /= np.sum(probabilities)

    # Find the index of the state with the highest probability
    max_prob_index = np.argmax(probabilities)

    # Compute the cut size for the state with the highest probability
    max_cut_string = Dirac.state_as_string(max_prob_index, nr_qubits)
    max_cut_bit_string = max_cut_string[1:-1]  # Strip off unwanted characters if necessary
    partition = {node: int(max_cut_bit_string[node]) for node in nodes}
    max_cut_size = cut_size(edges, partition)

    return max_cut_size

def plot_probabilities(state_vector, file_name:str):
    state_vector = state_vector.flatten()

    nr_qubits = int(np.log2(len(state_vector)))
    probabilities = np.abs(state_vector) ** 2
    probabilities /= np.sum(probabilities)

    labels = [Dirac.state_as_string(i, nr_qubits) for i in range(1 << nr_qubits)]

    plt.clf()
    plt.bar(labels, probabilities)
    plt.xticks(rotation='vertical') if nr_qubits > 3 else None
    plt.xlabel('Classical states')
    plt.ylabel('Probability')
    plt.title('Probabilities')
    plt.tight_layout()

    plt.savefig("data_noiseless/"+file_name)

def qaoa_circuit(gamma:list[float], beta:list[float], nodes:list, edges:list, p:int) -> Circuit:
    # Consistency check
    if len(gamma) != p or len(beta) != p:
        raise ValueError(f"Lists gamma and beta should be of length p = {p}")
    
    # Create circuit witn n qubits, where n is the number of nodes
    n = len(nodes)
    circuit = Circuit(n, use_lazy=USE_LAZY, use_GPU=USE_GPU, use_cache=USE_CACHE)
    
    # Initialize circuit by applying the Hadamard gate to all qubits
    for q in range(n):
        circuit.hadamard(q)

    # Construct p alternating cost and mixer layers
    for i in range(p):
    
        # Construct cost layer with parameter gamma[i]
        for edge in edges:
            circuit.cnot(edge[0], edge[1])
            circuit.rotate_z(2 * gamma[i], edge[1])
            circuit.cnot(edge[0], edge[1])

        # Construct mixer layer with parameter beta[i]
        for q in range(n):
            circuit.rotate_x(2 * beta[i], q)
    
    #return circuit
    return circuit

# ======================================================
# ======= Running the experiments ======================
# ======================================================

for graph_connectivity, graph_list in graphs.items():
    for (nodes, edges) in graph_list:
        # Calculating maxcut using bruteforce technique
        all_partitions_data, max_cut_index = maxcut_bruteforce(nodes, edges)
        optimal_cut_size = all_partitions_data[max_cut_index][1]

        # Building and executing QAOA circuit with 0 layers
        circuit = qaoa_circuit([], [], nodes, edges, 0)
        circuit.execute()
        statevector = circuit.state_vector.get_quantum_state()

        # Measuring circuit and generating data and graphs
        average_cut_size_random = compute_average_cut_size_from_probabilities(nodes, edges, statevector)

        # Optimizing parameteres (gamma and beta)
        p = 5
        gamma_guess: list[float] = [0.5] * p
        beta_guess: list[float] = [0.5] * p
        optimal_average_cut_size: float = 0
        optimal_parameters: list[float] = []
        optimal_statevector = []

        # Defining function to find gamma and beta
        def OptimizeParameters(parameters):
            global optimal_average_cut_size, optimal_parameters, optimal_statevector

            gamma = parameters[:p]
            beta = parameters[p:]
            circuit = qaoa_circuit(gamma, beta, nodes, edges, p)
            circuit.execute()
            statevector = circuit.state_vector.get_quantum_state()
            average_cut_size = compute_average_cut_size_from_probabilities(nodes, edges, statevector)

            if average_cut_size > optimal_average_cut_size:
                optimal_average_cut_size = average_cut_size
                optimal_parameters = parameters
                optimal_statevector = statevector

            return -average_cut_size  
        
        # Running SPSA optimization function to find gamma and beta
        minimize_spsa(OptimizeParameters, gamma_guess + beta_guess, iterations=NUM_OPTIMIZATION_ITERATIONS_SPSA, lr=0.2, lr_decay=0.602, px=0.2)
        gamma = optimal_parameters[:p]
        beta = optimal_parameters[p:]

        # Finding average cut size for top N of probabilities
        top_n = [10, 5, 3]
        average_cut_size_top_n = []
        for n in top_n:
            average_cut_size_top_n.append(compute_average_cut_size_from_probabilities(nodes, edges, optimal_statevector, top_n=n))

        # Finding actual maxcut based on probabilities
        maxcut = compute_max_cut_from_probabilities(nodes, edges, optimal_statevector)

        # Generating graph from probabilities
        file_name = f"n_{len(nodes)}_{graph_connectivity}.png"
        # plot_probabilities(optimal_statevector, file_name)

        with open(DATA_PATH, 'a') as file:
            file.writelines([
                f"\n[{file_name}]", 
                f"\navg_random = {average_cut_size_random}",
                f"\navg = {optimal_average_cut_size}", 
                f"\navg_top_{top_n[0]} = {average_cut_size_top_n[0]}",
                f"\navg_top_{top_n[1]} = {average_cut_size_top_n[1]}",  
                f"\navg_top_{top_n[2]} = {average_cut_size_top_n[2]}",  
                f"\nmaxcut_qaoa = {maxcut}",  
                f"\nmaxcut_bruteforce = {optimal_cut_size}", 
                f"\ngamma = {list(gamma)}", 
                f"\nbeta = {list(beta)}", 
                f"\np = {p}", 
                "\n"
                ])
