from quantumsim.quantumsim import Circuit, Dirac
import networkx as nx
from spsa import minimize as minimize_spsa
import matplotlib.pyplot as plt
import numpy as np

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

def compute_average_cut_size_from_probabilities(nodes:list, edges:list, state_vectors:list, top_n:int=-1) -> float:
    """
    Compute average cut size from list of state vectors for noisy QAOA circuits.

    Parameters:
    nodes         : list of nodes
    edges         : list of edges
    state_vectors : list of state vectors

    Returns:
    average cut size
    """
    # Determine the number of qubits
    nr_qubits = int(np.log2(len(state_vectors[0])))

    # Determine the probabilities
    probalities = np.square(np.abs(state_vectors))
    
    # For a noisy circuit, the sum of probabilities may not be equal to one
    probalities = [prob / np.sum(prob) for prob in probalities]

    # Compute mean and standard deviation for each index
    mean_probalities = np.mean(probalities, axis=0).flatten()

    # If top_n is not specified, use all states
    if top_n == -1:
        top_n = len(mean_probalities)

    # Get indices of the top_n most probable states
    top_indices = np.argsort(mean_probalities)[-top_n:]

    # Compute average cut size from probabilities
    sum_cut_size = 0
    for i in range(2**nr_qubits):
        string = Dirac.state_as_string(i, nr_qubits)
        bit_string = string[1:-1]
        partition = {node: bit_string[node] for node in nodes}
        partition_cut_size = cut_size(edges, partition)
        sum_cut_size += partition_cut_size*mean_probalities[i]

    average_cut_size = sum_cut_size/np.sum(mean_probalities[top_indices])
    return average_cut_size

def compute_max_cut_from_probabilities(nodes: list, edges: list, state_vector: list) -> int:
    # Determine the number of qubits
    nr_qubits = int(np.log2(len(state_vector)))

    # Determine the probabilities
    probabilities = np.square(np.abs(state_vector))

    # Normalize probabilities
    probabilities /= np.sum(probabilities)

    # Compute mean and standard deviation for each index
    mean_probalities = np.mean(probabilities, axis=0).flatten()

    # Find the index of the state with the highest probability
    max_prob_index = np.argmax(mean_probalities)

    # Compute the cut size for the state with the highest probability
    max_cut_string = Dirac.state_as_string(max_prob_index, nr_qubits)
    max_cut_bit_string = max_cut_string[1:-1]  # Strip off unwanted characters if necessary
    partition = {node: int(max_cut_bit_string[node]) for node in nodes}
    max_cut_size = cut_size(edges, partition)

    return max_cut_size

def plot_probability_distribution_noise(state_vectors, file_name, noise_factor=1):
    """
    Create a bar plot showing average and standard deviation of 
    probabilities of occurrences of classical states.

    Parameters:
    state_vectors : list of state vectors. 
    
    A state vector is an array of 2^N complex numbers representing the quantum state of a circuit of size N
    """

    # Determine the number of qubits
    nr_qubits = int(np.log2(len(state_vectors[0])))

    # Determine the probabilities
    probalities = np.square(np.abs(state_vectors))
    
    # For a noisy circuit, the sum of probabilities may not be equal to one
    probalities = [prob / np.sum(prob) for prob in probalities]

    # Compute mean and standard deviation for each index
    mean_probalities = np.mean(probalities, axis=0).flatten()
    std_probabilities = np.std(probalities, axis=0).flatten()

    # Create the labels for the x-axis
    labels = [Dirac.state_as_string(i, nr_qubits) for i in range(2**nr_qubits)]

    # Clear previous plot  
    plt.clf()
    # Show the distribution of probabilities using a bar plot
    plt.bar(labels, mean_probalities, yerr=std_probabilities, capsize=3)
    if nr_qubits > 3:
        plt.xticks(rotation='vertical')
    plt.xlabel('Classical states')
    plt.ylabel('Probability')
    plt.title(f'Mean and stadard deviation of probabilities (noise_factor={noise_factor})')
    
    plt.tight_layout()
    plt.savefig(file_name)

def execute_circuit(circuit:Circuit, nr_executions=100):
    result = []
    for i in range(nr_executions):
        circuit.execute()
        result.append(circuit.state_vector.get_quantum_state())
    return result

def qaoa_circuit(gamma:list[float], beta:list[float], nodes:list, edges:list, p:int, noise_factor:float) -> Circuit:
    """
    Creates a quantum circuit of p layers for the Quantum Approximate Optimiziation Algorithm

    Parameters:
    gamma        : list of length p containing values for gamma, 0 < gamma < pi
    beta         : list of length p containing values for beta, 0 < beta < pi
    nodes        : list of nodes 
    edges        : list of edges
    p            : number of layers
    noise_factor : indicate amount of noise

    Returns:
    QAOA circuit with p layers
    """

    # Consistency check
    if len(gamma) != p or len(beta) != p:
        raise ValueError(f"Lists gamma and beta should be of length p = {p}")
    
    # Create circuit witn n qubits, where n is the number of nodes
    n = len(nodes)
    circuit = Circuit(n, save_instructions=True, noise_factor=noise_factor)
    
    # Initialize circuit by applying the Hadamard gate to all qubits
    for q in range(n):
        #circuit.hadamard(q)
        circuit.noisy_hadamard(q)

    # Construct p alternating cost and mixer layers
    for i in range(p):
    
        # Construct cost layer with parameter gamma[i]
        for edge in edges:
            circuit.noisy_cnot(edge[0], edge[1])
            circuit.noisy_rotate_z(2 * gamma[i], edge[1])
            circuit.noisy_cnot(edge[0], edge[1])
        
        # Construct mixer layer with parameter beta[i]
        for q in range(n):
            circuit.noisy_rotate_x(2 * beta[i], q)
    
    #return circuit
    return circuit

# Define nodes and edges
nodes = [0, 1, 2, 3, 4]
edges = [(0,1), (0,2), (1,2), (1,3), (2,3), (1,4), (2,4), (3,4)]

NOISE_FACTORS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
NUM_OPTIMIZATION_ITERATIONS_SPSA = 10
DATA_PATH = "data_noise/data.toml"

def run(noise_factor):
    print(noise_factor)
    # Calculating maxcut using bruteforce technique
    all_partitions_data, max_cut_index = maxcut_bruteforce(nodes, edges)
    optimal_cut_size = all_partitions_data[max_cut_index][1]

    # Building and executing QAOA circuit with 0 layers
    circuit = qaoa_circuit([], [], nodes, edges, 0, noise_factor=noise_factor)
    statevectors = execute_circuit(circuit)

    # Measuring circuit and generating data and graphs
    average_cut_size_random = compute_average_cut_size_from_probabilities(nodes, edges, statevectors)

    # Optimizing parameteres (gamma and beta)
    p = 5
    gamma_guess: list[float] = [0.5] * p
    beta_guess: list[float] = [0.5] * p
    optimal_average_cut_size: float = 0
    optimal_parameters: list[float] = []
    optimal_statevectors = []

    def OptimizeParameters(parameters):
        nonlocal optimal_average_cut_size, optimal_parameters, optimal_statevectors

        gamma = parameters[:p]
        beta = parameters[p:]
        circuit = qaoa_circuit(gamma, beta, nodes, edges, p, noise_factor=noise_factor)
        statevectors = execute_circuit(circuit)
        average_cut_size = compute_average_cut_size_from_probabilities(nodes, edges, statevectors)

        if average_cut_size > optimal_average_cut_size:
            optimal_average_cut_size = average_cut_size
            optimal_parameters = parameters
            optimal_statevectors = statevectors

        return -average_cut_size

    # Finding optimal parameters
    minimize_spsa(OptimizeParameters, gamma_guess + beta_guess, iterations=NUM_OPTIMIZATION_ITERATIONS_SPSA, lr=0.2, lr_decay=0.602, px=0.2)
    gamma = optimal_parameters[:p]
    beta = optimal_parameters[p:]

    print("Optimized")

    # Finding average cut size for top N of probabilities
    top_n = [10, 5, 3]
    average_cut_size_top_n = []
    for n in top_n:
        average_cut_size_top_n.append(compute_average_cut_size_from_probabilities(nodes, edges, optimal_statevectors, top_n=n))

    # Finding actual maxcut based on probabilities
    maxcut = compute_max_cut_from_probabilities(nodes, edges, optimal_statevectors)

    # Plotting probabilities
    file_name = f"noise_{noise_factor}.png"
    plot_probability_distribution_noise(optimal_statevectors, file_name, noise_factor=noise_factor)

    with open(DATA_PATH, 'a') as file:
        file.writelines([
            f"\n[{file_name}]", 
            f"\nnoise_factor = {noise_factor}" , 
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

for factor in NOISE_FACTORS:
    run(factor)
