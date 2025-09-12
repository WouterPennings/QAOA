# QuantumSim Performante

The main improvements of quantumsim can be found in [QuantumSimComputationalImprovements.ipynb](QuantumSimComputationalImprovements.ipynb), and the full implementation in [quantumsim_performante.py](quantumsim_performante.py). This is only a minimal version (see: [quantumsim_minima.py](quantumsim_minima.py)), however it does proof that this approach is effective at optimizing quantumsim. Nico's repository for QuantumSim can be found [here](https://projects.fhict.nl/ai-en-big-data-lectoraat/quantum-computing/QuantumSim).

- [quantumsim.py](quantumsim.py): Nico's full implementation of QuantumSim
- [quantumsim_minima.py](quantumsim_minima.py): Only the basic features of quantum computer simulator, based on Nico's QuantumSim
- [quantumsim_performante.py](quantumsim_performante.py): An accelerated smaller version of QuantumSim, excluding noisy circuits.
- [QuantumSimComputationalImprovements.ipynb](QuantumSimComputationalImprovements.ipynb): Outlines the improvements made in `quantumsim_performante.py`
- [paper.pdf](paper.pdf): Experimental paper about performance of QAOA across various graphs
- [posters.pdf](posters.pdf): Two reseach posters, one about QAOA, the other about `quantumsim_performante.py`
- [paper/](paper/): Files for the paper
- [reference_notebooks/](reference_notebooks/): Essential reference notebooks used for paper's experiments
- [assets/](assets/): Assets for quantum simulations, basically only noise parameters. 

## QuantumSim Performante Example
```python
def qaoa_circuit(gamma:list[float], beta:list[float], nodes:list, edges:list, p:int) -> Circuit:
    # Create circuit witn n qubits, where n is the number of nodes
    n = len(nodes)
    circuit = Circuit(n, use_lazy=True, use_GPU=True, use_cache=True)
    
    # Initialize circuit by applying the Hadamard gate to all qubits
    for q in range(n): circuit.hadamard(q)
        
    # Construct p alternating cost and mixer layers
    for i in range(p):
        # Construct cost layer with parameter gamma[i]
        for edge in edges:
            circuit.cnot(edge[0], edge[1])
            circuit.rotate_z(2 * gamma[i], edge[1])
            circuit.cnot(edge[0], edge[1])

        # Construct mixer layer with parameter beta[i]
        for q in range(n): circuit.rotate_x(2 * beta[i], q)

    return circuit
```

## Noisy QuantumSim 

There was also an attempt at improving the performance of QuantumSim with noise. Sadly, this implementation was very incomplete, buggy and unvalidated. However, a similar approach could be used to implement the noisy version of QuantumSim.

## Recommendations

I recommend to use the computational improvements I propose in the actual noiseless version of QuantumSim. However, I also recommend that the entire implementation of Quantum (with noise) is checked and refactored. There are a bunch of poor and inconcistant design choices that make it difficult to add and improve the implementation. A incomplete list of poorly implemented parts of QuantumSim:

1. Two implementations of noise
2. Two approaches to building unitary matrices (look at noisy CNOT)
3. Some noisy gates, such as: `Rx`, are not implemented.
4. In the execution, it is doing a string lookup in the description of the gate to change what it is doing.

Also, no one seems to understand how noise is actually implemented. The implementation was copied from a paper without understanding how the implementation works. This is concerning.

# The paper

The paper presents an analysis of the Quantum Approximate Optimization Algorithm (QAOA)performance across various graph types, focusing on its application to the maxcut problem.

The paper uses QuantumSim Performante for its noiseless experiments. It does not talk about the implementation of QuantumSim Performante or its orignal implementation.