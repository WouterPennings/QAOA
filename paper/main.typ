#import "@preview/ssrn-scribe:0.7.0": *

// if you do not want to use the integrated packages, you can comment out the following lines
#import "extra.typ": *
#show: great-theorems-init

#show: paper.with(
  font: "PT Serif", // "Times New Roman"
  fontsize: 12pt, // 12pt
  maketitle: true, // whether to add new page for title
  title: [An Analysis of Quantum Approximate Optimization Algorithm Performance Across Various Graph Types],
  // subtitle: "A work in progress", // subtitle
  authors: (
    (
      name: "Wouter Pennings",
      affiliation: "Fontys University of Applied Sciences",
      email: "465288@student.fontys.nl",
    ),
  ),
  date: "June 2025",
  abstract: lorem(80), // replace lorem(80) with [ Your abstract here. ]
  keywords: [Quantum computing, QAOA, Quantum noise, Graphs, Maxcut, Optimization],
  // JEL: [G11, G12],
  acknowledgments: "This paper is a work in progress. Please do not cite without permission.", 
  bibliography: bibliography("bib.bib", title: "References", style: "ieee"),
)

= Introduction <sec:introduction>
In recent years, quantum computing has emerged as a promising field with the potential to revolutionize the way complex computational problems in fields such as medicine, material science and machine learning are approached. Various algorithms have garnered attention for their ability to solve problems more efficiently than their classical peers; Quantum Approximate Optimization Algorithm is one of them.

Quantum Approximate Optimization Algorithm (QAOA) is a hybrid quantum algorithm designed to accurately approximate optimization problems . This exploration uses QAOA to address the max-cut problem. Max-cut is a graph problem where the goal is to divide the nodes into two groups so that the number of cut edges is as large as possible. With QAOA, each node in the graph is represented by a qubit, which is put into a superposition state using a Hadamard (H) gate. The edges connecting these nodes are represented by entangling the corresponding qubits, using either a ZZ-gate or a sequence of CNOT-RZ-CNOT gates.

QAOA has three key parameters: β, γ, and p. The parameter p sets the depth of the circuit, essentially dictating how many layers of operations are applied. The parameters β and γ are the angles that are optimized to find the best possible solution to our problem. The β angles are used in the mixing layer, also known as the mixing Hamiltonian, where they control the mixing of quantum states by rotating qubits around the X-axis. This process helps explore different configurations of the solution space. On the other hand, the γ angles are applied in the cost layer, or the cost Hamiltonian, where they adjust the rotations, guiding the solution towards an optimal outcome. The implementation used in this paper draws on the approach detailed in Farhi et al., ensuring that the study builds on established methods while exploring new insights.

The objective of this paper is to explore the behavior of QAOA across a range of graph types. The generation of the graphs is achieved by altering two parameters: the number of nodes and connectivity. The majority of experiments are conducted in a noise-less environment; however, several experiments are also conducted with noise to study its impact. There is a lack of comprehensive studies that explore the practical application of QAOA, and this paper aims to address this gap in literature. In quantum computing research, algorithms are often proposed without sufficient experimentation to understand how well they work in practice, this study seeks to provide valuable insights in this regard.


= Methodology <sec:methodology>
The objective of this study is to understand how the Quantum Approximate Optimization Algorithm (QAOA) behaves when solving the max-cut problem across various graphs in both ideal and noisy environments.

Two types of experiments are conducted. The first type, noisy experiments, aims to understand how noise impacts QAOA's effectiveness. This is achieved by applying increasing levels of noise to a single graph until a realistic noise level is reached. The second type, noiseless or ideal experiments, investigates QAOA performance across different graphs, each with increasing numbers of nodes or edges.

For the ideal experiments, graphs consist of 4, 5, 6, 7, 8, 9, 10, or 15 nodes with randomly generated edges. Each size category has two variants: low connectivity and high connectivity. Low connectivity implies a 40% chance of connection between nodes, while high connectivity implies an 80% chance. This results in a total of sixteen distinct graphs, all generated using NetworkX.

The noisy experiments utilize a single graph, referred to as the "house with a X" (@house_with_x). These experiments explore how increasing noise levels degrade QAOA's effectiveness. The noise levels, implemented as `noise_factor`, considered are 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, and 1. Baseline experiments include randomly selected max-cut instances and an ideal circuit simulation. The noise models are based on Di Bartolomeo et al.'s implementation, with parameters derived from Qiskit. The additional parameter, `noise_factor`, adjusts the noise model's `T1`, `T2` (inversely 
proportional to noise_factor), and `P` (proportional to noise_factor) to assess its impact on QAOA output quality.

#figure(
  image("../data_noise/graph.png", width: 50%),
  caption: [Visualization of "house with a X"],
) <house_with_x>

Both ideal and noisy experiments employ the Simultaneous Perturbation Stochastic Approximation (SPSA) to optimize γ and β parameters, parameter `P` is fixed at five across all experiments. SPSA's parameters are set to default values except for iteration (set to 10), learning rate (0.2), learning rate decay (0.602), and perturbation magnitude (0.2).

Evaluation metrics for both experiment types include:
-	*mean_random*: The average cut size based on probabilities, assuming P is 0.
-	*mean*: The average cut size based on the statevector's probabilities.
-	*mean_top_10*: The average cut size of the top ten cuts with the highest probability.
-	*mean_top_5*: The average cut size of the top five cuts with the highest probability.
-	*mean_top_3*: The average cut size of the top three cuts with the highest probability.
-	*maxcut_qaoa*: The cut with the highest probability from QAOA.
-	*maxcut_bruteforce*: The actual max-cut of the graph determined through classical brute-force methods.

In the case of noisy experiments, the statevector probabilities are also shown, which indicate the deviation in results due to noise. These metrics help determine QAOA's performance effectiveness in solving the max-cut problem.

== QAOA implementation & measuring
The implementation used for the experiments in this paper and described below is based on Fahri et al. The experiments are not run on actual hardware, but simulated using QuantumSim, an educational quantum computer simulator developed at Fontys University of Applied Sciences. This paper uses a computationally improved version which allows for faster simulation and for more qubits to be simulated.

QAOA’s three main steps of this paper’s implementation (@qaoa_code) are described below:
- *Step 1*: Creating quantum circuit: Initializing a quantum circuit, each qubit represents a node of the graph.
- *Step 2*: Circuit into superposition: Putting all qubits into superposition using a Hadamard (`H`) gate.
- *Step 3*: Construction layers of cost and mixer Hamiltonian: Applying a `ZZ` gate between two qubits, which entangles them, this is the cost Hamiltonian. Using CNOT-RZ-CNOT gates to simulate a `ZZ` gate . Applying a `RX` gate to each qubit, this is the mixer Hamiltonian, the β value of the `RX` gate is multiplied by two.
- *Step 3.1*: 

After these steps a QAOA circuit is created and can be measured to find the maxcut of a graph. We will be directly looking at the probabilities of the simulator instead of measuring the statevector thousands of times. Looking directly at the probabilities of the statevector is only possible in simulators. It does not change the results of the quantum circuit as with enough measurements the results would reflect the probabilities of the statevector anyway. It does make the simulation process faster as measuring a circuit 100.000 times does take a significant amount of time. In the case of a noisy circuit, the circuit is executed a hundred times, and a mean of the probabilities is taken. The probabilities of the noisy circuit are indeterministic, therefore, the probabilities can be quite different between executes; requiring several executes. 

#figure(
  image("../ssrn-scribe/images/qaoa_code.png", width: 75%),
  caption: [Example implemenation of QAOA, implemented in QuantumSim],
) <qaoa_code>

#colbreak()
= Results <sec:Results>

== Noisy experiments

== Noiseless experiments

=== Low connectivity
#show table.cell.where(y: 0): set text(weight: "bold")

#figure(
  table(
    columns: (0.7fr, 1fr, 0.7fr, 0.8fr, 0.7fr, 0.7fr, 1fr, 1fr, 1.1fr),
    align: (center, center, center, center, center, center, center, center, center),
    inset: 4pt,

    // Table Headers
    [Nodes],
    [Mean Random],
    [Mean All],
    [Mean Top 10],
    [Mean Top 5],
    [Mean Top 3],
    [MaxCut QAOA],
    [MaxCut Brute Force],
    [Figure],

    // Data Rows
    [4], [1.50], [2.65], [2.66], [2.89], [2.96], [3], [3], [@graph_4_low],
    [5], [2.50], [3.17], [3.61], [4.00], [4.00], [4], [4], [@graph_5_low],
    [6], [3.00], [4.48], [5.00], [5.00], [5.00], [5], [5], [@graph_6_low],
    [7], [3.00], [4.71], [5.48], [5.69], [5.87], [6], [6], [@graph_7_low],
    [8], [6.00], [6.65], [7.58], [7.56], [7.30], [7], [10], [@graph_8_low],
    [9], [8.00], [8.83], [10.14], [10.37], [10.56], [11], [13], [@graph_9_low],
    [10], [8.00], [9.02], [12.86], [13.00], [13.00], [13], [13], [@graph_10_low],
    [15], [24.50], [27.54], [32.95], [32.56], [32.30], [32], [35], [@graph_15_low],
  )
) <noiseless_low_results_table>

=== High connectivity
#show table.cell.where(y: 0): set text(weight: "bold")

#figure(
  table(
    columns: (0.7fr, 1fr, 0.7fr, 0.8fr, 0.7fr, 0.7fr, 1fr, 1fr, 1.1fr),
    align: (center, center, center, center, center, center, center, center, center),
    inset: 4pt,

    // Table Headers
    [Nodes],
    [Mean Random],
    [Mean All],
    [Mean Top 10],
    [Mean Top 5],
    [Mean Top 3],
    [MaxCut QAOA],
    [MaxCut Brute Force],
    [Figure],

    // Data Rows
    [4], [2.50], [3.65], [3.78], [3.87], [3.95], [4], [4], [@graph_4_high],
    [5], [3.50], [4.37], [4.85], [5.00], [5.00], [5], [5], [@graph_5_high],
    [6], [7.00], [7.98], [8.85], [9.00], [9.00], [9], [9], [@graph_6_high],
    [7], [9.00], [9.64], [11.63], [12.00], [12.00], [12], [12], [@graph_7_high],
    [8], [11.00], [11.63], [12.40], [13.33], [13.83], [15], [15], [@graph_8_high],
    [9], [16.00], [16.69], [17.60], [17.43], [17.69], [18], [20], [@graph_9_high],
    [10], [21.50], [21.99], [20.78], [20.00], [20.00], [20], [25], [@graph_10_high],
    [15], [41.00], [42.16], [48.38], [48.80], [48.24], [48], [53], [@graph_15_high],
  )
) <noiseless_high_results_table>

= Discussion

#colbreak()
= Figures
#grid(
  columns: 2,
  gutter: 10pt,
  row-gutter: 20pt, // Added for better vertical spacing between rows of pairs
  // Pair 1
  [
    #figure(
      image("../data_noiseless/graph_4_low.png", width: 100%),
      caption: [Graph with 5 nodes and low connectivity],
    ) <graph_4_low>
  ],
  [
    #figure(
      image("../data_noiseless/graph_4_high.png", width: 100%),
      caption: [Graph with 5 nodes and high connectivity],
    ) <graph_4_high>
  ],
  [
    #figure(
      image("../data_noiseless/graph_5_low.png", width: 100%),
      caption: [Graph with 5 nodes and low connectivity],
    ) <graph_5_low>
  ],
  [
    #figure(
      image("../data_noiseless/graph_5_high.png", width: 100%),
      caption: [Graph with 5 nodes and high connectivity],
    ) <graph_5_high>
  ],
  // Pair 2
  [
    #figure(
      image("../data_noiseless/graph_6_low.png", width: 100%),
      caption: [Graph with 6 nodes and low connectivity],
    ) <graph_6_low>
  ],
  [
    #figure(
      image("../data_noiseless/graph_6_high.png", width: 100%),
      caption: [Graph with 6 nodes and high connectivity],
    ) <graph_6_high>
  ],
  // Pair 3
  [
    #figure(
      image("../data_noiseless/graph_7_low.png", width: 100%),
      caption: [Graph with 7 nodes and low connectivity],
    ) <graph_7_low>
  ],
  [
    #figure(
      image("../data_noiseless/graph_7_high.png", width: 100%),
      caption: [Graph with 7 nodes and high connectivity],
    ) <graph_7_high>
  ],
  // Pair 4
  [
    #figure(
      image("../data_noiseless/graph_8_low.png", width: 100%),
      caption: [Graph with 8 nodes and low connectivity],
    ) <graph_8_low>
  ],
  [
    #figure(
      image("../data_noiseless/graph_8_high.png", width: 100%),
      caption: [Graph with 8 nodes and high connectivity],
    ) <graph_8_high>
  ],
  // Pair 5
  [
    #figure(
      image("../data_noiseless/graph_9_low.png", width: 100%),
      caption: [Graph with 9 nodes and low connectivity],
    ) <graph_9_low>
  ],
  [
    #figure(
      image("../data_noiseless/graph_9_high.png", width: 100%),
      caption: [Graph with 9 nodes and high connectivity],
    ) <graph_9_high>
  ],
  // Pair 6
  [
    #figure(
      image("../data_noiseless/graph_10_low.png", width: 100%),
      caption: [Graph with 10 nodes and low connectivity],
    ) <graph_10_low>
  ],
  [
    #figure(
      image("../data_noiseless/graph_10_high.png", width: 100%),
      caption: [Graph with 10 nodes and high connectivity],
    ) <graph_10_high>
  ],
  // Pair 7
  [
    #figure(
      image("../data_noiseless/graph_15_low.png", width: 100%),
      caption: [Graph with 15 nodes and low connectivity],
    ) <graph_15_low>
  ],
  [
    #figure(
      image("../data_noiseless/graph_15_high.png", width: 100%),
      caption: [Graph with 15 nodes and high connectivity],
    ) <graph_15_high>
  ],
)