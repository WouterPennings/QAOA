// TODOs:
// - Put the results of this study better in context with other papers and research.
// - Abstract is temporary, this is bad
// - Fix "Conceptual problem: Parameter selection"
// - Add piece about future work in error correction codes.
// - Where does this SPSA approach come from??? Isnt the fahri paper with the increasing beta and decreasing gamma

#import "@preview/ssrn-scribe:0.7.0": *

// if you do not want to use the integrated packages, you can comment out the following lines
#import "extra.typ": *
#show: great-theorems-init

#show: paper.with(
  font: "PT Serif", // "Times New Roman"
  fontsize: 12pt, // 12pt
  maketitle: true, // whether to add new page for title
  title: [An Analysis of Quantum Approximate Optimization Algorithm Performance Across Various Graph Types],
  // subtitle: "A Practical Study of QAOA on Max-Cut with Varying Node Counts and Connectivity",
  authors: (
    (
      name: "Wouter Pennings",
      affiliation: "Fontys University of Applied Sciences",
      email: "465288@student.fontys.nl",
    ),
  ),
  date: "June 2025",
  abstract: [
    This paper presents an analysis of the Quantum Approximate Optimization Algorithm (QAOA) performance across various graph types, focusing on its application to the maxcut problem. Quantum computing, leveraging phenomena such as superposition and entanglement, offers potential advancements in solving complex computational problems more efficiently than classical methods. QAOA, a hybrid quantum algorithm, combines classical and quantum computational resources to approximate solutions to optimization problems. This study explores QAOA's effectiveness in both noiseless and noisy environments using a quantum computer simulator. The noiseless experiments involve varying graph sizes and connectivity levels, for realistic noisy quantum computing conditions experiments how different levels of noise impact its performance. Key parameters of QAOA, β, γ, and `P`, are optimized using the Simultaneous Perturbation Stochastic Approximation (SPSA) method. The results indicate that QAOA performs well in identifying optimal or near-optimal max-cuts for smaller graphs but faces challenges as graph size increases. Noise significantly impacts QAOA's performance, although it still outperforms random selection methods. The study highlights the need for further research to optimize QAOA parameters and explore its scalability and robustness in practical applications. The findings highlight QAOA's potential for approximating solutions to optimization problems on small-scale graphs, while also underscoring challenges related to scalability and performance consistency in the presence of quantum noise. Limitations include the limited diversity of graphs tested, particularly in noisy experiments, and the fixed `P` parameter, suggesting avenues for future research. This study contributes valuable practical insights into QAOA's behavior, addressing a gap in comprehensive experimental analysis of quantum algorithms.
  ], // replace lorem(80) with [ Your abstract here. ]
  keywords: [Quantum computing, QAOA, Quantum noise, Graphs, Maxcut, Optimization],
  // JEL: [G11, G12],
  acknowledgments: "Qin Zhoa and Nico Kuijpers were my supervisor during the process of writing paper.", 
  bibliography: bibliography("ref.bib", title: "References", style: "ieee"),
)

= Introduction <sec:introduction>
In recent years, quantum computing has emerged as a promising field with the potential to revolutionize the way complex computational problems in fields such as medicine, material science and machine learning are approached @hastings_improving_2014 @svore_quantum_2016 @biamonte_quantum_2017. Various algorithms have gotten attention for their theoratical ability to solve problems more efficiently than their classical peers; Quantum Approximate Optimization Algorithm is one of them. To understand this paper, it is important to grasp two key quantum phenomena: superposition and entanglement. Superposition allows a quantum bit (qubit) to exist in a both the 0 and 1 state simultaneously until it is measured, at which point it collapses into a single definite state @yanofsky_introduction_2007 @prashant_study_2007, this mechanic is represented by a Hadamard (H) gate. Entanglement, on the other hand, describes a connection between two or more qubits where their states are intertwined, meaning the state of one qubit influences the state of another @yanofsky_introduction_2007 @prashant_study_2007, this mechanic is represented by a controlled-not (CNOT) gate.

Quantum Approximate Optimization Algorithm (QAOA) is a hybrid quantum algorithm designed to accurately approximate optimization problems. A hybrid quantum algorithm, means it combines both classical and quantum computational resources, leveraging the strengths of each to solve complex problems @campos_hybrid_2024. This exploration uses QAOA to address the max-cut problem. Maxcut is a graph problem where the goal is to divide the nodes into two groups so that the number of cut edges connecting the different groups is as large as possible. This paper uses the varient where all edges have the same weight. With QAOA, each node in the graph is represented by a qubit, which is put into a superposition state using a Hadamard (H) gate. The edges connecting these nodes are represented by entangling the corresponding qubits, using either a ZZ-gate or a sequence of CNOT-RZ-CNOT gates.

QAOA has three key parameters: β, γ, and p. The parameter p sets the depth of the circuit, essentially dictating how many layers of operations are applied. The parameters β and γ are the angles that are classically optimized to find the best possible solution to our problem. The β angles are used in the mixing layer, also known as the mixing Hamiltonian, where they control the mixing of quantum states by rotating qubits around the X-axis. This process helps explore different configurations of the solution space. On the other hand, the γ angles are applied in the cost layer, or the cost Hamiltonian, where they adjust the rotations, guiding the solution towards an optimal outcome. The implementation used in this paper draws on the approach detailed in Farhi et al. @farhi_quantum_2014, ensuring that the study builds on established methods while exploring new insights.

Several works provide foundational context and highlight ongoing challenges in QAOA research. The review by Blekos et al. @blekos_review_2024 offers a recent comprehensive overview of QAOA and its variants, discussing its applicability, hardware challenges such as error susceptibility, and various parameter optimization strategies. While their work broadly surveys the landscape of QAOA, including some experimental implementations of variants, this analysis aims to address the lack of research on the practical application of QAOA, particularly how it behaves across different types of graphs. The study explores the effectiveness of QAOA on the maxcut problem by generating graphs through altering two key parameters: the number of nodes and connectivity. All experiments are conducted using a quantum computer simulator because it is more accessible than actual quantum hardware and has the unique capability to precisely control and vary noise levels. While most experiments are performed in a noiseless environment, some are conducted with increasing levels of noise to study its effect on performance. This approach builds upon existing work, such as that by Herrman et al. @herrman_impact_2021, who examined the relationship between particular graph features (e.g., symmetries, odd cycles, and density) and QAOA success by analyzing smaller graphs. Similar to their work, this paper examines how graph types influence QAOA performance for maxcut. However, our study differentiates itself by considering a wider range of graph sizes and incorporating noisy simulations. This provides a more generalized understanding of QAOA's behavior under conditions relevant to current and future quantum hardware. The study offers an experimental analysis of QAOA's practical performance amidst varied graph complexities and environmental noise.

Analyzing QAOA under noise-free conditions is important for understanding its fundamental capabilities and limitations, independent of the imperfections inherent in current quantum hardware. While it is widely acknowledged that current quantum computers operate in the Noisy Intermediate-Scale Quantum (NISQ) era, where noise significantly impacts computation, noiseness simulations offer a baseline to which more realistic experiment results can be compared to.

= Methodology <sec:methodology>
The objective of this study is to understand how the Quantum Approximate Optimization Algorithm (QAOA) behaves when solving the max-cut problem across various graphs in both ideal and noisy environments.

Two types of experiments are conducted. The first type, noisy experiments, aims to understand how noise impacts QAOA's effectiveness. This is achieved by applying increasing levels of noise to a single graph until a realistic noise level is reached. The second type, noiseless or ideal experiments, investigates QAOA performance across different graphs, each with increasing numbers of nodes or edges.

For the ideal experiments, graphs consist of 4, 5, 6, 7, 8, 9, 10, or 15 nodes with randomly generated edges. Each graph size has two variants: low connectivity and high connectivity. Low connectivity implies a 40% chance of connection between nodes, while high connectivity implies an 80% chance. This results in a total of sixteen distinct graphs, all generated using NetworkX. NetworkX is a Python library for studying graphs and networks @hagberg_exploring_2008. 

The noisy experiments utilize a single graph, referred to as the "house with a X" (@house_with_x). These experiments explore how increasing noise levels degrade QAOA's effectiveness. The noise levels, implemented as `noise_factor`, considered are 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, and 1. Baseline experiments include randomly selected max-cut instances and an ideal circuit simulation. The noise models are based on Di Bartolomeo et al.'s implementation @di_bartolomeo_noisy_2023, with parameters used from Qiskit Kyiv quantum computer. These parameters represent the single-qubit depolarizing error probability (`P`), amplitude damping time of a qubit in ns (`T1`) and the dephasing time ofa qubit in ns (`T2`). `P` is multiplied by `noise_factor` while `T1` and `T2` are divided by `noise_factor` as they are inversely proportional to `noise_factor`.  

#figure(
  image("../data_noise/graph.png", width: 50%),
  caption: [Visualization of "house with a X"],
) <house_with_x>

Both ideal and noisy experiments employ the Simultaneous Perturbation Stochastic Approximation (SPSA) to optimize γ and β parameters, parameter `P` is fixed at five across all experiments. SPSA's parameters are set to default values except for iteration (set to 10), learning rate (0.2), learning rate decay (0.602), and perturbation magnitude (0.2). The initial guess used in this sudy of γ and β for all layers is 0.5. There are many optimizers available, such as COBYLA, however SPSA seems to perform well in context such as these @pellow-jarman_comparison_2021.

To evaluate and compare the performance of the experiments, several evaluation metrics are used. The metrics are the same for both type of experiments. The metrics are:
-	*mean_random*: The average cut size based on probabilities, assuming P is 0.
-	*mean*: The average cut size based on the statevector's probabilities.
-	*mean_top_10*: The average cut size of the top ten cuts with the highest probability.
-	*mean_top_5*: The average cut size of the top five cuts with the highest probability.
-	*mean_top_3*: The average cut size of the top three cuts with the highest probability.
- *approximation_ratio*: $"mean_top_10"/"maxcut_bruteforce" = "approxmiation_ratio"$. This metric is used in other papers, such as @herrman_impact_2021 and @noauthor_maxcut_nodate
-	*maxcut_qaoa*: The cut with the highest probability from QAOA.
-	*maxcut_bruteforce*: The actual max-cut of the graph determined through classical brute-force methods.

Average cut size is a weighted mean where the weights are based on the probabilities.

In the case of noisy experiments, the statevector probabilities are also shown, which indicate the deviation in results due to noise. These metrics help determine QAOA's performance effectiveness in solving the max-cut problem.

== QAOA implementation & measuring
The implementation used for the experiments in this paper and described below is based on Fahri et al @farhi_quantum_2014. The experiments are not run on actual hardware, but simulated using QuantumSim @kuijpers_2025, an educational quantum computer simulator developed at Fontys University of Applied Sciences. This paper uses a computationally improved version which allows for faster simulation and for more qubits to be simulated.

QAOA’s three main steps of this paper’s implementation (@qaoa_code) are described below. @qaoa_circuit provides a visualization of the circuit.
- *Step 1*: Creating quantum circuit: Initializing a quantum circuit, each qubit represents a node of the graph.
- *Step 2*: Circuit into superposition: Putting all qubits into superposition using a Hadamard (`H`) gate.
- *Step 3*: Construction layers of cost and mixer Hamiltonian: Applying a `ZZ` gate between two qubits, which entangles them, this is the cost Hamiltonian. Using CNOT-RZ-CNOT gates to simulate a `ZZ` gate . Applying a `RX` gate to each qubit, this is the mixer Hamiltonian, the β value of the `RX` gate is multiplied by two.

#figure(
  image("images/QAOA_circuit_p2.png", width: 100%),
  caption: [The QAOA circuit was generated by Qiskit. Barriers are added before and after the cost layer to prevent Qiskit from rearranging them. The circuit has two layers, and both β and γ are `[1, 2]`. The graph is a basic triangle, it has three nodes, 1 is connected to 2 and 3, and 3 is also connected to 2.],
) <qaoa_circuit>


After these steps a QAOA circuit is created and can be measured to find the maxcut of a graph. We will be directly looking at the probabilities of the simulator instead of measuring the statevector thousands of times. Looking directly at the probabilities of the statevector is only possible in simulators. It does not change the results of the quantum circuit as with enough measurements the results would reflect the probabilities of the statevector anyway. It does make the simulation process faster as measuring a circuit 100.000 times does take a significant amount of time. In the case of a noisy circuit, the circuit is executed a 30 times, and a mean of the probabilities is taken. The probabilities of the noisy circuit are indeterministic, therefore, the probabilities can be quite different between executes; requiring several executes. 

= Results <sec:Results>

== Noisy experiments

@noisy_experiment_metrics_table show all metric tracked for the noisy experiments. `Noise factor` describes the amount of noise. `Mean random`, `Mean`, `Mean Top 10`, `Mean Top 5` and `Mean Top 3` all are average cut sizes, in the case of `Mean Top N` it only calculates the average cut size of the outcomes with the top N highest probabilities. `Maxcut QAOA` and `Maxcut BruteForce` show how many edges were cut. `Maxcut QAOA` is the cut with the highest probability of all possible soltions. 

#show table.cell.where(y: 0): set text(weight: "bold")
#figure(
  table(
    columns: (1fr, 1fr, 1fr, 1fr, 1fr, 0.7fr, 1fr, 1fr, 1fr), // Adjusted for 8 columns
    align: (center, center, center, center, center, center, center, center, center),
    inset: 4pt,

    // Table Headers
    [Noise Factor],
    [Mean Random],
    [Mean All],
    [Mean Top 10],
    [Mean Top 5],
    [Mean Top 3],
    [Approx. ratio],
    [MaxCut QAOA],
    [MaxCut BruteForce],

    // Data Rows
  [Noiseless], [4.00], [4.64], [4.86], [5.08], [5.45], [0.81], [6], [6],
  [0.001], [4.00], [4.16], [4.49], [4.43], [4.15], [0.75], [5], [6],
  [0.005], [4.00], [4.15], [4.45], [4.65], [4.73], [0.74], [5], [6],
  [0.01], [4.00], [4.13], [4.46], [4.66], [5.00], [0.74], [5], [6],
  [0.05], [4.00], [4.19], [4.40], [4.40], [4.34], [0.73], [5], [6],
  [0.1], [4.00], [4.16], [4.60], [5.18], [5.00], [0.77], [5], [6],
  [0.5], [4.00], [4.11], [4.70], [4.65], [5.00], [0.78], [5], [6],
  [1.0], [4.00], [4.13], [4.71], [4.82], [4.73], [0.78], [5], [6],
  ),
  caption: "This table shows all the metrics tracked for the noisy experiments, including the noiseless run, conducted on the \"house with an X\" graph (Figure 1)."
) <noisy_experiment_metrics_table>


#figure(
  image("images/qaoa_noise_graph.png", width: 100%),
  caption: [This graph shows the same data as @noisy_experiment_metrics_table.],
) <qaoa_noise_graph>

=== Probability plots

Probability plots illustrate the likelihood of each possible quantum state. The sum of all probabilities for a given plot is 1, with individual probabilities ranging from 0 to 1. These probabilities describe the chance that a specific state will be the outcome when a quantum circuit is measured.

The plots present results for various conditions:
- @data_noise_noiseless represents the probabilities for an ideal, noiseless circuit.
- @data_noise_0.001, @data_noise_0.005, @data_noise_0.01, @data_noise_0.05, @data_noise_0.1, @data_noise_0.5, and @data_noise_1 show the probabilities for each experiment at its specific noise factor.
- The black bar indicates the standard deviation.
- The red and yellow arrows mark the upper and lower bounds of each possible outcome, derived from 30 executions.

At the bottom of each probability plot, labels such as `|00111>` and `|10001>` are presented. These labels represent the measured quantum states of the qubits in Dirac notation. For illustrative purposes, the state `|00111>` can be decomposed as follows:
- Qubit 1 is in state 0 (Commenly noted as |0>)
- Qubit 2 is in state 0 (Commenly noted as |0>)
- Qubit 3 is in state 1 (Commenly noted as |1>)
- Qubit 4 is in state 1 (Commenly noted as |1>)
- Qubit 5 is in state 1 (Commenly noted as |1>)

In the context of this study, these qubit states define the partitioning of a graph's nodes into the two groups of the cut graph. Each qubit represends a specific node within the graph. The state `|00111>` signifies that the first and second nodes are assigned to group A, the third, fourth, and fifth nodes are assigned to group B. This allows calculation of the number of cut edges. @data_noise_noiseless shows that `|01100>` and `|10011>` are the two outcomes with the highest probability. Noiseless or ideal probabilities for maxcut are always symmetrical, `|01100>` and `|10011>` is identical the nodes are just put into the opposite group. Noisy quantum computers have a random component to them and are indeterministic. Therefore, the probability plots of the noisy experiments, E.G. @data_noise_0.01, is not symmetrical.

#figure(
  image("../data_noise/noiseless.png", width: 100%),
  caption: [Probabilities of noiseless QAOA on "House with a X"],
) <data_noise_noiseless>

#figure(
  image("../data_noise/noise_0.001.png", width: 100%),
  caption: [Probabilities of noisy QAOA with a `noise_factor` of 0.001 on "House with a X"],
) <data_noise_0.001>

#figure(
  image("../data_noise/noise_0.005.png", width: 100%),
  caption: [Probabilities of noisy QAOA with a `noise_factor` of 0.005 on "House with a X"],
) <data_noise_0.005>

#figure(
  image("../data_noise/noise_0.01.png", width: 100%),
  caption: [Probabilities of noisy QAOA with a `noise_factor` of 0.01 on "House with a X"],
) <data_noise_0.01>

#figure(
  image("../data_noise/noise_0.05.png", width: 100%),
  caption: [Probabilities of noisy QAOA with a `noise_factor` of 0.05 on "House with a X"],
) <data_noise_0.05>

#figure(
  image("../data_noise/noise_0.1.png", width: 100%),
  caption: [Probabilities of noisy QAOA with a `noise_factor` of 0.1 on "House with a X"],
) <data_noise_0.1>

#figure(
  image("../data_noise/noise_0.5.png", width: 100%),
  caption: [Probabilities of noisy QAOA with a `noise_factor` of 0.5 on "House with a X"],
) <data_noise_0.5>

#figure(
  image("../data_noise/noise_1.png", width: 100%),
  caption: [Probabilities of noisy QAOA with a `noise_factor` of 1 on "House with a X"],
) <data_noise_1>



== Noiseless experiments

@noiseless_low_results_table and @qaoa_low_connectivity show all metric tracked for the low connectivity graphs. @noiseless_high_results_table and @qaoa_high_connectivity show all metric tracked for the high connectivity graphs. `Nodes` describes how many nodes there are in the graph used in the experiment. `Mean random`, `Mean`, `Mean Top 10`, `Mean Top 5` and `Mean Top 3` all are average cut sizes, in the case of `Mean Top N` it only calculates the average cut size of the outcomes with the top N highest probabilities. `Maxcut QAOA` and `Maxcut BruteForce` show how many edges were cut. `Maxcut QAOA` is the cut with the highest probability of all possible solutions. `Figure` is a reference to the graph which are in the @sec:figures 

=== Low connectivity
#show table.cell.where(y: 0): set text(weight: "bold")
#figure(
  table(
    columns: (0.7fr, 1fr, 0.7fr, 0.8fr, 0.7fr, 0.7fr, 1fr, 1fr, 1fr, 1.1fr),
    align: (center, center, center, center, center, center, center, center, center, center),
    inset: 4pt,

    // Table Headers
    [Nodes],
    [Mean Random],
    [Mean All],
    [Mean Top 10],
    [Mean Top 5],
    [Mean Top 3],
    [Approx. ratio],
    [MaxCut QAOA],
    [MaxCut Brute Force],
    [Figure],

    // Data Rows
    [4], [1.50], [2.65], [2.66], [2.89], [2.96], [0.89], [3], [3], [@graph_4_low],
    [5], [2.50], [3.17], [3.61], [4.00], [4.00], [0.90], [4], [4], [@graph_5_low],
    [6], [3.00], [4.48], [5.00], [5.00], [5.00], [0.90], [5], [5], [@graph_6_low],
    [7], [3.00], [4.71], [5.48], [5.69], [5.87], [0.91], [6], [6], [@graph_7_low],
    [8], [6.00], [6.65], [7.58], [7.56], [7.30], [0.76], [7], [10], [@graph_8_low],
    [9], [8.00], [8.83], [10.14], [10.37], [10.56], [0.79], [11], [13], [@graph_9_low],
    [10], [8.00], [9.02], [12.86], [13.00], [13.00], [0.99], [13], [13], [@graph_10_low],
    [15], [24.50], [27.54], [32.95], [32.56], [32.30], [0.94], [32], [35], [@graph_15_low],
  ),
  caption: "This table shows all the metrics tracked for the noiseless experiments with low connectivity"
) <noiseless_low_results_table>
#figure(
  image("images/qaoa_low_connectivity.png", width: 100%),
  caption: [This graph show how well QAOA performance a maxcut on a variety of graphs will different sizes with low connectivity by comparing multiple metrics to `Maxcut bruteforce`. Same data as @noiseless_low_results_table ],
) <qaoa_low_connectivity>

=== High connectivity
#show table.cell.where(y: 0): set text(weight: "bold")
#figure(
  table(
    columns: (0.7fr, 1fr, 0.7fr, 0.8fr, 0.7fr, 0.7fr, 1fr, 1fr, 1fr, 1.1fr),
    align: (center, center, center, center, center, center, center, center, center, center),
    inset: 4pt,

    // Table Headers
    [Nodes],
    [Mean Random],
    [Mean All],
    [Mean Top 10],
    [Mean Top 5],
    [Mean Top 3],
    [Approx. ratio],
    [MaxCut QAOA],
    [MaxCut Brute Force],
    [Figure],

    // Data Rows
    [4], [2.50], [3.65], [3.78], [3.87], [3.95], [0.95], [4], [4], [@graph_4_high],
    [5], [3.50], [4.37], [4.85], [5.00], [5.00], [0.97], [5], [5], [@graph_5_high],
    [6], [7.00], [7.98], [8.85], [9.00], [9.00], [0.98], [9], [9], [@graph_6_high],
    [7], [9.00], [9.64], [11.63], [12.00], [12.00], [0.97], [12], [12], [@graph_7_high],
    [8], [11.00], [11.63], [12.40], [13.33], [13.83], [0.82], [15], [15], [@graph_8_high],
    [9], [16.00], [16.69], [17.60], [17.43], [17.69], [0.88], [18], [20], [@graph_9_high],
    [10], [21.50], [21.99], [20.78], [20.00], [20.00], [0.83], [20], [25], [@graph_10_high],
    [15], [41.00], [42.16], [48.38], [48.80], [48.24], [0.91], [48], [53], [@graph_15_high],
  ),
  caption: "This table shows all the metrics tracked for the noiseless experiments with high connectivity"
) <noiseless_high_results_table>
#figure(
  image("images/qaoa_high_connectivity.png", width: 100%),
  caption: [This graph show how well QAOA performance a maxcut on a variety of graphs will different sizes with high connectivity by comparing multiple metrics to `Maxcut bruteforce`. Same data as @noiseless_high_results_table],
) <qaoa_high_connectivity>

= Discussion

== Noisy experiments

The experiments conducted with noise, present insights into how the QAOA performs in more realistic conditions. The noiseless experiment(@data_noise_noiseless) was the only one that found the exact maxcut of the graph. Its probability plot (Figure 4) showed confidence in the solution, with the quantum states `|01100>` and `|10011>` having the highest probabilities. In contrast, all noisy experiments resulted in a maxcut of 5, which is close to optimal but not the true maximum of 6.

The probability plots for noisy QAOA (@data_noise_0.001, @data_noise_0.005, @data_noise_0.01, @data_noise_0.05, @data_noise_0.1, @data_noise_0.5, and @data_noise_1) reveal large differences between executions, indicated by the wide standard deviation, minimum, and maximum values. For example, the state `|01111>` in @data_noise_1 sometimes had a probability of 0 and sometimes over 0.2. This variability clearly shows the unpredictable nature of quantum circuits when noise is present.

Interestingly, the actual amount of noise did not consistently worsen the quality of the output. @qaoa_noise_graph even shows that the `Mean Top 10` metric improved as the noise factor increased, although, that is likely a coincidence. While `Mean Top 5` and `Mean Top 3` fluctuated, the `Mean` and `Mean Random` values remained consistently low. This suggests that the presence of any noise was more impactful than the specific level of noise; which is against expectations. Nevertheless, all metrics measured in the noisy experiments were still better than selecting a solution randomly, as `Mean Random` is always the lowest value.

== Noiseless experiments

The performance of QAOA without noise is detailed in the graphs and tables (@qaoa_low_connectivity, @noiseless_low_results_table, @qaoa_high_connectivity, @noiseless_high_results_table). The high connectivity graph with 10 nodes was an exception, as its `Mean random` value was higher than other `Mean Top N` metrics. Therefore, this particular case was excluded from the following general observations.

Key findings from the noiseless experiments include:
- QAOA consistently outperforms random max-cut selections, even in scenarios where its performance is suboptimal.

- For both low and high connectivity graphs, QAOA found the most optimal maxcut for graphs with up to 7 nodes.

- For graphs with 8 nodes or more, QAOA's results often differed from the true maximum cut found by brute force, with the exception of the low connectivity graph with 10 nodes, where QAOA was effective.

- `Mean` did not accurately show QAOA's effectiveness. It was more insightfull to look at the average cut size of the top N (`Mean Top N`) solutions with the highest probabilities. This does indicate the the outcomes with the highest amount of cut edges are also the ones with the highest probabilities.

- As graphs grew larger, for both high and low connectivity, there was a trend where focusing on a smaller number of top solutions (`Mean Top N` with smaller N) did not always lead to a higher average cut size. This indicates that the best cuts are not always the most probable ones.

- QAOA's performance was generally better with high-connectivity graphs compared to the low connectivity graphs. This is likely to do with the fact that to even have a valid maxcut on a high connectivity graph you likely have to cut as more edges, relatively speaking compared to low connectivity graphs. That is also why `Mean random` is closer to `Maxcut Bruteforce` for high connectivity graphs compared to low connectivity graphs.

Why `N=10` with high connectivity is such an outlier is not understood. Similarly, `N=8` for low connectivity is also an outlier as `Mean Top 5` an `Mean Top 3` are both lower than `Mean Top 10`, eventhough this is not the case for `N=9` and `N=10`. There is no existing literature on the inconsistent performance of QAOA. The fact that QAOA has such worse results with these two graphs without there being a explanation is concerning, as that makes QAOA unreliable.

In summary, these experiments indicate that QAOA provides a good approximation of the optimal Max-cut solution for small graph sizes, irrespective of connectivity levels. However, QAOA's effectiveness diminishes as graph sizes increase, while connectivity appears to have a limited impact.

== QAOA P Count

In this study, the p parameter for QAOA was set to a constant value of 5 for all experiments. QAOA showed its weakest performance with the largest graphs (15 nodes), and this trend was also seen with graphs of 9 and 10 nodes. Increasing the `P` parameter could potentially allow QAOA to perform more iterations, which might help it find a more effective maxcut solution. Increaseing `P` might not help for noisy quantum computers, as deeper circuits suffer from more noise @ivezic_2024 @muller_limitations_2025, which might limit it effectiveness. 

Instead of choosing a static `P`, it is also an option to optimize `P` together with β and γ. This does complicate the parameter optimzation as this adds extra dimensions to optimize.

== Limitations
This study has several limitations:

// TODO: This is written very poorly
- *Graph diversity:* Only one type of graph was used for each experiment. QAOA can behave differently with various graphs, so averaging results from multiple graph types might have reduced the impact of unusual behaviors, which likely caused the outlier results for low connectivity graphs with 8 nodes and high connectivity graphs with 10 nodes.

// TODO: There is a source for this find it. It is in your note somewhere.
- *Real-world and common graph structures:* The paper did not explore how QAOA performs on real-life and common graph structures, like social networks, which have distinct features that might affect QAOA's performance.

- *Noisy experiment graph:* The noisy experiments used only one graph. As seen in the noiseless experiments, some graph sizes can be outliers, and the "house with X" graph used for noisy tests might also be such an outlier, other graphs might perform better or worse.

- *Number of noisy experiment executions:* The noisy experiments involved 30 executions per experiment. Results varied greatly between runs, a few skewed executions could heavily influence the outcome. More executions, around a hundred or more, would likely provide more consistent results, however further testing is needed.

- *Noise model and parameters:* This study used one approach to model noise in quantum computer simulations with a single set of parameters (Qiskit Kyiv). Exploring other noise implementations and/or different noise parameters could lead to very different results.

- *Optimizer configuration:* The SPSA optimization function, used to find the best beta and gamma values, might not be ideally set up to find optimal values in a noisy quantum computer environment

== Conceptual problem: Parameter selection

The strategy of optimizing QAOA parameters individually for each graph raises a conceptual concern: using a classical optimizer, such as SPSA, to determine parameters that yield a good maxcut solution effectively involves solving the problem in advance. This introduces an apparent circularity to the approach. The process involves sampling a large set of parameter combinations and selecting those that yield favorable results. However, this raises a concern: if the goal is simply to identify high-quality solutions through classical optimization, would it not be more efficient to use a classical heuristic, such as simulated annealing, to solve maxcut directly? 

A response to this concern might be that exploring a wide range of QAOA parameters and selecting the best performing configurations is more effective or computationally efficient than relying on purely classical heuristics. Nonetheless, this highlights an area where further research is needed. QAOA, and quantum computing more broadly, remains largely theoretical and is still far from practical deployment for real world problems. It would be interesting to investigate whether a set of "universal parameters" could be found that perform reasonably well across many graph instances, even if not optimally for each one. In this context, Muller et al. @muller_limitations_2025 suggest a degree of universality of optimal β and γ parameters for QAOA, which supports the feasibility of this direction.

Alternatively, future work could explore whether good parameter choices can be heuristically "guessed" based on structural properties of the graph, such as its size, degree distribution, or symmetry @sureshbabu_parameter_2024. Such approaches could help mitigate the inefficiencies and circularity of instance-specific optimization, potentially making QAOA a more scalable and practical algorithm.

= Conclusion

In noiseless environments, QAOA consistently outperformed random selection for the maxcut problem on the graphs tested. For smaller graphs (up to 7 nodes), QAOA was effective, often finding the true maximum cut. As the graph size increased beyond 7 nodes, QAOA's ability to identify the exact maxcut diminished, regardless of graph connectivity. However, metrics  still demonstrated that QAOA is relativally effective at approximating an optimal maxcut. In general, these experiments demonstrate current limitation of QAOA's scalability for larger problem instances on ideal quantum computers.

The noisy experiments provided insights into QAOA's behavior in more realistic scenarios. While the noiseless QAOA experiment found the optimal maxcut, the noisy implementations consistently resulted in a near-optimal cut, showing a degradation from noiseless performance. A significant observation was the high variability in outcome probabilities in the presence of noise. Still, noisy QAOA performed better than random selection, demonstrating its potential even when affected by errors.

#colbreak()
= Figures <sec:figures>
#figure(
  ```python
  def qaoa_circuit(gamma:list[float], beta:list[float], nodes:list, edges:list, p:int) -> Circuit:
      # Consistency check
      if len(gamma) != p or len(beta) != p:
          raise ValueError(f"Lists gamma and beta should be of length p = {p}")
      
      # Create circuit witn n qubits, where n is the number of nodes
      n = len(nodes)
      circuit = Circuit(n)
      
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

      return circuit
  ```,
  caption: [Example implemenation of QAOA, implemented in QuantumSim],
) <qaoa_code>

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
