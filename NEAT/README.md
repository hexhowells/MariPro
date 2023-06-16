# NEAT (IN PROGRESS)

NeuroEvolution of Augmenting Topologies (NEAT) expands on neuroevolution by adding evolving network topologies along side evolving network weights. This method aims to solve the *Competing Conventions Problem* by enabling effective crossover of neuroevolution systems without destroying useful information and structures.

---

## Encoding

<p align="center">
  <img src="https://github.com/hexhowells/MariPro/blob/main/NEAT/images/genome.jpg" width=65%>
</p>

NEAT encodes the genome into two parts, Node Genes and Connect Genes. 

Node Genes represent each node in the neural network and include metadata about the node index and position in the network (input, hidden, output).

Connect Genes represent the connections between nodes, metatdata specifies the following:
- in and out connection indexes
- weight value 
- enabled, flag that represents if the node is active in the network
- innovation, value that keeps track of historical information about the gene, used for *artifical synapsis*

## Mutation

<p align="center">
  <img src="https://github.com/hexhowells/MariPro/blob/main/NEAT/images/mutation.jpg" width=60%>
</p>

Mutation can occur at both the weight connections and the network structure, a global innovation number is incremented with every new gene created through mutation, in order to keep track of the age of each mutation (used later on).

Weight mutation is implemented the same as with other NeuroEvolution methods.

Topology mutation can occur in two ways:

#### Add connection
- Add a new connection gene between two unconnected nodes.


#### Add node
- Add a new node gene. 
- Split a connection gene by disabling the connection and adding two new connections between the split nodes. the incoming weight is set to 1, the outgoing weight is set to the same as the old weight.

## Crossover

<p align="center">
  <img src="https://github.com/hexhowells/MariPro/blob/main/NEAT/images/crossover.jpg" width=50%>
</p>

Crossover uses the innovation numbers to identifying matching and non-matching genes between the two parents. Genes with an innovation number that occur inside the other parents innovation numbers are marked as disjoint and genes with an innovation number that occur outside the other parents innovation numbers are marked as excess. If two genes share the same innovation number then they are considered matching.

Matching genes are chosen randomly between each parent to be used in the offspring. Disjoint and Excess genes are inherited from the more fit parent. Since some genes may be disabled from a parent there is a chance that the offspring will inherit disabled genes.


## Speciation

Since topological mutations typically decrease the networks fitness, time should be given to allow these networks time to adapt to their mutation. This is done through speciation.

Speciation divdes the population into smaller species that only compete with other organisms within that species. A compatibility distance score is measured between each new organism and a random organism from each species in order to determine which species the new organism belongs to. A distance score that does not fit within the threshold of any species is placed in their own new species.

The distance function is specified as follows:

```math
\large{\delta = \frac{c_1E}{N} + \frac{c_2D}{N}+c_3 \cdot \bar{W}}
```

- ```E``` - Number of Excess nodes (in both genomes)
- ```D``` - Number of Disjoint nodes (in both genomes)
- ```N``` - Number of genes in the larger genome
- ```c's``` - Adjustable coefficients to set the importance of each factor
- ```W``` - Average weight differences of matching genes

## Fitness Function

Instead of using a normal fitness function specified by the developer, NEAT uses *explicit fitness sharing* where members of the same species share their fitness scores, this discourages species becoming too large and dominiating as the average fitness is likely to diminish with the increase of population size within a species. 

The modified fitness function is as follows:

```math
\large{ f_i^{'} = \frac{f_i}{ \sum\limits_{j=1}^n sh(\delta(i,j)) } }
```

Where ```sh``` is the sharing function, this is set to 0 if the distance is above a threshold and 1 otherwise.

This modified fitness function therefore scales an organism's fitness score (specified by the developer) by the number of organisms in the population that are similar to the organism being scored (More organisms that are similar, more the fitness gets penalised).

## Minimizing Dimensionality

Since its harder to find optimal solutions in larger networks due to a bigger search space, NEAT aims to minimize the dimensionality of the models, biasing towards networks with the fewest weights required.

This is done by initialising the population with zero hidden weights, connecting the inputs directly to the outputs. Nodes and connections are then incrementally added to the network during mutation, with only the most useful connections surviving, this helps reduce model size and thus search space, making convergence much more efficient.

