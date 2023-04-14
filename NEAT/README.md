# NEAT

NeuroEvolution of Augmenting Topologies (NEAT) expands on neuroevolution by adding evolving network topologies along side evolving network weights. This method aims to solve the *Competing Conventions Problem* by enabling effective crossover of neuroevolution systems without destroying useful information and structures.

---

NEAT encodes the genome into two parts, Node Genes and Connect Genes. 

Node Genes represent each node in the neural network and include metadata about the node index and position in the network (input, hidden, output).

Connect Genes represent the connections between nodes, metatdata specifies the following:
- in and out connection indexes
- weight value 
- enabled, flag that represents if the node is active in the network
- innovation, value that keeps track of historical information about the gene, used for *artifical synapsis*

