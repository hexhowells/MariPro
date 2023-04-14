# NEAT

NeuroEvolution of Augmenting Topologies (NEAT) expands on neuroevolution by adding evolving network topologies along side evolving network weights. This method aims to solve the *Competing Conventions Problem* by enabling effective crossover of neuroevolution systems without destroying useful information and structures.

---

## Encoding

NEAT encodes the genome into two parts, Node Genes and Connect Genes. 

Node Genes represent each node in the neural network and include metadata about the node index and position in the network (input, hidden, output).

Connect Genes represent the connections between nodes, metatdata specifies the following:
- in and out connection indexes
- weight value 
- enabled, flag that represents if the node is active in the network
- innovation, value that keeps track of historical information about the gene, used for *artifical synapsis*

## Mutation

Mutation can occur at both the weight connections and the network structure.

Weight mutation is implemented the same as with other NeuroEvolution methods.

Topology mutation can occur in two ways:

#### Add connection
- Add a new connection gene between two unconnected nodes.


#### Add node
- Add a new node gene. 
- Split a connection gene by disabling the connection and adding two new connections between the split nodes. the incoming weight is set to 1, the outgoing weight is set to the same as the old weight.
