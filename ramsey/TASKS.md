
# Tasks 

In this file there is a short description of the goals of this project. The 
main purpose is to prove that the answer to the ultimate question of life is 42.

Before proving such result some groundwork has to be performed. This work 
will contain results by itself.

## Definitions

### k-clique
Given a graph G, a k-clique of G is a complete subgraph of G with k nodes.

### The question
Given a positive integer k, what is the maximal number n such that, at a party 
with n persons, it is possible to have k persons that all know each other or 
k persons that do not know any of the (k-1) others?

In graph terms, what is the maximal size of a graph such that the graph or 
its complement does not contain a k-clique?

### Ramsey Number

Given a positive integer k, we call the Ramsey number of k, denoted by R(k), 
the number that is the answer to **_the question_**.

## First conjecture

For each graph G with n nodes, denote by $C_G$ the tensor with size 
$n \times 2 \times n-1$
where the entry $i, 0, k$ is the number of k cliques that the node i is part 
of and $i, 1, k$ is the number of k cliques that the node i is part in the 
complement of the graph G.

**Conjecture**: It is possible to recover the graph G from $C_G$.

### Quick computation of $C_G$

There is a method to compute the clique tensor of a graph that uses the 
adjacency matrix multiplication, but it is slow. The method implemented now 
is a message passing algorithm using lists that can be improved. As an 
exercise it would also be interesting to create a graph neural network to 
predict the tensor and see how well it fares.

## Detecting if a tensor is the clique tensor of a graph

This question comes in the spirit of determining the topology of the tensors 
that correspond to $C_G$ for a graph G. This could improve the search for 
graphs that do not have k-cliques for large k.

### Encoder-decoder

#### Clique tensor -> Graph -> Clique tensor

One idea is to go from clique tensor to graph to clique tensor. The accuracy 
needs to be measured ignoring the order of the vertices (it can be aligned 
by k-clique size, splitting ties with the next k+1-cliques).

The problems arising with the encoder is that equal number of cliques for 
two distinct nodes does not imply, at all, that the nodes should be 
connected. Furthermore, if one takes the example of a cycle-graph, we have 
the same number of k-cliques for all nodes but there need to be some 
randomness in choosing which edges to select.

#### Cliques tensor -> Embedding -> Clique tensor

A possible approach is to use a model similar to the one used in transformer 
models (e.g., BERT and RoBERTa). In particular, one can simply mask some 
entries of a clique tensor and train the model to determine them again. Then 
one would use the embedding to directly determine the corresponding 
graph. 

### Auto Generative network

## Pushing the bounds upward

It is known that $42 \leq R(5) \leq 47$. As claimed in the introduction, the 
goal is to prove that R(5)=42 but this might not be true. If we manage to 
find a graph having 43 vertices but no 5-clique in itself or its complement, 
then we find tighter bounds.

### Reinforcement learning

## Finding inequalities

The first relation one sees when looking at the clique tensor of a graph is 
that the 2-cliques add to an even number. Moreover, the 2-cliques of a node 
plus the 2-cliques on the complement graph add to the number of nodes of the 
graph. 
These are elementary relations, but having a model looking at a large dataset 
of graphs and clique tensors might be enough to provide us relations of the 
type: 

**condition(k-cliques) => \sum (k+1)-cliques > 0**

### Data analysis

### Reinforcement learning

## Adjacency matrix properties

Another point of view worth exploring is the following, given a graph G with 
n nodes, is there a relation between the first n powers of the adjacency 
matrix of G and its clique tensor?

There obviously is, the adjacency matrix completely defines the graph. But 
how much do we have to take from the powers of the matrix? 

**Conjecture**: Is it enough to look at the diagonal of the n first powers of 
the adjacency matrix of $G$ to completely describe $C_G$.
