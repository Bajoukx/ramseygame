"""In this file we test dgl capabilities"""

import dgl

# TODO: one-hot-encode edges names and pass heterograph successors to a tensor of size (-)*num_nodes
# TODO: create function to go from k-cliques to (k+1)-cliques

if __name__ == '__main__':

    # define a random graph with 43 vertices
    g = dgl.rand_graph(43, 450)

    # remove self-loops
    g = dgl.remove_self_loop(g)
    print(g.edges())

    # transform it into an undirected graph
    bg = dgl.to_bidirected(g)
    print(bg.edges())

    # define a heterograph
    hg = dgl.heterograph(
        {('node', 'is_part', '1-clique'): (g.edges()[0], g.edges()[1])}
    )
    print(hg.nodes('node'))
    print(hg.nodes('1-clique'))
    print(hg.successors(0, etype='is_part'))
    print(hg.successors(1, etype='is_part'))
