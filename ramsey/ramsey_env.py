"""Reinforcement Learning gym environment for k_clique finding.

The game consists of two players which take turns placing black and
white edges on a undirected graph. Then, there are three main variants
for this games:
- The player who creates the first k_clique of her color wins.
- The player who creates the first k_clique of her color loses.
- The first player wins if she can create a k_clique, the secondplayer wins if
there are no k_cliques and no more edges to place.

If we are playing the third variant of this game, with a 6 nodes graph
and a 3-clique objective, it is easy to show that player two will
always lose. That's because there will always be a black or 3-clique
in every graph with 6 or more edges.

This is too complicated for the first iteration, so let's try to train
something that creates a k_clique.
"""

import gym
import matplotlib.pyplot as plt
import networkx
import numpy as np


def graph_hot_encoder_dict(n_nodes):
    """A dictionary that encodes integers into graph edges.

    A undirected graph with n nodes has a maximum of n(n-1)/2 edges. So we can
    encode the integers from 1 to n(n-1)/2 has the i-th edge of the graph. This
    is done in the following way:

       int   |   edge
    ---------------------
    0        | (0, 0)
    1        | (0, 1)
    ...
    n-1      | (0,n-1)
    n        | (1, 1)
    ...
    n(n-1)/2 | (n-1, n-1)

    Generating a complete undirected graph with n nodes gives us all the
    necessary edges. The i-th tuple is then accessed by it's index:
    encoder_dictionary[i]
    """
    return list(networkx.complete_graph(n_nodes).edges)


def one_hot_encode(dictionary, graph_edges):
    """One hot encodes a graph into a binary list."""
    return [element in list(graph_edges) for element in dictionary]


class RamseyGame(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}  # Not sure if this needs to be here.

    def __init__(self, n_nodes, k_clique):
        """Inits the Ramsey Game gym environment."""
        super().__init__()
        self.n_nodes = n_nodes
        self.n_edges = int(self.n_nodes * (self.n_nodes - 1) / 2)
        self.k_clique = k_clique
        self.action_dictionary = graph_hot_encoder_dict(self.n_nodes)

        self.action_space = gym.spaces.Discrete(self.n_edges)
        self.observation_space = gym.spaces.MultiBinary(self.n_edges)

    def step(self, action):
        """Performs a step in the environment.

        The step consists of adding an edge to the graph, computting the cliques
        of the graph, giving a reward according to the size of the biggest
        clique and finnally encoding the graph to a binary vector.
        """
        # Place an edge in the graph.
        action_edge = self.action_dictionary[action]
        self.graph.add_edge(*action_edge)

        # Get biggest clique.
        self.biggest_clique = networkx.graph_clique_number(self.graph)

        # Check stopping condition.
        if self.biggest_clique >= self.k_clique:
            self.done = True

        # Get reward.
        self.step_reward = self._get_reward(self.biggest_clique)
        self.reward = self.step_reward + self.previous_reward
        self.previous_reward = self.reward

        # Update observation
        observation = one_hot_encode(self.action_dictionary, self.graph.edges)

        info = {}
        return observation, self.reward, self.done, info

    def reset(self):
        """Resets the environment when a k_clique is found."""
        self.score = 0
        self.done = False
        self.previous_reward = 0

        self.graph = networkx.empty_graph(self.n_nodes)
        self.nodes = list(self.graph.nodes)
        self.edges = np.zeros(self.n_edges, dtype=int)  #list(self.graph.edges)
        self.biggest_clique = 0

        observation = self.edges
        return observation  # reward, done, info can't be included

    def render(self):
        """Nice visualization of graph."""
        networkx.draw(self.graph)
        plt.pause(0.1)
        plt.clf()

    def close(self):
        pass

    def _get_reward(self, size_of_biggest_clique):
        """Reward function for k_clique finding.

        Ideas to improve this reward function:
        - Penalize non connected graphs.
        - Give reward for finding smaller cliques.
        - Use the stepaction. A action is placing an edge, start counting
        cliques from that edge.
        """
        if size_of_biggest_clique >= self.k_clique:
            return 10
        else:
            return -1
