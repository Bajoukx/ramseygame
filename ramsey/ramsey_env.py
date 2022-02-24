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

import itertools
from multiprocessing.dummy import active_children

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
    necessary edges. The i-th tuple is then accessed by it's index: encoder_dictionary[i]
    """
    return list(networkx.complete_graph(n_nodes).edges)

def hot_encode(dictionary, graph_edges):
    return [element in list(graph_edges) for element in  dictionary]

class RamseyGame(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}  # Not sure if this needs to be here.

    def __init__(self, n_nodes, k_clique):
        """Inits the Ramsey Game gym environment."""
        super(RamseyGame, self).__init__()
        self.n_nodes = n_nodes
        self.n_edges = int(self.n_nodes * (self.n_nodes - 1) / 2)
        self.k_clique = k_clique
        self.action_dictionary = graph_hot_encoder_dict(self.n_nodes)

        self.action_space = gym.spaces.Discrete(self.n_edges)
        self.observation_space = gym.spaces.MultiBinary(self.n_edges)

        self.reset()

    def step(self, action):
        # Place an edge.
        action_edge = self.action_dictionary[action]
        self.graph.add_edge(*action_edge)

        

        # 
        self.previous_cliques = self.cliques
        self.cliques = networkx.cliques_containing_node(
            self.graph, action_edge, self.previous_cliques)
        biggest_clique = max(self.cliques, default=0)

        # Check stopping condition.
        if biggest_clique >= self.k_clique:
            self.done = True

        # Get reward.
        self.step_reward = self._get_reward(biggest_clique)
        self.reward = self.step_reward + self.previous_reward
        self.previous_reward = self.reward

        # Update observation
        observation = hot_encode(self.action_dictionary, self.graph.edges)

        info = {}
        return observation, self.reward, self.done, info

    def reset(self):
        self.score = 0
        self.done = False
        self.previous_reward = 0

        self.graph = networkx.empty_graph(self.n_nodes)
        self.nodes = list(self.graph.nodes)
        self.edges = np.zeros(self.n_edges, dtype=int) #list(self.graph.edges)
        self.cliques = []

        observation = self.edges
        return observation  # reward, done, info can't be included

    def render(self, mode='human'):
        """Nice visualization of graph."""
        networkx.draw(self.graph)
        plt.show()

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

        #return self._size_biggest_clique() - self.n_nodes
        # if self._size_biggest_clique() >= self.k_clique:
        #     return 10
        # else:
        #     return -1
