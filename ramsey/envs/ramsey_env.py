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

import sys

import gym
import matplotlib.pyplot as plt
import networkx
import numpy as np

from ramsey import encoders
from ramsey import reward_functions


class RamseyGame(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, n_nodes, k_clique, save_counterexample=False):
        """Inits the Ramsey Game gym environment."""
        super().__init__()
        self.save_counterexample = save_counterexample
        self.n_nodes = n_nodes
        self.n_edges = int(self.n_nodes * (self.n_nodes - 1) / 2)
        self.k_clique = k_clique
        self.action_dictionary = encoders.graph_hot_encoder_dict(self.n_nodes)

        self.action_space = gym.spaces.MultiBinary(self.n_edges)
        self.observation_space = gym.spaces.MultiBinary(self.n_edges)

    def step(self, action):
        """Performs a step in the environment.

        The step consists of adding an edge to the graph, computting the cliques
        of the graph, giving a reward according to the size of the biggest
        clique and finnally encoding the graph to a binary vector.
        """
        self.graph = networkx.empty_graph(self.n_nodes)
        # The agent has the ability to change the entire graph with one action.
        actions = encoders.one_hot_decode(self.action_dictionary, action)
        for edge in actions:
            self.graph.add_edge(*edge)

        # Get reward and update done.
        reward = self._get_reward()

        # Update observation
        observation = encoders.one_hot_encode(self.action_dictionary,
                                              self.graph.edges)

        info = {}
        return observation, reward, self.done, info

    def reset(self):
        """Resets the environment when a k_clique is found."""
        self.score = 0
        self.done = False

        self.graph = networkx.empty_graph(self.n_nodes)
        self.nodes = list(self.graph.nodes)
        self.edges = np.zeros(self.n_edges, dtype=int)  #list(self.graph.edges)
        self.biggest_clique = 0
        self.previous_biggest_clique = 0

        observation = self.edges
        return observation  # reward, done, info can't be included

    def render(self, mode='human'):
        """Nice visualization of graph."""
        if mode == 'human':
            networkx.draw(self.graph)
            networkx.draw(networkx.complement(self.graph), node_color='r')
            plt.pause(0.1)
            plt.clf()

    def close(self):
        """Acts as a data saver.

        Given a graph with n_nodes we may find that there is a configuration
        where neither the graph, nor it's dual, contain a k_clique. We call this
        configuration a RamseyGame counterexample.
        When a counterexample is found, we save it to a file.

        TODO(@ze): Create a callback function for saving the the graph when the
        model is trained. The callback function should be a input for the
        training function.
        """

        networkx.write_edgelist(self.graph, 'ramsey/graphs/win.txt')
        sys.exit()

    def _get_reward(self):
        """Reward function for k_clique finding.

        Currently this function also handles the logic behind saving a
        RamseyGame counterexample. See self.close() method.

        Ideas to improve this reward function:
        - Penalize non connected graphs.
        - Give reward for finding smaller cliques.
        - Use the stepaction. A action is placing an edge, start counting
        cliques from that edge.
        """
        # Get biggest clique.
        biggest_clique = reward_functions.ramsey_number(self.graph)
        reward = -biggest_clique

        # See self.close() method for TODO comment.
        if biggest_clique <= self.k_clique:
            self.done = True

            if self.save_counterexample:
                if biggest_clique < self.k_clique:
                    self.close()
        return reward
