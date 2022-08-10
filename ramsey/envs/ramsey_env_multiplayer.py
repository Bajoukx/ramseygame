"""Reinforcement Learning gym environment for k_clique finding."""

import sys

from absl import logging

import gym
import matplotlib.pyplot as plt
import networkx
import numpy as np

from ramsey import encoders
from ramsey import reward_functions


class RamseyGameMultiplayer(gym.Env):
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

        self.action_space = gym.spaces.Discrete(self.n_edges)
        self.observation_space = gym.spaces.MultiBinary(self.n_edges)

    def _place_edge(self, action):
        """Places an edge in the graph for the current player."""
        action_edge = self.action_dictionary[action]
        print('action_edge: %s', action_edge)
        if self.current_player == 1:
            print('player 1 previous graph: %s', list(self.graph.edges))
            self.graph.add_edge(*action_edge, player=1)
            print('player 1 following graph: %s', list(self.graph.edges))
        else:
            print('player 2 previous graph: %s', list(self.graph.edges))
            self.graph.add_edge(*action_edge, player=2)
            print('player 2 following graph: %s', list(self.graph.edges))

        # Check winning conditions.
        current_player_edge_list = []
        for edge in self.graph.edges:
            if self.graph.get_edge_data(*edge)['player'] == self.current_player:
                current_player_edge_list.append(edge)
        
        subgraph = networkx.Graph()
        subgraph.add_edges_from(current_player_edge_list)
        if networkx.graph_clique_number(subgraph) >= self.k_clique:
            self.done = True
            print('win lets go')

    def step(self, action):
        """Performs a step in the environment.

        The step consists of adding an edge to the graph, computting the cliques
        of the graph, giving a reward according to the size of the biggest
        clique and finnally encoding the graph to a binary vector.
        """
        self._place_edge(action)
        if self.current_player == 1:
            self.current_player == 2
        else:
            self.current_player == 1
        # self.current_step += 1
        #print('graph: %s', list(self.graph.edges))

        # Get reward and update done.
        reward = self._get_reward()

        # Update observation
        observation = encoders.one_hot_encode(self.action_dictionary,
                                              self.graph.edges)
        print('observation: %s', observation)
        info = {}
        return observation, reward, self.done, info

    def _reset_players_score(self):
        """Resets the players score."""
        self.player_1_score = 0
        self.player_2_score = 0


    def reset(self):
        """Resets the environment when a k_clique is found."""
        self.current_player = 1
        self.current_step = 0

        self._reset_players_score()
        self.done = False

        self.graph = networkx.empty_graph(self.n_nodes)
        self.nodes = list(self.graph.nodes)
        self.edges = np.zeros(self.n_edges, dtype=int)  #list(self.graph.edges)
        self.biggest_clique = 0
        self.previous_biggest_clique = 0

        observation = self.edges
        print('reset done')
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
        file_name = 'ramsey/graphs/win_' + str(self.k_clique) + '_clique_' + \
            str(self.n_nodes) + '_nodes.txt'
        networkx.write_edgelist(self.graph, file_name)
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
