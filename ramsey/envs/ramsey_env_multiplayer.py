"""Reinforcement Learning gym environment for k_clique finding."""

import sys

from absl import logging

import gym
import matplotlib.pyplot as plt
import networkx
import numpy as np

from ramsey import encoders


class RamseyGameMultiplayer(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, config):
        """Inits the Ramsey Game gym environment."""
        super().__init__()

        # TODO: Create a fonfiguration file where the environment parameters
        # are saved.
        self.config = config
        self.action_dictionary = encoders.graph_hot_encoder_dict(self.config.n_nodes)

        self.agents = ['player_1', 'player_2']

        self.action_space = gym.spaces.Discrete(self.config.n_edges)
        self.observation_space = gym.spaces.MultiBinary(self.config.n_edges)

    def _place_edge(self, action):
        """Places an edge in the graph for the current player."""
        self.previous_graph = self.graph

        action_edge = self.action_dictionary[action]
        logging.debug('action_edge: %s', action_edge)
        if self.current_player == 1:
            logging.debug('player 1 previous graph: %s', list(self.graph.edges))
            self.graph.add_edge(*action_edge, player=1)
            logging.debug('player 1 following graph: %s', list(self.graph.edges))
        else:
            logging.debug('player 2 previous graph: %s', list(self.graph.edges))
            self.graph.add_edge(*action_edge, player=2)
            logging.debug('player 2 following graph: %s', list(self.graph.edges))

    def step(self, action):
        """Performs a step in the environment.

        The step consists of adding an edge to the graph, computting the cliques
        of the graph, giving a reward according to the size of the biggest
        clique and finnally encoding the graph to a binary vector.
        """
        self._place_edge(action)
        logging.debug('current_player: %s', self.current_player)
        if self.current_player == 1:
            self.current_player = 2
        else:
            self.current_player = 1
        # self.current_step += 1
        #print('graph: %s', list(self.graph.edges))

        # Get reward and update done.
        reward = self._get_reward()

        # Update observation
        observation = encoders.one_hot_encode(self.action_dictionary,
                                              self.graph.edges)
        observation = np.array(observation)
        logging.debug('observation: %s', observation)
        info = {}
        #self.render()
        return observation, reward, self.done, info

    def _reset_players_score(self):
        """Resets the players score."""
        self.player_1_score = 0
        self.player_2_score = 0


    def reset(self):
        """Resets the environment when a k_clique is found."""
        self.current_player = 1
        self.current_step = 0

        self.reward = 0
        self._reset_players_score()
        self.player_biggest_clique = 0
        self.done = False

        self.graph = networkx.empty_graph(self.config.n_nodes)
        self.previous_graph = networkx.empty_graph(self.config.n_nodes)
        self.nodes = list(self.graph.nodes)
        self.edges = np.zeros(self.config.n_edges, dtype=int)  #list(self.graph.edges)
        self.biggest_clique = 0
        self.previous_biggest_clique = 0

        observation = self.edges
        logging.debug('reset done')
        return observation  # reward, done, info can't be included

    def render(self, mode='human'):
        """Nice visualization of graph."""
        if mode == 'human':
            logging.debug('edges: %s', self.graph.edges.data())
            colors = []
            for edge in self.graph.edges:
                if self.graph.get_edge_data(*edge)['player'] == 1:
                    colors.append('red')
                else:
                    colors.append('green')
            networkx.draw(self.graph, edge_color=colors)
            #networkx.draw(networkx.complement(self.graph), node_color='r')
            plt.pause(1)
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
        file_name = 'ramsey/graphs/win_' + str(self.config.k_clique) + '_clique_' + \
            str(self.config.n_nodes) + '_nodes.txt'
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
        # Check winning conditions.
        current_player_edge_list = []
        for edge in self.graph.edges:
            if self.graph.get_edge_data(*edge)['player'] == self.current_player:
                current_player_edge_list.append(edge)

        subgraph = networkx.Graph()
        subgraph.add_edges_from(current_player_edge_list)

        previous_player_biggest_clique = self.player_biggest_clique
        self.player_biggest_clique = networkx.graph_clique_number(subgraph)

        self.reward -= 1
        reward = self.reward

        # Penalty for not adding an edge.
        if self.previous_graph.number_of_edges() == self.graph.number_of_edges():
            self.done = True
            reward -= self.config.n_edges * 10

        if self.player_biggest_clique >= self.config.k_clique:
            logging.debug('Player %s has found a clique of size %s', self.current_player, self.player_biggest_clique)
            self.done = True
            logging.debug('game finished')

        # Check problem solved conditions.
        if self.config.save_counterexample:
            if previous_player_biggest_clique < self.config.k_clique and \
                    self.player_biggest_clique < self.config.k_clique and \
                        self.graph.number_of_edges == self.config.n_edges:
                self.close()
        return reward
