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
import networkx


class RamseyGame(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}  # Not sure if this needs to be here.

    def __init__(self, n_nodes, k_clique):
        """Inits the Ramsey Game gym environment."""
        super(RamseyGame, self).__init__()
        self.n_nodes = n_nodes
        n_edges = self.n_nodes * (self.n_nodes - 1) / 2
        self.k_clique = k_clique
        self.action_dictionary = list(networkx.complete_graph(self.n_nodes).edges)

        self.action_space = gym.spaces.Discrete(n_edges)
        self.observation_space = gym.spaces.Tuple(
            (gym.spaces.Discrete(self.n_nodes),
             gym.spaces.Discrete(self.n_nodes)))  # This has double of the
        # necessary space.

    def step(self, action):
        networkx.draw(self.graph)

        # Place an edge
        action_edge = self.action_dictionary[action]
        #print(action_edge)
        self.graph.add_edge(*action_edge)

        # Check stopping condition.
        if self._size_biggest_clique() >= self.k_clique:
            self.done = True

        self.reward = self._get_reward()
        observation = list(self.graph.nodes)
        info = {}
        return observation, self.reward, self.done, info

    def reset(self):
        self.score = 0
        self.done = False

        self.graph = networkx.empty_graph(self.n_nodes)
        self.nodes = list(self.graph.nodes)
        self.edges = list(self.graph.edges)
        observation = self.edges
        return observation  # reward, done, info can't be included

    def render(self, mode='human'):
        """Nice visualization of graph."""
        pass

    def close(self):
        ...

    def _size_biggest_clique(self):
        cliques = networkx.find_cliques(self.graph)
        cliques_size = [len(clique) for clique in cliques]
        return max(cliques_size)

    def _get_reward(self):
        """Reward function for k_clique finding.

        Ideas to improve this reward function:
        - Penalize non connected graphs.
        - Give reward for finding smaller cliques.
        - Use the stepaction. A action is placing an edge, start counting
        cliques from that edge.
        """
        if self._size_biggest_clique() >= self.k_clique:
            return 1
        else:
            return -1
