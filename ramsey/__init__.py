"""Registers the environment with gym."""

from gym.envs.registration import register

register(
    id='ramsey',
    entry_point='ramsey.envs:RamseyGame',
)
