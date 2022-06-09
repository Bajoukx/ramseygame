"""Registers the ramsey environment with gym."""

from gym.envs.registration import register

register(
    id='RamseyGame-v0',
    entry_point='ramsey.envs:RamseyGame',
)
