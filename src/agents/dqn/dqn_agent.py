"""
Agent that uses DQN as its action sampler
"""

from agents.abstract_agent import AbstractAgent

from .model import DQN

class DQNAgent(AbstractAgent):
    def __init__(self, action_space, name="DQN"):
        self.action_space = action_space
        self.name = name

    def choose_action(self, state):
        pass
