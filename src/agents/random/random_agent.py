"""
Agent that always chooses randomly
"""

import numpy as np
from agents.abstract_agent import AbstractAgent

class RandomAgent(AbstractAgent):
    """
    An agent that chooses from its possible actions at random
    """

    def __init__(self, action_space):
        """
        Init the random agent
        :param action_space: list of possible actions
        """
        self.action_space = action_space

    def choose_action(self, state):
        """
        Randomly chooses an action from the action space
        :param state: unused; passed because of the interface
        :return:
        """
        return self.action_space[np.random.randint(0, len(self.action_space) - 1)]
