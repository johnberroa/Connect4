"""
Abstract Agent Class to be inherited
"""

from abc import abstractmethod


class AbstractAgent:
    """Abstract agent class"""

    @abstractmethod
    def choose_action(self, state):
        """Given a state, choose an action"""
        raise NotImplementedError
