"""
Agent that uses DQN as its action sampler
"""
import logging
import random
import sys
from collections import deque

import numpy as np

from agents.abstract_agent import AbstractAgent
from .model import DQN

LOG = logging.getLogger(__name__)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)




class DQNAgent(AbstractAgent):
    def __init__(self, action_space, max_memory, int_repr, gamma=.99, eps_greedy=.1 ,name="DQN"):
        self.action_space = action_space
        self.name = name
        self.int_repr = int_repr

        self.policy = DQN(5,7,5)
        self.memory = deque(maxlen=max_memory)
        self.gamma = gamma

        self.eps_greedy = eps_greedy

    def add_to_memory(self, state_transition):
        self.memory.append(state_transition)

    def _sample_from_memory(self, batchsize):
        if len(self.memory) < batchsize:
            LOG.warning("Attempting to sample from a memory who's size < batchsize! Skipping...")
            return False
        else:
            return random.sample(self.memory, batchsize)


    def train(self, batchsize):
        batch = self._sample_from_memory(batchsize)
        if batch:
            # Unpack the batch
            state, action, reward, next_state, done = batch
            if not done:
                q_update = (reward + self.gamma * np.amax(self.policy.predict(next_state)))
            else:
                q_update = reward
            q_values = self.policy.predict(state)
            q_values[action] = q_update
            self.policy.fit(state, q_values, verbose=0)  #FIXME
        else:
            pass

    def choose_action(self, state):
        greedy = True if np.random.random() < self.eps_greedy else False
        if greedy:
            action = self.action_space.sample()
        else:
            q_values = self.policy.predict(state)
            action = np.argmax(q_values)

        return action

    def summary(self):
        pass

    def save(self):
        pass

    def load(self):
        pass
