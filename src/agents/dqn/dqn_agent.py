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

        self.policy = DQN(7,6,7)
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

    def _unpack_batch(self, batch):
        state = []
        action = np.zeros(len(batch))
        reward = np.zeros(len(batch))
        next_state = []
        done = np.zeros(len(batch))

        for i, element in enumerate(batch):
            state.append(element[0])
            action[i] = element[1]
            reward[i] = element[2]
            next_state.append(element[3])
            done[i] = element[4]

        return np.array(state), action, reward, np.array(next_state), done

    def _apply_q_update(self, q_vals, q_update, actions):
        for i in range(len(q_vals)):
            q_vals[int(actions[i])] += q_update[i]
        return q_vals

    def train(self, batchsize):
        batch = self._sample_from_memory(batchsize)
        if batch:
            # Unpack the batch
            state, action, reward, next_state, done = self._unpack_batch(batch)
            q_update = reward
            if 1 in done:  # prevent empty array updates when there are no dones
                q_update[done==True] = (reward[done==True] + self.gamma * np.amax(self.policy.predict(next_state[done==True])))
            q_values = self.policy.predict(state)
            q_values = self._apply_q_update(q_values, q_update, action)
            self.policy.fit(state, q_values, verbose=0)  #FIXME to tensorflow
            return True
        else:
            return False

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
