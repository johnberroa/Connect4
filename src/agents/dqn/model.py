"""
Deep Q Network
"""
import tensorflow as tf
import keras
from keras.layers import Dense
from keras.models import Sequential
# class DQN:
#     def __init__(self):
#         self.losses = []
#
#     def compile(self):
#         pass
#
#     def fit(self):
#         pass
#
#     def predict(self, state):
#         pass
#
#     def summary(self):
#         pass
#
#     def save(self):
#         pass
#
#     def load(self):
#         pass
#
#     def plot_curves(self):
#         pass
import numpy as np

class DQN:
    def __init__(self, x, y, num_actions):
        self.model = Sequential()

        self.x = x
        self.y = y
        self.num_actions=num_actions

        self.compile()

    def compile(self):
        self.model.add(Dense(100, input_dim=self.x*self.y, activation='relu'))
        self.model.add(Dense(self.num_actions, activation='linear'))

        self.model.compile(optimizer='adam', loss='mse')

    def predict(self, state):
        self.model.predict(np.expand_dims(state,0))

    def fit(self, state, q, verbose):
        self.model.train_on_batch(state, q)

