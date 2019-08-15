"""
Deep Q Network
"""
import tensorflow as tf
import keras
from keras.layers import Dense, BatchNormalization, Conv2D, Flatten
from keras.models import Sequential
from keras.optimizers import SGD
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
        self.losses = []

        self.x = x
        self.y = y
        self.num_actions = num_actions

        self.compile()

    def compile(self):
        self.model.add(Conv2D(8, (3,3), padding='same', input_shape=(self.x, self.y, 1), activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(100, activation='relu'))
        # self.model.add(Dense(100, input_dim=self.x * self.y, activation='relu'))
        # self.model.add(BatchNormalization())
        self.model.add(Dense(50, activation='relu'))
        # self.model.add(BatchNormalization())
        self.model.add(Dense(self.num_actions, activation='linear'))

        sgd = SGD(lr=.0001)
        self.model.compile(optimizer=sgd, loss='mae')

    def predict(self, state):
        try:
            states = []
            for s in state:
                states.append(np.expand_dims(np.reshape(s, (self.x, self.y)),2))
            state = np.array(states)
        except:
            state = np.expand_dims(np.reshape(state, (self.x, self.y)), 2)
        if len(state.shape) == 3: # FIXME
            state = np.expand_dims(state, 0)
        try:
            return self.model.predict(state)
        except ValueError:
            print()

    def fit(self, state, q, verbose):
        states = []
        for s in state:
            states.append(np.expand_dims(np.reshape(s, (self.x, self.y)), 2))
            state = np.array(states)
        try:
            loss = self.model.train_on_batch(state, q)
        except ValueError:
            print()
        self.losses.append(loss)
