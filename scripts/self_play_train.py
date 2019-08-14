"""
Play script for human players
"""

import argparse
import logging
import os
import sys

from agents.dqn.dqn_agent import DQNAgent
from connect4 import environments
from connect4.settings import PLAYERS

# Append the module to the path so that it can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

LOG = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def parse():
    parser = argparse.ArgumentParser(description='Optionally pass in Connect4 parameters via commandline.')

    parser.add_argument('-x', type=int, default=0, help='X dimension of the board')
    parser.add_argument('-y', type=int, default=0, help='Y dimension of the board')
    parser.add_argument('-lr', type=float, default=.01, help='Learning rate')
    parser.add_argument('-batch', type=int, default=128, help='Minibatch size')
    parser.add_argument('-layers', type=str, default='[10]',
                        help='Number of layers and neurons within them.  Of format '
                             '[100,50] where the len of the list constitutes number'
                             ' of layers and elements correspond to number of neurons.')
    parser.add_argument('-episodes', type=int, default=1, help='Number of episodes to train.')
    parser.add_argument('-debug', type=bool, default=False, help='Activate debug display')
    args = parser.parse_args()
    LOG.debug("Args are: {}".format(args))
    return args.x, args.y, args.lr, args.batch, args.layers, args.episodes, args.debug


def parse_layers(arg):
    layers = []
    arg = arg[1:-1]  # strip []
    neurons = arg.split(",")
    for neuron_count in neurons:
        layers.append(int(neuron_count))
    return layers


if __name__ == '__main__':
    x, y, lr, batch, layers, episodes, debug = parse()
    layers = parse_layers(layers)

    # Change field size if desired
    if x == 0 and y == 0:
        kwargs = {}
    else:
        kwargs = {"x": x, "y": y}



    env = environments.SelfPlayAgentEnvironment(5,7,debug=debug, **kwargs)
    # FIXME
    red_agent = DQNAgent(action_space=env.action_space, max_memory=1000, int_repr=1)#lr=lr, batch=batch, layers=layers)
    blue_agent = DQNAgent(action_space=env.action_space, max_memory=1000, int_repr=2)#lr=lr, batch=batch, layers=layers)
    agents = [red_agent, blue_agent]


    for ep in range(episodes):
        state = env.reset()
        done = False
        while not done:
            for player in agents:
                action = player.choose_action(state)
                next_state, reward, done, info = env.step((action, player.int_repr))

                # Add experience to memory
                player.add_to_memory((state, action, reward, next_state, done))
                player.train(128)

                env.render()
                state = next_state

            # TODO: Saving shouldinvlude the board size
