"""
Play script for human players
"""

import argparse
import logging
import os
import sys

# Append the module to the path so that it can be imported
from connect4.settings import PLAYERS

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from connect4 import environments

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

    # FIXME
    red_agent = Agent(lr=lr, batch=batch,layers=layers)
    blue_agent = Agent(lr=lr, batch=batch, layers=layers)
    agents = [red_agent, blue_agent]

    env = environments.SelfPlayAgentEnvironment(debug=debug, **kwargs)

    for ep in range(episodes):
        state = env.reset()
        done = False
        while not done:
            for player in PLAYERS:
                action = agents[player].choose_action(state)
                state, reward, done, info = env.step((action, player))

                # Add experience to memory
                agents[player].add_to_memory(??)



    #TODO: Saving shouldinvlude the board size