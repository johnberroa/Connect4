"""
Play script for human players
"""

import argparse
import logging
import os
import sys
import numpy as np

import matplotlib.pyplot as plt
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
    parser.add_argument('-episodes', type=int, default=1000, help='Number of episodes to train.')
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

    import time
    start = time.time()

    env = environments.SelfPlayAgentEnvironment(7,6,debug=debug, **kwargs)
    # FIXME
    red_agent = DQNAgent(action_space=env.action_space, max_memory=1000, int_repr=1)#lr=lr, batch=batch, layers=layers)
    blue_agent = DQNAgent(action_space=env.action_space, max_memory=1000, int_repr=2)#lr=lr, batch=batch, layers=layers)
    agents = [red_agent, blue_agent]

    skip_red = False
    skip_blue = False
    trained = False
    only = True
    for ep in range(episodes):
        state = env.reset()
        done = False

        while True:
            for player in agents:
                valid = False
                while not valid:  # Enforce valid moves
                    action = player.choose_action(state)
                    valid = env.field.check_piece(action)
                next_state, reward, done, info = env.step((action, player.int_repr))

                # Add experience to memory
                player.add_to_memory((state, action, reward, next_state, done))
                print("VEFORE TRAIN: SR {} SB {}".format(skip_red, skip_blue))
                if player.int_repr == 1 and not skip_red:
                    trained = player.train(128)
                elif player.int_repr == 2 and not skip_blue:
                    trained = player.train(128)
                else:
                    print("TRAINING PLAYER SKIPPED DUE TO INBALANCE!!!!!!")

                env.render()
                state = next_state
                if trained and only:
                    env.red_wins = 0
                    env.blue_wins = 0
                    print("TRAINING HAS BEGUN")
                    only = False # make this happen only once
                else:
                    print("IN CONDITIONAL")
                    print("R{}-{}B".format(env.red_wins, env.blue_wins))
                    # Prevent runaway models
                    if env.red_wins - env.blue_wins > 10 and not skip_red:
                        print("UPDATING BLUE AGENT WEIGHTS ", env.red_wins - env.blue_wins)
                        blue_agent.policy.model.set_weights(red_agent.policy.model.get_weights())
                        skip_red = True
                    elif env.blue_wins - env.red_wins > 10 and not skip_blue:
                        print("UPDATING RED AGENT WEIGHTS ", env.blue_wins - env.red_wins)
                        red_agent.policy.model.set_weights(blue_agent.policy.model.get_weights())
                        skip_blue = True
                    elif np.abs(env.red_wins - env.blue_wins) <= 10:
                        skip_blue = False
                        skip_red = False

                if done:
                    break
            if done:
                break

    print("END", (time.time() - start) / 60)
    figure, ax = plt.subplots(nrows=2, figsize=(20,10))
    ax[0].plot(red_agent.policy.losses)
    ax[0].set_title("Red")
    ax[1].plot(blue_agent.policy.losses)
    ax[1].set_title('Blue')
    plt.show()

            # TODO: Saving shouldinvlude the board size
