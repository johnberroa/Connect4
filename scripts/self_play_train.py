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

    env_red = environments.SelfPlayAgentEnvironment(7, 6, debug=debug, **kwargs)
    env_blue = environments.SelfPlayAgentEnvironment(7, 6, debug=debug, **kwargs)
    red_agent = DQNAgent(action_space=env_red.action_space, max_memory=1000,
                         int_repr=1)  # lr=lr, batch=batch, layers=layers)
    blue_agent = DQNAgent(action_space=env_blue.action_space, max_memory=1000,
                          int_repr=2)  # lr=lr, batch=batch, layers=layers)
    agents = [red_agent, blue_agent]
    envs = [env_red, env_blue]

    skip_red = False
    skip_blue = False
    trained = False
    only = True
    for ep in range(episodes):
        state_red = env_red.reset()
        state_blue = env_blue.reset()
        states = [state_red, state_blue]
        done = False

        while True:
            for player in agents:
                valid = False
                while not valid:  # Enforce valid moves
                    action = player.choose_action(states[player.int_repr - 1])
                    valid = envs[player.int_repr - 1].field.check_piece(action)
                next_state, reward, done, info = envs[player.int_repr - 1].step((action, 1))
                if player.int_repr == 1:
                    envs[1].swap_field(envs[0].field.field)
                else:
                    envs[0].swap_field(envs[1].field.field)
                # Add experience to memory
                player.add_to_memory((states[player.int_repr - 1], action, reward, next_state, done))
                if player.int_repr == 1 and not skip_red:
                    trained = player.train(128)
                elif player.int_repr == 2 and not skip_blue:
                    trained = player.train(128)
                else:
                    print("TRAINING PLAYER SKIPPED DUE TO IMBALANCE!!!!!!")

                envs[0].render()
                states[player.int_repr - 1] = next_state
                if trained and only:
                    env_red.red_wins = 0
                    env_red.blue_wins = 0
                    env_blue.red_wins = 0
                    env_blue.blue_wins = 0
                    print("TRAINING HAS BEGUN")
                    only = False  # make this happen only once
                else:
                    print("R{}-{}B".format(envs[0].red_wins, envs[1].red_wins))
                    # Prevent runaway models; all reference red player because that is player 1
                    if env_red.red_wins - env_blue.red_wins > 10 and not skip_red:
                        print("UPDATING BLUE AGENT WEIGHTS ", env_red.red_wins - env_blue.red_wins)
                        blue_agent.policy.model.set_weights(red_agent.policy.model.get_weights())
                        skip_red = True
                    elif env_blue.red_wins - env_red.red_wins > 10 and not skip_blue:
                        print("UPDATING RED AGENT WEIGHTS ", env_blue.red_wins - env_red.red_wins)
                        red_agent.policy.model.set_weights(blue_agent.policy.model.get_weights())
                        skip_blue = True
                    elif np.abs(env_red.red_wins - env_blue.red_wins) <= 10:
                        skip_blue = False
                        skip_red = False

                if done:
                    break
            if done:
                break

    print("END", (time.time() - start) / 60)
    figure, ax = plt.subplots(nrows=2, figsize=(20, 10))
    ax[0].plot(red_agent.policy.losses)
    ax[0].set_title("Red")
    ax[1].plot(blue_agent.policy.losses)
    ax[1].set_title('Blue')
    plt.show()

    # TODO: Saving shouldinvlude the board size
