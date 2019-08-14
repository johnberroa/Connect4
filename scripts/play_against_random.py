"""
Play script for playing against a random player
"""

import argparse
import logging
import os
import sys

# Append the module to the path so that it can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from connect4 import environments
from agents.random.random_agent import RandomAgent

LOG = logging.getLogger(__name__)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def parse():
    parser = argparse.ArgumentParser(description='Optionally pass in Connect4 parameters via commandline.')

    parser.add_argument('--max_score', type=int, default=0, help="Max score to win the entire game.")
    parser.add_argument('-x', type=int, default=0, help='X dimension of the board')
    parser.add_argument('-y', type=int, default=0, help='Y dimension of the board')
    args = parser.parse_args()
    LOG.debug("Args are: {}".format(args))
    return args.x, args.y, args.max_score


if __name__ == '__main__':
    x, y, max_score = parse()

    # Change field size if desired
    if x == 0 and y == 0:
        kwargs = {}
    else:
        kwargs = {"x": x, "y": y}
    # Change max score if desired
    if max_score == 0:
        max_score = 3

    agent = RandomAgent(list(range(y)))
    env = environments.AgentEnvironment(agent, **kwargs)

    result = False
    while not result:
        for player in PLAYERS
        env.step()
