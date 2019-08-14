"""
Play script for human players
"""

import argparse
import logging
import os
import sys

# Append the module to the path so that it can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from connect4 import environments

LOG = logging.getLogger(__name__)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def parse():
    parser = argparse.ArgumentParser(description='Optionally pass in Connect4 parameters via commandline.')

    parser.add_argument('--max_score', type=int, default=0, help="Max score to win the entire game.")
    parser.add_argument('-x', type=int, default=0, help='X dimension of the board')
    parser.add_argument('-y', type=int, default=0, help='Y dimension of the board')
    parser.add_argument('-debug', type=bool, default=False, help='Activate debug display')
    args = parser.parse_args()
    LOG.debug("Args are: {}".format(args))
    return args.x, args.y, args.max_score, args.debug


if __name__ == '__main__':
    x, y, max_score, debug = parse()

    # Change field size if desired
    if x == 0 and y == 0:
        kwargs = {}
    else:
        kwargs = {"x": x, "y": y}
    # Change max score if desired
    if max_score == 0:
        max_score = 3

    env = environments.HumanEnvironment(max_score=max_score, debug=debug, **kwargs)
    env.play_match()
    env.display_history
