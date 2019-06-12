import argparse
import logging
import sys

from connect4 import environments

LOG = logging.getLogger(__name__)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def parse():
    parser = argparse.ArgumentParser(description='Optionally pass in Connect4 parameters via commandline.')

    parser.add_argument('-x', type=int, default=0, help='X dimension of the board')
    parser.add_argument('-y', type=int, default=0, help='Y dimension of the board')
    args = parser.parse_args()
    LOG.debug("Args are: {}".format(args))
    return args.x, args.y


if __name__ == '__main__':
    x, y = parse()
    if x == 0 and y == 0:
        kwargs = {}
    else:
        kwargs = {"x": x, "y": y}

    env = environments.HumanEnvironment(max_score=2, **kwargs)
    env.play_match()
