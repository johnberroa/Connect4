import logging
import sys
import time

from gym import Env

from .board import Field
from .settings import PLAYERS

LOG = logging.getLogger(__name__)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class HumanEnvironment:
    def __init__(self, max_score, x=7, y=5, debug=False):
        self.field = Field(x, y)
        self.debug = debug

        self.max_score = max_score

        self.red_wins = 0
        self.blue_wins = 0

        LOG.debug("Human environment initialized")

    def reset(self):
        """Reset the field for a new game"""
        self.field.new_field()
        LOG.debug("Field reset.")

    def new_game(self):
        """Resets everything for a new game"""
        self.red_wins = 0
        self.blue_wins = 0
        LOG.debug("Game reset.")

    def play_rounds(self):
        """Plays a round"""
        LOG.debug("Starting new round...")
        start = time.time()

        # Play the round
        done = False
        print(self.field.display(self.debug))
        while True:
            for player in PLAYERS:
                result = False
                while not result:
                    move = input("Where do you want to move, Player {}?: ".format(player))
                    result = self.field.place_piece(move, player)
                print(self.field.display(self.debug))
                if result == 1:
                    print("Player {} wins the round!".format(player))
                    if player == 'r':
                        self.red_wins += 1
                    else:
                        self.blue_wins += 1
                    done = True
                elif result == 2:
                    print("Tie!")
                    done = True
                if done: break
            if done: break
        end = time.time()
        print("Round time: {}m\n\n\n".format(round((end - start) / 60, 2)))

        # Check if the overall game is over
        if self.red_wins >= self.max_score:
            print("Player 'r' wins the entire match!\n\n\n")
            self.new_game()
        elif self.blue_wins >= self.max_score:
            print("Player 'b' wins the entire match!\n\n\n")
            self.new_game()
        else:
            self.reset()
            self.play_rounds()

    def play_match(self):
        """Plays rounds until the max score is met"""
        start = time.time()
        self.play_rounds()
        end = time.time()
        print("Match time: {}m".format(round((end - start) / 60, 2)))

        cont = None
        # Make sure there is valid input
        while cont is None:
            cont = input("New game? (y/n): ")
            if cont.lower() not in ['y', 'n']:
                cont = None

        if cont.lower() == "y":
            self.new_game()
            self.play_match()
            self.red_wins = 0
            self.blue_wins = 0
        else:
            print("Thank you for playing!")


class RLEnvironment(Env):

    def __init__(self):
        pass

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass

# Heuristic?
# Plotting
# Experiments on transfer learning on size (need to not use all dense then)
