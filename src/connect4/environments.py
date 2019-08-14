"""
Game environments that allow Connect4 to be played
"""

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

        self.history = []
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

    def _append_history(self):
        """Appends the current state of the board to the history"""
        self.history.append(self.field.display(self.debug))

    def play_rounds(self):
        """Plays a round"""
        LOG.debug("Starting new round...")
        start = time.time()

        # Play the round
        done = False
        print(self.field.display(self.debug))
        while True:
            for player in PLAYERS:
                self._append_history()
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
            self._append_history()
            print("Player 'r' wins the entire match!\n\n\n")
            self.new_game()
        elif self.blue_wins >= self.max_score:
            self._append_history()
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

    @property
    def display_history(self):
        print("History of the round:")
        for board in self.history:  # TODO: Make this flush the std with controls to go to next frame or previous
            print(board)
        # Controls something along the line of...
        # index = 0
        # choice = input("Previous or next? (p/n)")
        # if p: index-=1
        # elif n: index+=1
        # print(self.history[index])


class AgentEnvironment(Env):

    def __init__(self, agent, x, y, debug=False):
        self.field = Field(x, y)
        self.agent = agent

        self.x = x
        self.y = y

        self.history = []
        self.debug = debug
        LOG.debug("Agent environment initialized")

    def _append_history(self):
        """Appends the current state of the board to the history"""
        self.history.append(self.field.display(self.debug))

    def step(self, player, action):
        """Plays a round"""
        LOG.debug("Starting new round...")
        start = time.time()

        self._append_history()
        result = self.field.place_piece(action, player)
        if result == 1:
            print("Player {} wins the round!".format(player))
        elif result == 2:
            print("Tie!")
        end = time.time()
        print("Round time: {}m\n\n\n".format(round((end - start) / 60, 2)))
        return result

    def reset(self):
        self.field.new_field()

    def render(self, mode='human'):
        print(self.field.display(self.debug))

# Heuristic?
# Plotting
# Experiments on transfer learning on size (need to not use all dense then)
