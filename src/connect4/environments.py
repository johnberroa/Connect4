"""
Game environments that allow Connect4 to be played
"""

import logging
import sys
import time

import numpy as np
from gym import Env, spaces

from .board import Field
from .settings import PLAYERS, PLAYER_MAP, WIN_REWARD, TIE_REWARD

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


class SelfPlayAgentEnvironment(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, x, y, debug=False):
        super().__init__()
        self.field = Field(x, y)

        self.x = x
        self.y = y

        self.reward_range = (0, 1)
        self.action_space = spaces.Discrete(x)
        self.observation_space = spaces.Box(low=0, high=2, shape=(x* y,), dtype=np.int16)  # flattened representation

        self.history = []
        self.red_wins = 0
        self.blue_wins = 0
        self.steps_taken = 0
        self.debug = debug
        LOG.debug("Agent environment initialized")

    def _append_history(self):
        """Appends the current state of the board to the history"""
        self.history.append(self.field.display(self.debug))

    def step(self, action_tuple):
        """
        Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        start = time.time()
        action = action_tuple[0]
        assert action in self.action_space, "Invalid action selected!"
        player = action_tuple[1]
        print("Making move for player %s", PLAYER_MAP[player])
        print(self.field.colored_display)
        reward = 0
        self._append_history()

        result = self.field.place_piece(action, 1)
        if result == 1:
            print("Player {} wins the round!".format(PLAYER_MAP[player]))
            reward = WIN_REWARD
            self.reset()
        elif result == 2:
            print("Tie!")
            reward = TIE_REWARD
        end = time.time()
        print("Step time: {}s\n\n\n".format(round((end - start), 2)))

        # For clarity
        observation = self.field.flattened_field
        done = True if result in [1, 2] else False
        self.steps_taken += 1
        return observation, reward, done, {"Red Wins": self.red_wins, "Blue Wins": self.blue_wins,
                                           "Steps Taken": self.steps_taken}

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        Returns:
            observation (object): the initial observation.
        """
        self.steps_taken = 0
        self.history = []
        self.field.new_field()
        return self.field.flattened_field

    def render(self, mode='human'):
        """Render for visualization"""
        print(self.field.display(self.debug))

    def close(self):
        """Stops the environment"""
        self.field = None

# Heuristic?
# Plotting
# Experiments on transfer learning on size (need to not use all dense then)
