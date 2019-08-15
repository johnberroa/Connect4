"""
Connect 4 board with score checking
"""
import logging, sys
import numpy as np
from colorama import Fore, Style

from .settings import WIN_CONDITIONS, PLAYERS, PLAYER_MAP

LOG = logging.getLogger(name=__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class Field:
    """
    Connect4 Field represented as a list of lists.
    """

    def __init__(self, x=7, y=5):
        """
        Initalize the playing field
        :param x: # columns
        :param y: # rows
        """
        assert x > 0 and y > 0, "X and Y must be greater than 0!"
        assert x >= 4 or y >= 4, "X or Y must be greater than or equal to 4 in order to meet the win condition!"
        self.x = x
        self.y = y

        self.field = None
        self.new_field()

    def check_piece(self, x):
        """
        Make sure the selected field is valid.
        :param x: column to add a piece
        :return: bool
        """
        try:
            x = int(x)
            column = self.field[x]
        except (IndexError, ValueError):
            LOG.debug("Not a valid column!")
            return False
        if all(column) != 0:
            LOG.debug("Full!")
            return False
        else:
            return column

    def place_piece(self, x, player):
        """
        Places a piece in the given column by the given player.
        :param x: column index
        :param player: player representation
        :return: False if invalid column (full or otherwise), 'Valid' if successful, 1 if win condition achieved
        """
        assert player in PLAYERS, "Invalid player ID!"
        column = self.check_piece(x)
        column[column.index(0)] = player
        self.field[x] = column
        if self.check_for_winner(player):
            return 1
        # Check for a tie
        if all(all(col) != 0 for col in self.field):
            return 2
        return 'Valid'

    def _check_for_four(self, potential):
        """
        Checks a boolean list for 4 consecutive Trues
        :param potential: boolean list
        :return: True if four consecutive Trues exist
        """
        return True if (True, True, True, True) in zip(potential, potential[1:], potential[2:],
                                                       potential[3:]) else False

    def _straight_check(self, bool_field):
        """
        Check across a straight line for the win condition
        :param bool_field: boolean field where True is where the player's pieces are
        :return: True if win, False if not
        """
        values = []
        for potential in bool_field:
            values.append(self._check_for_four(potential))
        if True in values:
            return True
        return False

    def _diagonal_check(self, bool_field):
        """
        Check across a diagonal line for the win condition
        :param bool_field: boolean field where True is where the player's pieces are
        :return: True if win, False if not
        """
        values = []
        for potential_index in range(-self.x + 1, self.y):
            potential = bool_field.diagonal(potential_index)
            values.append(self._check_for_four(potential))
        if True in values:
            return True
        return False

    def check_for_winner(self, player):
        """
        Checks for all four win conditions for the given player
        :param player: player string
        :return: True if win, False if not
        """
        field = np.array(self.field)
        field = field == player

        checks = []
        # Check for all four winning conditions.  Keep these in this order!
        checks.append(self._straight_check(field.T))
        checks.append(self._straight_check(field))
        checks.append(self._diagonal_check(field))
        checks.append(self._diagonal_check(np.fliplr(field)))

        for i, win_condition in enumerate(checks):
            if win_condition == True:
                print("Win Condition Met: {}".format(WIN_CONDITIONS[i]))
                return True
        return False

    def new_field(self):
        """Resets field"""
        self.field = [[0] * self.y for _ in range(self.x)]

    @property
    def flattened_field(self):
        return np.array(self.field).flatten()

    @property
    def raw_display(self):
        """Internal representation"""
        return self.field

    @property
    def debug_display(self):
        """Array representation"""
        return np.flipud(np.array(self.field).T)

    @property
    def colored_display(self):
        """Colored representation"""
        board = np.flipud(np.array(self.field).T)
        representation = []
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                tile = board[i, j]
                if PLAYER_MAP[tile] == 'r':
                    representation.append("{}R{} ".format(Fore.RED, Style.RESET_ALL))
                elif PLAYER_MAP[tile] == 'b':
                    representation.append("{}B{} ".format(Fore.BLUE, Style.RESET_ALL))
                else:
                    representation.append("0 ")
                if j == board.shape[1] - 1:
                    representation.append('\n')
        return "".join(representation)

    def display(self, debug=False):
        """
        Display the board, either as the colored end user product, or the debug display
        :param debug: bool, true to display the debug display
        :return:
        """
        if debug:
            return self.debug_display
        else:
            return self.colored_display
