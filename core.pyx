import numpy as np
cimport numpy as np
cimport cython
from typing import Tuple

brandubh = """\
X..A..X
...A...
...D...
AADKDAA
...D...
...A...
X..A..X"""

char_to_num = {'X': 4,
               '.': 0,
               'A': 1,
               'D': 2,
               'K': 3}


class GameNode:
    """The GameNode is the fundamental unit of the program that contains almost all data and functionality.

    :param player: Whose turn is it in this GameNode? Player 1 is attackers, player 0 is defenders.
    :type player: int
    :param board: A string or numpy array used to initialize the game state of this GameNode.
    :type board: str or numpy.ndarray
    :param parent: A reference to this GameNode's parent GameNode.
    :type parent: class:`GameNode`, optional
    :param action_index: The action index that spawned this GameNode.
    :type action_index: int, optional
    :param piece_counts: A dictionary tracking how many of each piece remain on the board.
    :type piece_counts: dict, optional
    """

    BLANK = 0
    ATTACKER = 1
    DEFENDER = 2
    KING = 3
    CORNER = 4

    def __init__(self, player: int = 1, board=brandubh, parent=None, action_index=None, piece_counts=None) -> None:
        """Constructor method
        """
        if isinstance(board, str):
            self.board = np.array([[char_to_num[char] for char in list(c)]
                                   for c in board.splitlines()],
                                  dtype=np.intc)
        elif isinstance(board, np.ndarray):
            self.board = np.array(board, dtype=np.intc)
        else:
            raise Exception("Unrecognized board type")

        # Whose turn is it in this node?
        self.player = player

        # What action did the parent choose that spawned this Node?
        self.action_index = action_index

        # If a child, the parent will quickly reassign these; otherwise, they're assigned here.
        if parent is None:
            self.winner = -1
            self.all_actions = self.get_action_space()
            self.unexpanded_children = [action.item() for action in np.argwhere(self.action_space == 1)]
        else:
            self.winner = None
            self.all_actions = None
            self.unexpanded_children = None

        # These are used for memoization (caching)
        if piece_counts is None:
            self.piece_counts = {1: 8, 2: 4, 3: 1}
        else:
            self.piece_counts = dict(piece_counts)
        self.need_update = []
        self.need_scan = []

        # These are used for MCTS
        self.policy = None
        self.visits = 0
        self.value = 0
        self.children = []
        self.parent = parent

        # This is for neural MCTS and RL training
        self.prior = None
        self.value_estimate = None
        self.selected_action = None
        self.selected_action_prob = None

        # This is used for recording
        self.legal_actions = None

    @property
    def is_terminal(self) -> bool:
        """Syntactic sugar to check whether the game is over.

        :return: True if the game is terminal, False otherwise
        :rtype: bool
        """
        return False if self.winner == -1 else True

    @property
    def is_fully_expanded(self) -> bool:
        """Syntactic sugar to check whether all child nodes have been expanded.

        :return: True if no more unexpanded children, False otherwise
        :rtype: bool
        """
        return True if len(self.unexpanded_children) == 0 else False

    @property
    def action_space(self) -> np.ndarray:
        """Get the legal action space.

        :return: NumPy array of shape (24, 7, 7) where 1 is a legal action and 0 is illegal.
        :rtype: numpy.ndarray
        """
        if self.player == 0:
            return np.where((self.board == self.KING) | (self.board == self.DEFENDER),
                            self.all_actions,
                            0).flatten()
        else:
            return np.where(self.board == self.ATTACKER, self.all_actions, 0).flatten()

    def get_winner(self) -> int:
        if self.winner is None:
            raise Exception("GameNode's is_terminal property is not set")
        else:
            return self.winner

    def get_action_space(self) -> np.ndarray:
        """Determine the entire action space across the game board.

        This will typically only be called once since it is cached and the cache is updated as needed.

        :return: A shape (24, 7, 7) NumPy array containing the entire action space.
        :rtype: numpy.ndarray
        """
        action_space = np.zeros((24, 7, 7), dtype=np.intc)
        for i in range(7):
            for j in range(7):
                action_space[:, i, j] = self.get_actions((i, j))
        return action_space

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_actions(self,
                    index: Tuple[int, int],
                    update_mode=False) -> np.ndarray:
        """Get the legal actions at index.

        This is the most time-consuming operation in the game. Give an index, it computes legal actions in
        each of the four cardinal directions.

        :param index: The index of the cell whose legal actions we want.
        :type index: tuple
        :param update_mode: False to find actions at index. True to find what indices need an update.
        :type update_mode: bool
        :return: A (24,) NumPy array of legal actions.
        :rtype: numpy.ndarray
        """
        if self.board[index[0], index[1]] in [self.BLANK, self.CORNER] and not update_mode:
            return np.zeros(24, dtype=np.intc)
        else:
            legal_moves = np.zeros(24, dtype=np.intc)

        # Check restricted tiles
        cdef int restrictions[4]
        if self.board[index[0], index[1]] != self.KING:
            restrictions[0], restrictions[1], restrictions[2], restrictions[3] = self.ATTACKER, self.DEFENDER, self.KING, self.CORNER
        else:
            restrictions[0], restrictions[1], restrictions[2], restrictions[3] = self.ATTACKER, self.DEFENDER, self.KING, -10

        # Check the legality of the 24 possible moves this cell could make
        # 0-5 is up, 6-11 is down, 12-17 is left, 18-23 is right.
        cdef int dx[4]
        cdef int dy[4]
        dx[0], dx[1], dx[2], dx[3] = 0, 0, 1, 1
        dy[0], dy[1], dy[2], dy[3] = -1, 1, -1, 1
        cdef int init_row
        cdef int init_col
        cdef int tmp_index[2]
        init_row, init_col = index[0], index[1]
        tmp_index[0], tmp_index[1] = init_row, init_col
        cdef int k
        cdef int i
        cdef int axis
        cdef int direction
        cdef int cell

        # For each of the four cardinal directions...
        for k in range(4):
            axis = dx[k]
            direction = dy[k]
            tmp_index[0] = init_row
            tmp_index[1] = init_col
            i = k * 6
            # While we haven't exceeded the 24 possible moves in that direction...
            while i < (k + 1) * 6:
                # Take a step in that direction...
                tmp_index[axis] = tmp_index[axis] + direction
                # If we're off the edge, we are done with this direction.
                if ((tmp_index[0] < 0) or
                        (tmp_index[0] > 6) or
                        (tmp_index[1] < 0) or
                        (tmp_index[1] > 6)):
                    break

                # If we aren't off an edge, check the cell that we stepped into...
                cell = self.board[tmp_index[0], tmp_index[1]]
                if (cell != restrictions[0]) & (cell != restrictions[1]) & (cell != restrictions[2]) & (cell != restrictions[3]):
                    # If it isn't restricted, add it to the legal moves.
                    legal_moves[i] = 1
                else:
                    # Otherwise, another piece is blocking the path
                    if update_mode:
                        self.need_update.append(tuple(tmp_index))
                    break
                i += 1

        return legal_moves

    def take_action(self,
                    action: int,
                    player=None,
                    ) -> Tuple[int, int]:
        """Take an action by moving a piece.

        :param action: The action to be taken.
        :type action: int
        :param player: Which player is taking the action? 1 is attackers, 0 is defenders.
        :type player: int
        :return: The new index that the piece ended up at after the action was taken.
        :rtype: tuple
        """
        # Action is an integer, we need to unravel it to fit our (24, 7, 7) action space.
        move, row, col = np.unravel_index(action, (24, 7, 7))
        if player == 1:
            legal_pieces = [self.ATTACKER]
        elif player == 0:
            legal_pieces = [self.DEFENDER, self.KING]
        else:
            legal_pieces = [self.ATTACKER, self.DEFENDER, self.KING]

        # If the action isn't legal, print out what was attempted then raise an error
        if (self.board[row, col] not in legal_pieces or
                self.all_actions[move, row, col] == 0):
            print(self.board)
            print(self.board[row, col])
            print(row, col)
            print(move)
            print(self.all_actions[:, row, col])
            raise Exception("Invalid action")

        # Get the move axis, direction, and number of tiles.
        axis = 0 if move < 12 else 1
        direction = 1 if move > 17 or (6 <= move <= 11) else -1
        num = (move % 6) + 1

        # Get the new index to which the piece at `index` will move.
        new_index = [row, col]
        new_index[axis] += direction * num
        new_index = tuple(new_index)

        # We will need to scan the old and new index for required cache updates due to the action taken.
        self.need_scan.append((row, col))
        self.need_scan.append(new_index)

        # Make the move
        self.board[new_index[0], new_index[1]] = self.board[row, col]
        self.board[row, col] = 0
        return new_index

    def refresh_cache(self) -> None:
        """Refresh the action space cache."""
        # Any index involved in a move gets checked to see if there are more indices that have been invalidated.
        for index in self.need_scan:
            self.get_actions(index, update_mode=True)
        # After the scan, update everything that was found to need an update.
        for index in set(self.need_scan + self.need_update):
            self.all_actions[:, index[0], index[1]] = self.get_actions(index)
        # Empty both self.need_scan and self.need_update since the cache is now fresh.
        self.need_scan = []
        self.need_update = []

    def capture(self,
                index: Tuple[int, int],
                player: int,
                ) -> None:
        """Capture any enemy pieces adjacent to index.

        :param index: A tuple containing the (row, col) around which we search for captures.
        :type index: tuple
        :param player: Which player owns the piece at index?
        :type player: int
        """
        enemies = [self.ATTACKER, self.CORNER] if player == 0 else [self.DEFENDER, self.KING, self.CORNER]
        friends = [self.DEFENDER, self.KING, self.CORNER] if player == 0 else [self.ATTACKER, self.CORNER]
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        # For each of the four cardinal directions...
        for dr, dc in directions:
            adjacent_row, adjacent_col = index[0] + dr, index[1] + dc
            flanker_row, flanker_col = adjacent_row + dr, adjacent_col + dc
            # If the adjacent cell is an enemy and the next cell over is an ally...
            if (0 <= adjacent_row < 7 and 0 <= adjacent_col < 7 and
                    0 <= flanker_row < 7 and 0 <= flanker_col < 7 and
                    self.board[adjacent_row, adjacent_col] in enemies and
                    self.board[flanker_row, flanker_col] in friends):
                # There is an adjacent enemy who is flanked. Eliminate it and adjust self.piece_counts.
                self.need_scan.append((adjacent_row, adjacent_col))
                self.piece_counts[self.board[adjacent_row, adjacent_col]] -= 1
                self.board[adjacent_row, adjacent_col] = self.BLANK

    def check_winner(self) -> int:
        """Check if the game has a winner yet.

        :return: -1 if no winner, 0 if defenders won, 1 if attackers won.
        :type: int
        """
        if (self.board[0, 0] == 3 or
                self.board[0, 6] == 3 or
                self.board[6, 0] == 3 or
                self.board[6, 6] == 3 or
                self.piece_counts[self.ATTACKER] == 0
        ):
            return 0
        elif self.piece_counts[self.KING] == 0:
            return 1
        elif len(self.unexpanded_children) == 0:
            return 1 if self.player == 0 else 1
        else:
            return -1

    def walk_back(self) -> None:
        """Print the game tree by recursively printing out the board of each GameNode in this game's trajectory."""
        print(self.board)

        print("Above Player: ", self.player)
        if self.parent is not None:
            self.parent.walk_back()

    def step(self, action: int) -> GameNode:
        """Take a step to the next GameNode in the game tree.

        After an action has been selected, this method contains the logic to use that action to progress the game.

        :param action: Which action to use when we step forward in the game.
        :type action: int
        :return: The next GameNode in the tree.
        :rtype: class:`GameNode`
        """
        # We start by instantiating a new node that is the next turn of the game.
        next_node = GameNode(player=0 if self.player == 1 else 1,
                             board=self.board,
                             parent=self,
                             action_index=action,
                             piece_counts=self.piece_counts)
        # It starts with the same action space as the current node.
        next_node.all_actions = np.array(self.all_actions)
        # Then the action is taken.
        next_index = next_node.take_action(action, self.player)
        # Check for and execute possible captures.
        next_node.capture(next_index, self.player)
        # Refresh the cache of this new GameNode since we took an action and possibly captured pieces.
        next_node.refresh_cache()
        # Make a list of the new GameNode's unexpanded children.
        next_node.unexpanded_children = [a.item() for a in np.argwhere(next_node.action_space == 1)]
        # Check whether there is a winner yet.
        next_node.winner = next_node.check_winner()
        # Append the new GameNode to the list of children of the current GameNode.
        self.children.append(next_node)
        # Return the new GameNode to be used in carrying the game forward.
        return next_node

    # The below methods are used exclusively for MCTS
    def clone(self) -> GameNode:
        """Make a clone of the current node.

        When running MCTS, we need a GameNode from which to run a rollout; however, we don't want the actions in the
        rollout to impact the actual game tree, so we make a clone, rollout from the clone, save the results, then
        discard the clone and the rollout game tree.

        :return: A GameNode to use as a stand-in of the current GameNode during MCTS rollouts
        :rtype: class:`GameNode`
        """
        clone = GameNode(player=self.player,
                         board=self.board,
                         parent=-1,
                         action_index=self.action_index)
        clone.piece_counts = dict(self.piece_counts)
        clone.all_actions = np.array(self.all_actions)
        clone.winner = self.winner
        clone.unexpanded_children = [action.item() for action in np.argwhere(clone.action_space == 1)]
        return clone

    def backpropagate(self, value: float) -> None:
        """Recursively update values up through the game tree.

        :param value: The value to propagate.
        :type value: float
        """
        self.visits += 1
        self.value += value
        if self.parent is not None:
            self.parent.backpropagate(1 - value)

    def reset_mcts(self) -> None:
        """Reset the variables used in MCTS.

        This is used to clean the best child chosen at the end of MCTS since it will be attached to the real
        game tree and become part of the game.
        """
        self.visits = 0
        self.value = 0
        self.children = []
        self.unexpanded_children = [action.item() for action in np.argwhere(self.action_space == 1)]
        self.need_scan = []
        self.need_update = []
