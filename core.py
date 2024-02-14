import numpy as np
from mcts import ucb1

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
    BLANK = 0
    ATTACKER = 1
    DEFENDER = 2
    KING = 3
    CORNER = 4

    def __init__(self, player=1, board=brandubh, parent=None, action_index=None, piece_counts=None):
        if isinstance(board, str):
            self.board = np.array([[char_to_num[char] for char in list(c)]
                                   for c in board.splitlines()],
                                  dtype=np.int8)
        elif isinstance(board, np.ndarray):
            self.board = np.array(board, dtype=np.int8)
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

        # These are used for memoization
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

    @property
    def is_terminal(self):
        return False if self.winner == -1 else True

    @property
    def is_fully_expanded(self):
        return True if len(self.unexpanded_children) == 0 else False

    @property
    def action_space(self):
        if self.player == 0:
            return np.where((self.board == self.KING) | (self.board == self.DEFENDER),
                            self.all_actions,
                            0).flatten()
        else:
            return np.where(self.board == self.ATTACKER, self.all_actions, 0).flatten()

    def get_winner(self):
        if self.winner is None:
            raise Exception("GameNode's is_terminal property is not set")
        else:
            return self.winner

    def get_action_space(self, player=None):
        action_space = np.zeros((24, 7, 7))
        for i in range(7):
            for j in range(7):
                if player == 1 and self.board[i, j] != self.ATTACKER:
                    continue
                elif player == 0 and self.board[i, j] not in [self.DEFENDER, self.KING]:
                    continue
                else:
                    action_space[:, i, j] = self.get_actions((i, j))
        return action_space

    def get_actions(self,
                    index,
                    fetch_mode='get_actions'):
        if self.board[index[0], index[1]] in [self.BLANK, self.CORNER]:
            return np.zeros(24)
        else:
            legal_moves = np.zeros(24)

        # Check the legality of the 24 possible moves this cell could make
        # 0-5 is up, 6-11 is down, 12-17 is left, 18-23 is right.
        dx = [0, 0, 1, 1]
        dy = [-1, 1, -1, 1]
        init_row, init_col = index
        tmp_index = [init_row, init_col]

        # Check restricted tiles
        if self.board[index[0], index[1]] != self.KING:
            restrictions = [self.ATTACKER, self.DEFENDER, self.KING, self.CORNER]
        else:
            restrictions = [self.ATTACKER, self.DEFENDER, self.KING]

        for k in range(4):
            axis = dx[k]
            direction = dy[k]
            tmp_index[0] = init_row
            tmp_index[1] = init_col
            i = k * 6
            while i < (k + 1) * 6:
                tmp_index[axis] = tmp_index[axis] + direction
                if ((tmp_index[0] < 0) or
                        (tmp_index[0] > 6) or
                        (tmp_index[1] < 0) or
                        (tmp_index[1] > 6)):
                    break

                if self.board[tmp_index[0], tmp_index[1]] not in restrictions:
                    # There is no other piece blocking the path
                    legal_moves[i] = 1
                else:
                    # There is another piece blocking the path
                    if fetch_mode == 'fetch_updates':
                        self.need_update.append(tuple(tmp_index))
                    break
                i += 1
        # Cache would go here if updating a cache of legal moves.
        return legal_moves

    def take_action(self,
                    action,
                    player=None,
                    ):
        move, row, col = np.unravel_index(action, (24, 7, 7))
        if player == 1:
            legal_pieces = [self.ATTACKER]
        elif player == 0:
            legal_pieces = [self.DEFENDER, self.KING]
        else:
            legal_pieces = [self.ATTACKER, self.DEFENDER, self.KING]

        # Should be upgraded to a cache in the future
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

        self.need_scan.append((row, col))
        self.need_scan.append(new_index)

        # Make the move
        self.board[new_index[0], new_index[1]] = self.board[row, col]
        self.board[row, col] = 0
        return new_index

    def refresh_cache(self):
        for index in self.need_scan:
            self.get_actions(index, fetch_mode='fetch_updates')
        for index in set(self.need_scan + self.need_update):
            self.all_actions[:, index[0], index[1]] = self.get_actions(index)
        self.need_scan = []
        self.need_update = []

    def capture(self,
                index,
                player,
                ):
        """Capture any enemy pieces adjacent to index."""
        enemies = [self.ATTACKER, self.CORNER] if player == 0 else [self.DEFENDER, self.KING, self.CORNER]
        friends = [self.DEFENDER, self.KING, self.CORNER] if player == 0 else [self.ATTACKER, self.CORNER]
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            adjacent_row, adjacent_col = index[0] + dr, index[1] + dc
            flanker_row, flanker_col = adjacent_row + dr, adjacent_col + dc
            if (0 <= adjacent_row < 7 and 0 <= adjacent_col < 7 and
                    0 <= flanker_row < 7 and 0 <= flanker_col < 7 and
                    self.board[adjacent_row, adjacent_col] in enemies and
                    self.board[flanker_row, flanker_col] in friends):
                # There is an adjacent enemy who is flanked. Eliminate it and adjust self.piece_counts.
                self.need_scan.append((adjacent_row, adjacent_col))
                self.piece_counts[self.board[adjacent_row, adjacent_col]] -= 1
                self.board[adjacent_row, adjacent_col] = self.BLANK

    def check_winner(self):
        if (self.board[0, 0] == 3 or
                self.board[0, 6] == 3 or
                self.board[6, 0] == 3 or
                self.board[6, 6] == 3 or
                self.piece_counts[self.ATTACKER] == 0
        ):
            # print("Defenders Win!")
            return 0
        elif self.piece_counts[self.KING] == 0:
            # print("Attackers Win!")
            return 1
        elif len(self.unexpanded_children) == 0:
            return 1 if self.player == 0 else 1
        else:
            return -1

    def walk_back(self):
        print(self.board)

        print("Player: ", self.player)
        if self.parent is not None:
            self.parent.walk_back()

    def step(self, action):
        next_node = GameNode(player=0 if self.player == 1 else 1,
                             board=self.board,
                             parent=self,
                             action_index=action,
                             piece_counts=self.piece_counts)
        next_node.all_actions = np.array(self.all_actions)
        next_index = next_node.take_action(action, self.player)
        next_node.capture(next_index, self.player)
        next_node.refresh_cache()
        next_node.unexpanded_children = [action.item() for action in np.argwhere(next_node.action_space == 1)]
        next_node.winner = next_node.check_winner()
        self.children.append(next_node)

        return next_node

    # The below methods are used exclusively for MCTS
    def clone(self):
        clone = GameNode(player=self.player,
                         board=self.board,
                         parent=-1,
                         action_index=self.action_index)
        clone.piece_counts = dict(self.piece_counts)
        clone.all_actions = np.array(self.all_actions)
        clone.winner = self.winner
        clone.unexpanded_children = [action.item() for action in np.argwhere(clone.action_space == 1)]
        return clone

    def backpropagate(self, value):
        """Recursively update values up through the game tree."""
        self.visits += 1
        self.value += value
        if self.parent is not None:
            self.parent.backpropagate(value)

    def reset_mcts(self):
        """
        A helper function to reset the variables used in MCTS.
        This is used to clean the best child chosen at the end of MCTS.
        """
        self.visits = 0
        self.value = 0
        self.children = []
        self.unexpanded_children = [action.item() for action in np.argwhere(self.action_space == 1)]
        self.need_scan = []
        self.need_update = []
