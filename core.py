import numpy as np

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

    def __init__(self, board=brandubh):
        if isinstance(board, str):
            self.board = np.array([[char_to_num[char] for char in list(c)]
                                   for c in board.splitlines()]
                                  )
        elif isinstance(board, np.ndarray):
            self.board = board
        else:
            raise Exception("Unrecognized board type")

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
                    index):
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
                    break
                i += 1
        # Cache would go here if updating a cache of legal moves.
        return legal_moves

    def take_action(self,
                    action,
                    player=None,
                    ):
        move, row, col = action
        if player == 1:
            legal_pieces = [self.ATTACKER]
        elif player == 0:
            legal_pieces = [self.DEFENDER, self.KING]
        else:
            legal_pieces = [self.ATTACKER, self.DEFENDER, self.KING]

        # Should be upgraded to a cache in the future
        if (self.board[row, col] not in legal_pieces or
                self.get_actions((row, col))[move] == 0):
            print(self.board)
            print(self.board[row, col])
            print(row, col)
            print(move)
            print(self.get_actions((row, col)))
            raise Exception("Invalid action")

        # Get the move axis, direction, and number of tiles.
        axis = 0 if move < 12 else 1
        direction = 1 if move > 17 or (6 <= move <= 11) else -1
        num = (move % 6) + 1

        # Get the new index to which the piece at `index` will move.
        new_index = [row, col]
        new_index[axis] += direction * num
        new_index = tuple(new_index)

        # Make the move
        self.board[new_index[0], new_index[1]] = self.board[row, col]
        self.board[row, col] = 0
        return new_index

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
                # There is an adjacent enemy who is flanked. Eliminate it.
                self.board[adjacent_row, adjacent_col] = self.BLANK

    def check_terminal(self):
        if (self.board[0, 0] == 3 or
                self.board[0, 6] == 3 or
                self.board[6, 0] == 3 or
                self.board[6, 6] == 3 or
                not np.isin(self.board, [self.ATTACKER]).any()):
            # print("Defenders Win!")
            return 0
        elif not np.isin(self.board, [self.KING]).any():
            # print("Attackers Win!")
            return 1
        else:
            return -1

    def step(self, action, player):
        next_node = GameNode(self.board)
        next_index = next_node.take_action(action, player)
        next_node.capture(next_index, player)
        return next_node
