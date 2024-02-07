import numpy as np

brandubh = """\
X..A..X
...A...
...D...
AADKDAA
...D...
...A...
X..A..X"""

char_to_num = {'X': -1,
               '.': 0,
               'A': 1,
               'D': 2,
               'K': 3}


class GameNode:
    def __init__(self, board):
        self.board = np.array([[char_to_num[char] for char in list(c)]
                               for c in board.splitlines()]
                              )

    def get_action_space(self, player=None):
        action_space = np.zeros((24, 7, 7))
        for i in range(7):
            for j in range(7):
                if player == 1 and self.board[i, j] != 1:
                    continue
                elif player == 0 and self.board[i, j] not in [2, 3]:
                    continue
                else:
                    action_space[:, i, j] = self.get_actions((i, j))
        return action_space

    def get_actions(self,
                    index):
        if self.board[index[0], index[1]] <= 0:
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
        if self.board[index[0], index[1]] != 3:
            restrictions = [-1, 1, 2, 3]
        else:
            restrictions = [1, 2, 3]

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

    def make_move(self,
                  index,
                  action,
                  player=None,
                  ):
        if player == 1:
            legal_pieces = [1]
        elif player == 0:
            legal_pieces = [2, 3]
        else:
            legal_pieces = [1, 2, 3]

        # Should be upgraded to a cache in the future
        if (self.board[index[0], index[1]] not in legal_pieces or
                self.get_actions(index)[action] == 0):
            print(self.board)
            print(self.board[index[0], index[1]])
            print(index)
            print(action)
            print(self.get_actions(index))
            raise Exception("Invalid action")

        # Get the move axis, direction, and number of tiles.
        axis = 0 if action < 12 else 1
        direction = 1 if action > 17 or (6 <= action <= 11) else -1
        num = (action % 6) + 1

        # Get the new index to which the piece at `index` will move.
        new_index = list(index)
        new_index[axis] += direction * num
        new_index = tuple(new_index)

        # Make the move
        self.board[new_index[0], new_index[1]] = self.board[index[0], index[1]]
        self.board[index[0], index[1]] = 0
        return new_index

    def capture(self,
                index,
                player,
                ):
        enemies = [-1, 1] if player == 0 else [-1, 2, 3]
        friends = [-1, 2, 3] if player == 0 else [-1, 1]
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            adjacent_row, adjacent_col = index[0] + dr, index[1] + dc
            flanker_row, flanker_col = adjacent_row + dr, adjacent_col + dc
            if (0 <= adjacent_row < 7 and 0 <= adjacent_col < 7 and
                    0 <= flanker_row < 7 and 0 <= flanker_col < 7 and
                    self.board[adjacent_row, adjacent_col] in enemies and
                    self.board[flanker_row, flanker_col] in friends):
                # There is an adjacent enemy who is flanked. Eliminate it.
                self.board[adjacent_row, adjacent_col] = 0

    def check_terminal(self):
        if (self.board[0, 0] == 3 or
                self.board[0, 6] == 3 or
                self.board[6, 0] == 3 or
                self.board[6, 6] == 3 or
                not np.isin(self.board, [1]).any()):
            # print("Defenders Win!")
            return 0
        elif not np.isin(self.board, [3]).any():
            # print("Attackers Win!")
            return 1
        else:
            return -1
