import core
import random
import numpy as np

CHAR_MAP = {0: '.',
            1: 'A',
            2: 'D',
            3: 'K',
            4: 'X',
            }


def np_to_string(board):
    board_string = ""
    for i in range(7):
        if i != 0:
            board_string += '\n'
        for j in range(7):
            board_string += CHAR_MAP[board[i, j]]
    return board_string


def get_attacker_curriculum(n=1):
    # Only implemented for n=1 currently
    curriculum_board = np.zeros((7, 7), dtype=int)
    curriculum_board[0, 0] = 4
    curriculum_board[0, 6] = 4
    curriculum_board[6, 0] = 4
    curriculum_board[6, 6] = 4

    king_r, king_c = random.randint(1, 5), random.randint(1, 5)
    curriculum_board[king_r, king_c] = 3

    if random.random() < 0.5:
        curriculum_board[king_r - 1, king_c] = 1
        curriculum_board[king_r + 1, king_c] = 1
        attacker_r, attacker_c = random.choice([(king_r - 1, king_c),
                                                (king_r + 1, king_c)])
    else:
        curriculum_board[king_r, king_c - 1] = 1
        curriculum_board[king_r, king_c + 1] = 1
        attacker_r, attacker_c = random.choice([(king_r, king_c - 1),
                                                (king_r, king_c + 1)])
    board_string = np_to_string(curriculum_board)
    curriculum_game = core.GameNode(board_string)
    action_space = curriculum_game.get_actions((attacker_r, attacker_c))
    action_selected = random.choice(np.argwhere(action_space==1)).item()
    curriculum_game.take_action((attacker_r, attacker_c), action_selected, player=1)
    return curriculum_game
