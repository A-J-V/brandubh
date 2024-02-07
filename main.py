"""A minimal implementation of (simplified) Brandubh. Created to train RL agents."""

import ai
import ai_utils
from core import *


if __name__ == '__main__':
    winner_dict = {0: 0,
                   1: 0
                   }
    for i in range(1):
        attacker = ai.PpoAttentionAgent(1)
        defender = ai.RandomAI(0)
        game = GameNode(brandubh)
        recorder = ai_utils.GameRecorder()
        player = 1
        while game.check_terminal() == -1:
            recorder.player.append(player)
            recorder.state.append(game.board.flatten())
            action_space = game.get_action_space(player)
            recorder.action_space.append(action_space.flatten())
            if player == 1:
                action_selected, prob = attacker.select_action(game.board, action_space, recorder)
                _ = defender.predict_value(game.board, recorder)
            elif player == 0:
                action_selected, prob = defender.select_action(game.board, action_space, recorder)
                _ = attacker.predict_value(game.board, recorder)
            else:
                raise Exception("Unknown player")
            move, row, col = action_selected
            game.make_move((row, col), move)
            game.capture((row, col), player)
            terminal = game.check_terminal()
            player = 1 if player == 0 else 0
            if terminal != -1:
                recorder.terminal.append(1)
                recorder.winner = terminal
            else:
                recorder.terminal.append(0)
            recorder.tick()
        winner_dict[terminal] += 1
    print(winner_dict)
    recorder.record().to_csv("./game_records/happy_test.csv", index=False)
    print(game.board)
