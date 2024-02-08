import ai
import ai_utils
from core import *
import curriculum_learning

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class Standard:
    def __init__(self,
                 recorded: bool,
                 attacker=None,
                 defender=None,
                 device='cpu',
                 id=0,
                 ):
        self.id = id
        self.game = GameNode()
        self.device = device
        if attacker is None:
            self.attacker = ai.RandomAI(player=1)
        else:
            self.attacker = attacker.to(self.device)
        if defender is None:
            self.defender = ai.RandomAI(player=0)
        else:
            self.defender = defender.to(self.device)
        if recorded:
            self.recorder = ai_utils.GameRecorder()
        else:
            self.recorder = None

    def play(self):
        player = 1
        while self.game.check_terminal() == -1:
            action_space = self.game.get_action_space(player)
            # First check whether this player has any moves.
            # If not, the last turn was actually terminal, so
            # update the record and end the game.
            if np.sum(action_space) == 0:
                self.recorder.terminal[-1] = 1
                self.recorder.winner = 1 if player == 0 else 0
                terminal = 1 if player == 0 else 0
                if player == 1:
                    print("Attackers have no moves--defenders win!")
                elif player == 0:
                    print("Defenders have no moves--attackers win!")
                else:
                    raise Exception("Invalid Player!")
                break

            self.recorder.player.append(player)
            self.recorder.state.append(self.game.board.flatten())
            self.recorder.action_space.append(action_space.flatten())
            if player == 1:
                action_selected, prob = self.attacker.select_action(self.game.board,
                                                                    action_space,
                                                                    self.recorder,
                                                                    device=self.device)
                _ = self.defender.predict_value(self.game.board, self.recorder, device=self.device)
            elif player == 0:
                action_selected, prob = self.defender.select_action(self.game.board,
                                                                    action_space,
                                                                    self.recorder,
                                                                    device=self.device)
                _ = self.attacker.predict_value(self.game.board, self.recorder, device=self.device)
            else:
                raise Exception("Unknown player")
            self.game = self.game.step(action_selected, player)
            terminal = self.game.check_terminal()
            player = 1 if player == 0 else 0
            if terminal != -1:
                self.recorder.terminal.append(1)
                self.recorder.winner = terminal
            else:
                self.recorder.terminal.append(0)
            self.recorder.tick()
        if self.recorder is not None:
            self.recorder.record().to_csv(f"./game_records/game_{self.id}.csv", index=False)
        return terminal
