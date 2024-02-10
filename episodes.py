import ai
import ai_utils
from core import *
import curriculum_learning
import mcts

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class Standard:
    def __init__(self,
                 attacker=None,
                 defender=None,
                 device='cpu',
                 uid=0,
                 ):
        self.uid = uid
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
        self.recorder = ai_utils.GameRecorder()

    def play(self):
        while self.game.terminal == -1:
            action_space = self.game.action_space

            self.recorder.player.append(self.game.player)
            self.recorder.state.append(self.game.board.flatten())
            self.recorder.action_space.append(action_space.flatten())
            if self.game.player == 1:
                action_selected, prob = self.attacker.select_action(self.game.board,
                                                                    action_space,
                                                                    self.recorder,
                                                                    device=self.device)
                _ = self.defender.predict_value(self.game.board, self.recorder, device=self.device)
            elif self.game.player == 0:
                action_selected, prob = self.defender.select_action(self.game.board,
                                                                    action_space,
                                                                    self.recorder,
                                                                    device=self.device)
                _ = self.attacker.predict_value(self.game.board, self.recorder, device=self.device)
            else:
                raise Exception("Unknown player")
            self.game = self.game.step(action_selected)
            terminal = self.game.terminal
            if terminal != -1:
                self.recorder.terminal.append(1)
                self.recorder.winner = terminal
            else:
                self.recorder.terminal.append(0)
            self.recorder.tick()
        if self.recorder is not None:
            self.recorder.record().to_csv(f"./game_records/game_{self.uid}.csv", index=False)
        #self.game.walk_back()
        return terminal


class MCTSGame:
    """An episode designed to test and debug MCTS."""
    def __init__(self, num_iter=2000):
        self.game = GameNode()
        self.num_iter = num_iter

    def play(self):
        while not self.game.is_terminal:
            self.game = mcts.run_mcts(self.game, num_iter=self.num_iter)
        self.game.walk_back()
