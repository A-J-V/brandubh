import ai
from core import *
import mcts


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

    def play(self):
        while self.game.winner == -1:
            action_space = self.game.action_space

            if self.game.player == 1:
                action_selected, prob = self.attacker.select_action(self.game.board,
                                                                    action_space,
                                                                    device=self.device)
                _ = self.defender.predict_value(self.game.board, device=self.device)
            elif self.game.player == 0:
                action_selected, prob = self.defender.select_action(self.game.board,
                                                                    action_space,
                                                                    device=self.device)
                _ = self.attacker.predict_value(self.game.board, device=self.device)
            else:
                raise Exception("Unknown player")
            self.game = self.game.step(action_selected)

        return self.game


class MCTSGame:
    """An episode designed to test and debug MCTS."""
    def __init__(self, base_iter):
        self.game = GameNode()
        self.base_iter = base_iter

    def play(self):
        while not self.game.is_terminal:
            self.game = mcts.run_mcts(self.game, base_iter=self.base_iter)

        return self.game
