import ai
from core import *
import mcts
import graphics


class RandomSelfPlay:
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


class MctsSelfPlay:
    """An episode designed to test and debug MCTS."""
    def __init__(self, base_iter):
        self.game = GameNode()
        self.base_iter = base_iter

    def play(self):
        """Play the game until termination."""
        while not self.game.is_terminal:
            self.game = mcts.run_mcts(self.game, base_iter=self.base_iter)

        return self.game


class HumanVMcts:
    """A game in which a human player will play against an MCTS-based AI."""
    def __init__(self, base_iter=10, human=1):
        self.game = GameNode()
        self.base_iter = base_iter
        self.human = human
        print(f"Human is playing as {'attacker' if human == 1 else 'defender'}.")

    def play(self):
        """Play the game until termination, alternating between MCTS and human input."""
        display = graphics.initialize()
        graphics.refresh(self.game.board, display)

        while not self.game.is_terminal:
            print(f"Player: {self.game.player}")
            if self.game.player == self.human:
                human_move = input("Your move, human! Type input as 'move, row, col'.")
                action = np.ravel_multi_index([int(x) for x in human_move.split(',')],
                                              dims=(24, 7, 7),
                                              ).item()
                self.game = self.game.step(action)
            else:
                self.game = mcts.run_mcts(self.game, base_iter=self.base_iter)

            graphics.refresh(self.game.board, display)
        print(f"{self.game.winner} wins!")

