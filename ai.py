import torch as T
from torch import nn
import numpy as np


class RandomAI:
    def __init__(self, player):
        self.player = player

    def to(self, device=None):
        # Dummy function
        pass

    def select_action(self, _, action_space, device=None):
        action_space = action_space
        action_probs = np.where(action_space == 1, 1 / np.sum(action_space), 0)
        action_selection = np.argmax(np.random.multinomial(1, action_probs))

        return action_selection, _

    def predict_value(self, _, device=None):
        return 0.0
