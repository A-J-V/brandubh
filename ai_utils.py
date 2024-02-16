import numpy as np
import pandas as pd


class GameRecorder:
    """Helper class to record a game's data for later AI training."""
    def __init__(self):
        self.state = []
        self.legal_actions = []
        self.policy_target = []
        self.player = []
        self.winner = None

    def extract(self, node):
        if node.is_terminal:
            self.winner = node.winner
        else:
            self.state.append(node.board.flatten())
            self.legal_actions.append(node.legal_actions)
            self.policy_target.append(node.policy)
            self.player.append(node.player)

        if node.parent is not None:
            self.extract(node.parent)
        return self

    def record(self):
        assert len(self.state) == len(self.legal_actions) == len(self.player) == len(self.policy_target)
        self.state.reverse()
        self.legal_actions.reverse()
        self.policy_target.reverse()
        self.player.reverse()
        state_df = pd.DataFrame(self.state)
        state_df.columns = [f'c_{i}' for i, _ in enumerate(state_df.columns)]
        policy_df = pd.DataFrame(self.policy_target)
        policy_df.columns = [f'p_{i}' for i, _ in enumerate(policy_df.columns)]
        action_space_df = pd.DataFrame(self.legal_actions)
        action_space_df.columns = [f'as_{i}' for i, _ in enumerate(action_space_df.columns)]
        game_record = pd.concat([state_df, policy_df, action_space_df], axis=1)
        game_record['value_target'] = np.where(np.array(self.player) == self.winner, 1, -1)
        return game_record
