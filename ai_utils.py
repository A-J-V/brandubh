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


class NeuralGameRecorder:
    """Helper class to record a game's data for RL training.

    This is a more detailed version of GameRecorder that stores the data required to use PPO training.
    """
    def __init__(self):
        self.player = []
        self.state = []
        self.action_space = []
        self.actions = []
        self.action_probs = []
        self.policy_target = []
        self.value_estimate = []
        self.winner = None

    def extract(self, node):
        if node.is_terminal:
            self.winner = node.winner
        else:
            self.player.insert(0, node.player)
            self.state.insert(0, node.board.flatten())
            self.action_space.insert(0, node.legal_actions)
            self.actions.insert(0, node.selected_action)
            self.action_probs.insert(0, node.selected_action_prob)
            self.policy_target.insert(0, node.policy)
            self.value_estimate.insert(0, node.value_estimate)

        if node.parent is not None:
            self.extract(node.parent)
        return self

    def get_td_error(self,
                     v_est: np.ndarray,
                     v_est_next: np.ndarray,
                     player: int,
                     winner: int,
                     ) -> np.ndarray:
        """Calculate the temporal difference error for the game from a given perspective."""
        rewards = np.zeros_like(v_est)
        if winner == player:
            rewards[-1] = 1
        else:
            rewards[-1] = 0

        # Confirm that the terminal node doesn't have any future value estimated.
        v_est_next[-1] = 0
        td_error = v_est_next - v_est + rewards
        return td_error

    def calculate_GAE(self, td_error, gamma=0.99, lambda_=0.90):
        """Calculate the Generalized Advantage Estimate."""
        gae = []
        gae_t = 0
        for t in reversed(range(len(td_error))):
            delta = td_error.iloc[t]
            gae_t = delta + gamma * lambda_ * gae_t
            gae.insert(0, gae_t)
        return gae

    def record(self):
        """Record the game's data for later AI training."""
        assert (len(self.player) ==
                len(self.state) ==
                len(self.action_space) ==
                len(self.actions) ==
                len(self.action_probs) ==
                len(self.policy_target) ==
                len(self.value_estimate)), "Game record is incomplete"

        state_df = pd.DataFrame(self.state)
        state_df.columns = [f'c_{i}' for i, _ in enumerate(state_df.columns)]
        policy_df = pd.DataFrame(self.policy_target)
        policy_df.columns = [f'p_{i}' for i, _ in enumerate(policy_df.columns)]
        action_space_df = pd.DataFrame(self.action_space)
        action_space_df.columns = [f'as_{i}' for i, _ in enumerate(action_space_df.columns)]
        game_record = pd.concat([state_df, policy_df, action_space_df], axis=1)
        game_record['value_target'] = np.where(np.array(self.player) == self.winner, 1, 0)
        game_record['player'] = self.player
        game_record['action_taken'] = self.actions
        game_record['action_prob'] = self.action_probs
        game_record['winner'] = self.winner
        game_record['terminal'] = 0
        game_record.iloc[-1, game_record.columns.get_loc('terminal')] = 1
        game_record['v_est'] = self.value_estimate
        game_record['v_est_next'] = game_record['v_est'].shift(-1, fill_value=0)
        game_record['attacker_td_error'] = self.get_td_error(game_record['v_est'].values,
                                                             game_record['v_est_next'].values,
                                                             player=1,
                                                             winner=self.winner,
                                                             )
        game_record['defender_td_error'] = self.get_td_error(1 - game_record['v_est'].values,
                                                             1 - game_record['v_est_next'].values,
                                                             player=0,
                                                             winner=self.winner,
                                                             )
        game_record['attacker_gae'] = self.calculate_GAE(game_record['attacker_td_error'])
        game_record['defender_gae'] = self.calculate_GAE(game_record['defender_td_error'])

        return game_record
