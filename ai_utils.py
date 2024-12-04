"""These are some tools used to help gather data and train the AI"""

from ai import load_agent, load_value_function
import numpy as np
import os
import pandas as pd
import torch as torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


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

    @staticmethod
    def get_td_error(v_est: np.ndarray,
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

    @staticmethod
    def calculate_GAE(td_error, gamma=0.99, lambda_=0.90):
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


class PPOBrandubhDataset(Dataset):

    def __init__(self, data_paths, player=1):
        df_list = []
        print("Loading dataset...")
        for data_path in data_paths:
            for file in (os.listdir(data_path)):
                record_path = os.path.join(data_path, file)
                df_list.append(
                    pd.read_csv(
                        record_path,
                        on_bad_lines='skip'))

        self.player = player
        self.data = pd.concat(df_list, ignore_index=True)

        if player == 1:
            self.data = self.data.loc[self.data['player'] == 1]
        elif player == 0:
            self.data = self.data.loc[self.data['player'] == 0]

        self.data = self.data.to_numpy(dtype=np.float32, copy=True)
        print(f"Loaded {len(self.data)} examples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state_space = torch.tensor(self.data[idx][:49]).float()
        action_space = torch.tensor(self.data[idx][1225:2401]).float()
        action_taken = torch.tensor(self.data[idx][2403]).type(torch.LongTensor)
        action_prob = torch.tensor(self.data[idx][2404]).float()
        gae = torch.tensor(self.data[idx][-1]).float() if self.player == 0 else torch.tensor(self.data[idx][-2]).float()

        return (state_space,
                action_space,
                action_taken,
                action_prob,
                gae)


class ValueBrandubhDataset(Dataset):

    def __init__(self, data_paths):
        df_list = []
        print("Loading dataset...")
        for data_path in data_paths:
            for file in (os.listdir(data_path)):
                record_path = os.path.join(data_path, file)
                df_list.append(
                    pd.read_csv(
                        record_path,
                        on_bad_lines='skip'))

        self.data = pd.concat(df_list, ignore_index=True)
        self.data = self.data.loc[self.data['winner'] != -1]
        self.data = self.data.to_numpy(dtype=np.float32, copy=True)
        print(f"Loaded {len(self.data)} examples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state_space = torch.tensor(self.data[idx][:49]).float()
        player = torch.tensor(self.data[idx][2402]).float()
        winner = torch.tensor(self.data[idx][2405]).float()

        return (state_space,
                player,
                winner,
               )


class PPOLoss(nn.Module):
    def __init__(self, e=0.2, c=None):
        super().__init__()
        self.e = e
        self.c = c

    def forward(self,
                policy_probs,
                action_taken,
                action_prob,
                gae):

        # For PPO, we want to compare the probability of the previously taken action under the old
        # versus new policy, so we need to know which action was taken and what its probability is
        # under the new policy.
        np_prob = torch.gather(policy_probs, 1, action_taken.unsqueeze(-1)).squeeze()

        ratio = (np_prob / action_prob) * gae
        clipped_ratio = torch.clamp(ratio, 1 - self.e, 1 + self.e) * gae
        clipped_loss = torch.min(ratio, clipped_ratio)
        clipped_loss = -clipped_loss.mean()

        total_loss = clipped_loss

        if self.c is not None:
            entropy = -(policy_probs * (policy_probs + 0.0000001).log()).sum(-1)
            total_loss -= self.c * entropy.mean()

        return total_loss


def train_agent(policy_network, loss_fn, device, dataloader, optimizer):
    policy_network.train()

    for batch_idx, (state_space,
                    action_space,
                    action_taken,
                    action_prob,
                    gae) in enumerate(dataloader):
        state_space = state_space.to(device)
        action_space = action_space.to(device)
        action_taken = action_taken.to(device)
        action_prob = action_prob.to(device)
        gae = gae.to(device)
        optimizer.zero_grad()

        policy_probs = policy_network.predict_probs(state_space, action_space)

        loss = loss_fn(policy_probs,
                       action_taken,
                       action_prob,
                       gae)

        loss.backward()

        optimizer.step()


def train_value(value_network, loss_fn, device, dataloader, optimizer):
    value_network.train()

    for batch_idx, (state_space,
                    player,
                    winner,
                    ) in enumerate(dataloader):
        state_space = state_space.to(device)
        player = player.to(device).unsqueeze(1)

        winner = winner.to(device).unsqueeze(1)

        optimizer.zero_grad()

        value_est = value_network.forward(state_space, player)

        loss = loss_fn(value_est,
                       winner,
                       )

        loss.backward()

        optimizer.step()


def train_all(attacker_path: str,
              defender_path: str,
              value_path: str,
              data_paths: list,
              checkpoint_path: str,
              epochs: int,
              iteration: int,
              device: str = 'cuda',
              ):
    """Run a complete training iteration on all networks and checkpoint them

    :param attacker_path: The path to the neural network for the attacker policy
    :type attacker_path: str
    :param defender_path: The path to the neural network for the defender policy
    :type defender_path: str
    :param value_path: The path to the neural network for the value function
    :type value_path: str
    :param data_paths: The path to the data to use for this training
    :type data_paths: str
    :param checkpoint_path: The path to the folder to save network checkpoints to
    :type checkpoint_path: str
    :param epochs: The number of epochs to train over the most recently generated data
    :type epochs: int
    :param iteration: Which iteration in the global training pipeline this is. Used for checkpoint versioning.
    :type iteration: int
    :param device: Which device to use while training, 'cpu' or 'cuda'
    :type device: str
    """

    print("Running a training update...")

    attacker_policy_network = load_agent(attacker_path, dropout=0.2, player=1)
    defender_policy_network = load_agent(defender_path, dropout=0.2, player=0)
    value_network = load_value_function(value_path, dropout=0.2)

    attacker_optimizer = torch.optim.Adam(attacker_policy_network.parameters(), lr=0.0001,)
    defender_optimizer = torch.optim.Adam(defender_policy_network.parameters(), lr=0.0001, )
    value_optimizer = torch.optim.Adam(value_network.parameters(), lr=0.0001, )

    # Step 1: Load all datasets
    attacker_dataset = PPOBrandubhDataset(data_paths=data_paths,
                                          player=1,
                                          )
    defender_dataset = PPOBrandubhDataset(data_paths=data_paths,
                                          player=0,
                                          )
    value_dataset = ValueBrandubhDataset(data_paths=data_paths)

    # Step 2: Prepare all dataloaders
    attacker_loader = DataLoader(attacker_dataset,
                                 batch_size=2048,
                                 shuffle=True,
                                 )
    defender_loader = DataLoader(defender_dataset,
                                 batch_size=2048,
                                 shuffle=True,
                                 )
    value_loader = DataLoader(value_dataset,
                              batch_size=2048,
                              shuffle=True,
                              )

    policy_loss_fn = PPOLoss(c=0.025)
    value_loss_fn = nn.BCELoss()

    # Step 3: Train both policy networks and the value network
    for epoch in range(epochs):
        train_agent(attacker_policy_network, policy_loss_fn, device, attacker_loader, attacker_optimizer)
        train_agent(defender_policy_network, policy_loss_fn, device, defender_loader, defender_optimizer)
        train_value(value_network, value_loss_fn, device, value_loader, value_optimizer)

    # Step 4: Checkpoint the networks
    torch.save(attacker_policy_network.state_dict(), checkpoint_path + f"/attacker_cp{iteration + 1}.pth")
    torch.save(defender_policy_network.state_dict(), checkpoint_path + f"/defender_cp{iteration + 1}.pth")
    torch.save(value_network.state_dict(), checkpoint_path + f"/value_cp{iteration + 1}.pth")


