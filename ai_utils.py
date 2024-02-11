import torch as T
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os


class GameRecorder:
    """Helper class to record a game's data for later AI training."""
    def __init__(self):
        self.state = []
        self.policy_target = []
        self.value_target = []
        self.player = []
        self.winner = None

    def extract(self, node):
        self.state.append(node.board.flatten())
        self.player.append(node.player)
        if node.is_terminal:
            self.winner = node.winner
            self.policy_target.append(np.zeros(24*7*7))
        else:
            self.policy_target.append(node.policy)
        if node.parent is not None:
            self.extract(node.parent)

    def record(self):
        state_df = pd.DataFrame(self.state)
        state_df.columns = [f'c_{i}' for i, _ in enumerate(state_df.columns)]
        policy_df = pd.DataFrame(self.policy_target)
        policy_df.columns = [f'p_{i}' for i, _ in enumerate(policy_df.columns)]
        game_record = pd.concat([state_df, policy_df], axis=1)
        game_record['value_target'] = np.where(np.array(self.player) == self.winner, 1, -1)
        return game_record


class BrandubhDataset(Dataset):

    def __init__(self, data_path, player=1):
        self.player = player
        df_list = []
        print("Loading dataset...")
        for file in (os.listdir(data_path)):
            record_path = os.path.join(data_path, file)
            df_list.append(
                pd.read_csv(
                    record_path,
                    on_bad_lines='skip'))

        self.data = pd.concat(df_list, ignore_index=True)

        if player == 1:
            self.data = self.data.loc[self.data['player'] == 1]
        elif player == 0:
            self.data = self.data.loc[self.data['player'] == 0]
        else:
            raise Exception("Invalid player for dataset")

        self.data = self.data.to_numpy(dtype=np.float32, copy=True)
        print(f"Loaded {len(self.data)} examples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state_space = T.tensor(self.data[idx][:49]).float()
        action_space = T.tensor(self.data[idx][49:1225]).float()
        player = T.tensor(self.data[idx][1226]).float()
        action_taken = T.tensor(self.data[idx][1227]).type(T.LongTensor)
        action_prob = T.tensor(self.data[idx][1228]).float()
        winner = T.tensor(self.data[idx][-5]).float()
        if self.player == 1:
            gae = T.tensor(self.data[idx][-2]).float()
        else:
            gae = T.tensor(self.data[idx][-1]).float()

        return (state_space,
                action_space,
                player,
                action_taken,
                action_prob,
                winner,
                gae)


class PPOLoss(nn.Module):
    def __init__(self, e=0.2, c1=1.0, c2=None):
        super().__init__()
        self.e = e
        self.c1 = c1
        self.c2 = c2
        self.vf_loss = nn.MSELoss()

    def forward(self,
                policy_probs,
                value_est,
                action_taken,
                action_prob,
                winner,
                player,
                gae):

        # The value target is the terminal reward
        value_target = T.where(player == winner, 1, -1).unsqueeze(-1).float()

        # For PPO, we want to compare the probability of the previously taken action under the old
        # versus new policy, so we need to know which action was taken and what its probability is
        # under the new policy.
        np_prob = T.gather(policy_probs, 1, action_taken.unsqueeze(-1)).squeeze()

        ratio = (np_prob / action_prob) * gae
        clipped_ratio = T.clamp(ratio, 1 - self.e, 1 + self.e) * gae
        clipped_loss = T.min(ratio, clipped_ratio)
        clipped_loss = -clipped_loss.mean()

        value_loss = self.vf_loss(value_est, value_target)

        total_loss = clipped_loss + self.c1 * value_loss

        if self.c2 is not None:
            entropy = -(policy_probs * (policy_probs + 0.0000001).log()).sum(-1)
            total_loss -= self.c2 * entropy.mean()

        return total_loss


def train_agent(model, loss_fn, device, dataloader, optimizer):
    model.train()
    for batch_idx, (state_space,
                    action_space,
                    player,
                    action_taken,
                    action_prob,
                    winner,
                    gae) in enumerate(dataloader):

        state_space = state_space.to(device)
        action_space = action_space.to(device)
        player = player.to(device)
        action_taken = action_taken.to(device)
        action_prob = action_prob.to(device)
        winner = winner.to(device)
        gae = gae.to(device)
        optimizer.zero_grad()

        policy_probs, value_est = model.predict_probs(state_space, action_space)

        loss = loss_fn(policy_probs,
                       value_est,
                       action_taken,
                       action_prob,
                       winner,
                       player,
                       gae)

        loss.backward()

        optimizer.step()
