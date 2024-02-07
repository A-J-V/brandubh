import torch as T
from torch import nn
import numpy as np


class TestAgent(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding='same')
        self.policy = nn.Sequential(
            nn.Linear(16*7*7, 32),
            nn.ReLU(),
            nn.Linear(32, 24*7*7)
        )

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = self.network(x)
        x = self.policy(x.view(x.size(0), -1))
        return x

    def pred_probs(self, x, action_space):
        x = self.forward(x)
        probs = T.softmax(x - x.max(), dim=1)
        probs = T.where(action_space == 1, probs, T.zeros_like(probs))
        probs /= T.sum(probs, dim=1, keepdim=True)

        if T.isnan(probs).any():
            raise Exception("NaNs detected in pred_probs")

        return probs

    def select_action(self, x, action_space, recorder=None):
        with T.inference_mode():
            x = T.from_numpy(x).float().unsqueeze(0)
            action_space = T.from_numpy(action_space).float().unsqueeze(0)
            action_space = action_space.view(action_space.shape[0], -1)
            action_probs = self.pred_probs(x, action_space)

            # torch.multinomial doesn't work, it samples elements with 0.0 probability.
            # action_selection = T.multinomial(action_probs, 1)

            # The below code replicates the functionality of torch.multinomial() without the bug.
            action_cdf = T.cumsum(action_probs, dim=1)
            uniform = T.rand((action_probs.shape[0],))
            action_selection = (uniform.unsqueeze(0) < action_cdf).long().argmax(1)

            # For debugging
            selected_prob = action_probs[0, action_selection].item()
            #

            if recorder is not None:
                recorder.actions.append(action_selection.item())
                action_probs = action_probs.squeeze(0).detach().cpu().numpy()
                recorder.action_probs.append(action_probs[action_selection])
                recorder.v_est.append(0.0)
                #recorder.v_est.append(value_pred.detach().cpu().numpy().item())

            action_selection = np.unravel_index(action_selection.item(), (24, 7, 7))
            return action_selection, selected_prob


class TransformerBlock(nn.Module):
    def __init__(self, n_dims, n_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(n_dims, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(n_dims)
        self.mlp = nn.Sequential(
            nn.Linear(n_dims, n_dims * 2),
            nn.GELU(),
            nn.Linear(n_dims * 2, n_dims),
        )
        self.norm2 = nn.LayerNorm(n_dims)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(attn_out + x)
        mlp_out = self.mlp(x)
        x = self.norm2(mlp_out + x)
        return x


class PpoAttentionAgent(nn.Module):
    def __init__(self, player):
        super().__init__()
        self.player = player
        board_size = 7
        position_tensor = T.zeros((2, board_size, board_size))

        for i in range(position_tensor.shape[-1]):
            for j in range(position_tensor.shape[-2]):
                position_tensor[0, i, j] = i
                position_tensor[1, i, j] = j
        position_tensor = position_tensor / 10
        self.position_tensor = position_tensor.unsqueeze(0)

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 62, kernel_size=(3, 3), stride=1, padding='same'),
            nn.ReLU(),
        )

        self.attn = nn.Sequential(
            TransformerBlock(64, 2),
        )

        self.policy = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 24 * 7 * 7),
        )
        self.value = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape((batch_size, 1, 7, 7))
        x = self.conv(x)
        batch_position = self.position_tensor.to(x.device.type).expand(x.shape[0], 2, 7, 7)
        x = T.cat((batch_position, x), dim=1)

        x = x.view(x.shape[0], x.shape[1], x.shape[2] ** 2)
        x = x.permute(0, 2, 1)

        x = self.attn(x)

        x = x.view(x.size(0), -1)
        policy = self.policy(x)
        value = self.value(x)
        return policy, value

    def pred_probs(self, x, action_space):
        policy_pred, value_pred = self.forward(x)
        probs = T.softmax(policy_pred - policy_pred.max(), dim=1)
        probs = T.where(action_space == 1, probs, T.zeros_like(probs))
        probs /= T.sum(probs, dim=1, keepdim=True)

        if T.isnan(probs).any():
            raise Exception("NaNs detected in pred_probs")

        return probs, value_pred

    def select_action(self, x, action_space, recorder=None):
        with T.inference_mode():
            x = T.from_numpy(x).float().unsqueeze(0)
            action_space = T.from_numpy(action_space).float().unsqueeze(0)
            action_space = action_space.view(action_space.shape[0], -1)
            action_probs, value_pred = self.pred_probs(x, action_space)

            # torch.multinomial doesn't work, it samples elements with 0.0 probability.
            # action_selection = T.multinomial(action_probs, 1)

            # The below code replicates the functionality of torch.multinomial() without the bug.
            action_cdf = T.cumsum(action_probs, dim=1)
            uniform = T.rand((action_probs.shape[0],))
            action_selection = (uniform.unsqueeze(0) < action_cdf).long().argmax(1)

            # For debugging
            selected_prob = action_probs[0, action_selection].item()
            #

            if recorder is not None:
                recorder.actions.append(action_selection.item())
                action_probs = action_probs.squeeze(0).detach().cpu().numpy()
                recorder.action_probs.append(action_probs[action_selection])
                recorder.v_est.append(value_pred.detach().cpu().numpy().item())

            action_selection = np.unravel_index(action_selection.item(), (24, 7, 7))
            return action_selection, selected_prob