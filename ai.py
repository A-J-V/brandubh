import torch as T
from torch import nn
import numpy as np


class RandomAI:
    def __init__(self, player):
        self.player = player

    def to(self, device=None):
        # Dummy function
        pass

    def select_action(self, _, action_space, recorder=None, device=None):
        action_space = action_space
        action_probs = np.where(action_space == 1, 1 / np.sum(action_space), 0)
        action_selection = np.argmax(np.random.multinomial(1, action_probs))

        if recorder is not None:
            recorder.actions.append(action_selection)
            recorder.action_probs.append(action_probs[action_selection])
            if self.player == 1:
                recorder.v_est_1.append(0.0)
            elif self.player == 0:
                recorder.v_est_0.append(0.0)

        return action_selection, _

    def predict_value(self, _, recorder=None, device=None):
        if recorder is not None:
            if self.player == 1:
                recorder.v_est_1.append(0.0)
            elif self.player == 0:
                recorder.v_est_0.append(0.0)
        return 0.0


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

    def forward(self, x, mask):
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
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
        position_tensor = position_tensor / 7
        self.position_tensor = position_tensor.unsqueeze(0)

        self.embedding = nn.Embedding(5, 62, padding_idx=0)

        self.attn = TransformerBlock(64, 1)
        self.attn2 = TransformerBlock(64, 1)

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

    def forward(self, x, compute_policy=True):
        batch_size = x.shape[0]
        x = x.long().view(batch_size, -1)
        mask = T.where(x == 0, 1, 0).float()
        x = self.embedding(x)
        x = x.view((batch_size, 62, 7, 7))
        batch_position = self.position_tensor.to(x.device.type).expand(x.shape[0], 2, 7, 7)
        x = T.cat((batch_position, x), dim=1)

        x = x.view(x.shape[0], x.shape[1], x.shape[2] ** 2)
        x = x.permute(0, 2, 1)

        x = self.attn(x, mask)
        x = self.attn2(x, mask)

        x = x.view(x.size(0), -1)
        value = self.value(x)
        if not compute_policy:
            return value
        policy = self.policy(x)
        return policy, value

    def predict_probs(self, x, action_space):
        policy_pred, value_est = self.forward(x)
        probs = T.softmax(policy_pred - policy_pred.max(), dim=1)
        legal_probs = T.where(action_space == 1, probs, T.zeros_like(probs))
        legal_probs /= T.sum(legal_probs, dim=1, keepdim=True)

        if T.isnan(probs).any():
            T.set_printoptions(threshold=10_000)
            print("### POLICY PRED ###")
            print(policy_pred)
            print("### PROBS ###")
            print(probs)
            print("### LEGAL PROBS ###")
            print(legal_probs)
            raise Exception("NaNs detected in pred_probs")

        return legal_probs, value_est

    def select_action(self, x, action_space, recorder=None, device='cuda'):
        with T.inference_mode():
            x = T.from_numpy(x).float().unsqueeze(0).to(device)
            action_space = T.from_numpy(action_space).float().unsqueeze(0).to(device)
            action_space = action_space.view(action_space.shape[0], -1)
            action_probs, value_est = self.predict_probs(x, action_space)

        # torch.multinomial doesn't work, it samples elements with 0.0 probability.
        # action_selection = T.multinomial(action_probs, 1)

        # The below code replicates the functionality of torch.multinomial() without the bug.
        action_cdf = T.cumsum(action_probs, dim=1)
        uniform = T.rand((action_probs.shape[0],)).to(device)
        action_selection = (uniform.unsqueeze(0) < action_cdf).long().argmax(1)

        # For debugging
        selected_prob = action_probs[0, action_selection].item()
        #

        if recorder is not None:
            recorder.actions.append(action_selection.item())
            action_probs = action_probs.squeeze(0).detach().cpu().numpy()
            recorder.action_probs.append(action_probs[action_selection])
            if self.player == 1:
                recorder.v_est_1.append(value_est.detach().cpu().numpy().item())
            elif self.player == 0:
                recorder.v_est_0.append(value_est.detach().cpu().numpy().item())

        action_selection = np.unravel_index(action_selection.item(), (24, 7, 7))
        return action_selection, selected_prob

    def predict_value(self, x, recorder=None, device='cuda'):
        x = T.from_numpy(x).float().unsqueeze(0).to(device)
        value_est = self.forward(x, compute_policy=False)
        if recorder is not None:
            if self.player == 1:
                recorder.v_est_1.append(value_est.detach().cpu().numpy().item())
            elif self.player == 0:
                recorder.v_est_0.append(value_est.detach().cpu().numpy().item())
        return value_est
