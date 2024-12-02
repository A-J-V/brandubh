import torch
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


class TransformerBlock(nn.Module):
    def __init__(self, n_dims, n_heads, dropout=0.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(n_dims, n_heads, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(n_dims)
        self.mlp = nn.Sequential(
            nn.Linear(n_dims, n_dims * 2),
            nn.GELU(),
            nn.Linear(n_dims * 2, n_dims),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(n_dims)

    def forward(self, x, mask=None):
        if mask is not None:
            attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        else:
            attn_out, _ = self.attention(x, x, x)
        attn_out = self.dropout1(attn_out)
        x = self.norm1(attn_out + x)
        mlp_out = self.mlp(x)
        mlp_out = self.dropout2(mlp_out)
        x = self.norm2(mlp_out + x)
        return x


class AttentionAgent(nn.Module):
    def __init__(self, player, token_count=5, n_heads=1, embedding_dim=16, dropout=0.0):
        super().__init__()
        self.player = player
        self.token_count = token_count
        board_size = 7
        self.position_tensor = nn.Parameter(torch.randn(board_size * board_size, embedding_dim).unsqueeze(0))
        self.embedding = nn.Embedding(num_embeddings=token_count,
                                      embedding_dim=embedding_dim,
                                      )

        self.attn = TransformerBlock(embedding_dim, n_heads, dropout=dropout)
        self.attn2 = TransformerBlock(embedding_dim, n_heads, dropout=dropout)

        self.policy = nn.Linear(embedding_dim * board_size * board_size,
                                24 * board_size * board_size)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.long().view(batch_size, -1)

        x = self.embedding(x)
        # assert x.shape == self.position_tensor.shape, (
        #     f"Input shape {x.shape} does not match position tensor shape {self.position_tensor.shape}"
        # )
        x = x + self.position_tensor
        x = self.attn(x)
        x = self.attn2(x)
        x = x.view(x.size(0), -1)

        policy = self.policy(x)

        return policy

    def predict_probs(self, x, action_space):
        policy_pred = self.forward(x)
        assert policy_pred.shape == action_space.shape
        probs = torch.softmax(policy_pred - policy_pred.max(), dim=1)
        legal_probs = torch.where(action_space == 1, probs, torch.zeros_like(probs))
        legal_probs /= torch.sum(legal_probs, dim=1, keepdim=True)

        return legal_probs

    def select_action(self, x, action_space, device='cuda'):
        """Probabilistically select an action.

        This is only used during training, because it samples actions based on their probability prediction.
        """

        x = torch.from_numpy(x).float().unsqueeze(0).to(device)
        action_space = torch.from_numpy(action_space).float().unsqueeze(0).to(device)
        action_space = action_space.view(action_space.shape[0], -1)
        action_probs = self.predict_probs(x, action_space)

        # torch.multinomial doesn't work, it samples elements with 0.0 probability.
        # action_selection = T.multinomial(action_probs, 1)

        # The below code replicates the functionality of torch.multinomial() without the bug.
        action_cdf = torch.cumsum(action_probs, dim=1)
        uniform = torch.rand((action_probs.shape[0],)).to(device)
        action_selection = (uniform.unsqueeze(0) < action_cdf).long().argmax(1)

        # For debugging
        selected_prob = action_probs[0, action_selection].item()

        action_selection = np.unravel_index(action_selection.item(), (24, 7, 7))
        return action_selection, selected_prob


class ValueFunction(nn.Module):
    def __init__(self, player, token_count=5, n_heads=1, embedding_dim=16, dropout=0.0):
        super().__init__()
        self.player = player
        self.token_count = token_count
        board_size = 7
        self.position_tensor1 = nn.Parameter(torch.randn(board_size * board_size, embedding_dim).unsqueeze(0))
        self.position_tensor2 = nn.Parameter(torch.randn(board_size * board_size, embedding_dim).unsqueeze(0))
        self.embedding = nn.Embedding(num_embeddings=token_count,
                                      embedding_dim=embedding_dim,
                                      )

        self.attn = TransformerBlock(embedding_dim, n_heads, dropout=dropout)
        self.attn2 = TransformerBlock(embedding_dim, n_heads, dropout=dropout)
        self.attn3 = TransformerBlock(embedding_dim, n_heads, dropout=dropout)

        self.output = nn.Sequential(
            nn.Linear(embedding_dim * board_size * board_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, player):
        batch_size = x.size(0)
        x = x.long().view(batch_size, -1)

        x = self.embedding(x)

        x = x + self.position_tensor1
        player = player.view(batch_size, 1, 1)
        x = x + (self.position_tensor2 * player)

        x = self.attn(x)
        x = self.attn2(x)
        x = self.attn3(x)
        x = x.view(x.size(0), -1)

        output = self.output(x)

        return output

def load_value_function(player: str = 1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ValueFunction(player, embedding_dim=24)
    model.load_state_dict(torch.load("./assets/value_function_v1.pth"))
    model = model.to(device)
    model.eval()
    return model


def load_agent(player: str = "attacker"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionAgent(player, embedding_dim=24)
    if player == "attacker":
        model.load_state_dict(torch.load("./assets/attacker_agent_v1.pth"))
    else:
        model.load_state_dict(torch.load("./assets/defender_agent_v1.pth"))
    model = model.to(device)
    model.eval()
    return model