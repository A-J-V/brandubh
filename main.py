"""A minimal implementation of (simplified) Brandubh. Created to train RL agents."""

import ai
import ai_utils
from core import *
import curriculum_learning
import episodes
import trainers

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

attacker_winrate = []
if __name__ == '__main__':
    device = 'cpu'
    attacker = ai.PpoAttentionAgent(player=1)
    print(f"Model has {sum(p.numel() for p in attacker.parameters() if p.requires_grad)} parameters")
    attacker.to(device)
    optimizer = torch.optim.Adam(attacker.parameters(), lr=0.0001)
    trainer = trainers.Trainer(player=1,
                               data_path="./game_records",
                               batch_size=1024,
                               attacker=attacker,
                               defender=ai.RandomAI(player=0),
                               loss_fn=ai_utils.PPOLoss(),
                               num_iter=5,
                               games_per_iter=1000,
                               optimizer=optimizer,
                               device=device,
                               epochs_per_iter=2,
                               mode='standard',
                               nuke_records=True,
                               )
    trainer.train()
