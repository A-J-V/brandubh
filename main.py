"""A minimal implementation of (simplified) Brandubh. Created to train RL agents."""

import ai
import ai_utils
from core import *
import curriculum_learning
import episodes

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
    for iteration in range(100):
        winner_dict = {0: 0,
                       1: 0}
        for i in tqdm(range(2000)):
            env = episodes.Standard(recorded=True)
            terminal = env.play()
            winner_dict[terminal] += 1
        winrate = winner_dict[1] / (winner_dict[1] + winner_dict[0])
        print(f"Attacker winrate: {winrate}")
        attacker_winrate.append(winrate)
        dataset = ai_utils.BrandubhDataset(data_path="./game_records", player=1)
        dataloader = DataLoader(dataset=dataset,
                                batch_size=1024,
                                shuffle=True)
        loss_fn = ai_utils.PPOLoss(c2=0.0001)
        for epoch in range(3):
            ai_utils.train_agent(attacker, loss_fn, device, dataloader, optimizer)

        files = os.listdir("./game_records")
        for file in files:
            file_path = os.path.join("./game_records", file)
            if os.path.isfile(file_path):
                os.remove(file_path)
