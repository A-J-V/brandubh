import ai
import ai_utils
from core import *
import curriculum_learning
import episodes

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:

    def __init__(self,
                 player,
                 data_path,
                 batch_size,
                 attacker,
                 defender,
                 loss_fn,
                 num_iter,
                 games_per_iter,
                 optimizer,
                 device='cpu',
                 epochs_per_iter=2,
                 mode='standard',
                 nuke_records=True
                 ):
        # Currently only works for one player.
        self.player = player
        self.data_path = data_path
        self.batch_size = batch_size
        self.device = device
        self.attacker = attacker
        self.defender = defender
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.num_iter = num_iter
        self.num_games = games_per_iter
        self.epochs = epochs_per_iter
        self.mode = mode
        self.nuke = nuke_records

    def train(self):
        for iteration in range(self.num_iter):
            # Add something here to monitor stats over the training sessions
            for i in tqdm(range(self.num_games)):
                # Open a new environment and run through the game
                if self.mode == 'standard':
                    env = episodes.Standard(attacker=self.attacker,
                                            device=self.device,
                                            uid=i)
                    terminal = env.play()

            # Prepare the data for training
            dataset = ai_utils.BrandubhDataset(data_path=self.data_path,
                                               player=self.player)
            dataloader = DataLoader(dataset=dataset,
                                    batch_size=self.batch_size,
                                    shuffle=True)

            for epoch in range(self.epochs):
                if self.player == 1:
                    ai_utils.train_agent(self.attacker,
                                         self.loss_fn,
                                         self.device,
                                         dataloader,
                                         self.optimizer)
                elif self.player == 0:
                    ai_utils.train_agent(self.defender,
                                         self.loss_fn,
                                         self.device,
                                         dataloader,
                                         self.optimizer)

            if self.nuke:
                files = os.listdir("./game_records")
                for file in files:
                    file_path = os.path.join(self.data_path, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)