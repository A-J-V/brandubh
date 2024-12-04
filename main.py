"""A minimal implementation of (simplified) Brandubh. Created to train RL agents."""

import ai_utils
import episodes
import os
import time
import torch
from torch import nn

from episodes import HumanVNeural
from trainer import *
from ai_utils import train_all


if __name__ == '__main__':

    # episodes.BatchNeuralSelfPlay(num_iters=100,
    #                              num_games=1,
    #                              attacker_path="attacker_cp5",
    #                              defender_path="defender_cp5",
    #                              value_path="value_cp5",
    #                              show=True).play()

    HumanVNeural(set=5).play()

    # TODO Modify generation pipeline so that it uses multiple CPUs to generate data faster.
    # TODO Modify generation pipeline so that games are generated using a mix of the most recent agents and older agents.
    # TODO Modify training pipeline to use 70% most recent data and 30% data from the last several iterations

    # sets = 100
    # for set_ in range(1, sets):
    #
    #     latest_attacker = f"attacker_cp{set_}"
    #     latest_defender = f"defender_cp{set_}"
    #     latest_value = f"value_cp{set_}"
    #
    #     # Step 1: Generate a new set of data
    #
    #     generate_data(latest_attacker,
    #                   latest_defender,
    #                   latest_value,
    #                   100,
    #                   500,
    #                   "/home/alexander/Data/brandubh/neural_game_records",
    #                   set_,
    #                   chunks=2
    #                   )
    #
    #     # Step 2: Train on the generated set of data
    #
    #     data_paths = ["/home/alexander/Data/brandubh/neural_game_records/set_" + str(set_),
    #                   "/home/alexander/Data/brandubh/neural_game_records/set_" + str(set_ - 1),
    #                   ]
    #
    #     train_all(latest_attacker,
    #               latest_defender,
    #               latest_value,
    #               data_paths=data_paths,
    #               checkpoint_path="/home/alexander/Data/brandubh/checkpoints",
    #               epochs=1,
    #               iteration=set_,
    #               device='cuda',
    #               )
    #
    #     # Step 3: Run a benchmark to ensure that each agent is still balanced
    #
    #     bench(f"attacker_cp{set_ + 1}",
    #           f"defender_cp{set_ + 1}",
    #           f"value_cp{set_ + 1}",
    #           num_games=50,
    #           )
    #
    #     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    #     print(f"------------------------------- Completed Set {set_}---------------------------------------")
    #     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    #     print()
