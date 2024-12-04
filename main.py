"""A minimal implementation of (simplified) Brandubh. Created to train RL agents."""

import ai_utils
import episodes
import multiprocessing
import os
import random
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

    # HumanVNeural(set=5).play()

    # Temperature annealing schedule
    def anneal_temperature(iteration, start=2.0, end=1.0, total_iterations=500):
        progress = min(iteration / total_iterations, 1.0)
        return start + progress * (end - start)


    sets = 400
    for set_ in range(0, sets):
        set_start = time.time()

        attacker_stem = f"attacker_cp"
        defender_stem = f"defender_cp"
        value_stem = f"value_cp"

        latest_attacker = attacker_stem + str(set_)
        latest_defender = defender_stem + str(set_)
        latest_value = value_stem + str(set_)

        # Step 1: Generate a new set of data

        processes = []
        num_processes = 10
        num_iters = 200
        num_games = 25
        output_path = "/home/alexander/Data/brandubh/neural_game_records"
        set_num = set_
        chunks = 10
        temperature = anneal_temperature(set_, start=1.5, end=1.1, total_iterations=sets)

        for i in range(num_processes):
            value_function = latest_value
            if i <= 2:
                attacker_agent = latest_attacker
                random_defender_num = random.randint(0, set_)
                defender_agent = defender_stem + str(random_defender_num)
            elif i <= 4:
                defender_agent = latest_defender
                random_attacker_num = random.randint(0, set_)
                attacker_agent = attacker_stem + str(random_attacker_num)
            else:
                attacker_agent = attacker_stem + str(set_)
                defender_agent = defender_stem + str(set_)



            process = multiprocessing.Process(target=generate_data,
                                              args=(
                                                  attacker_agent,
                                                  defender_agent,
                                                  value_function,
                                                  num_iters,
                                                  num_games,
                                                  output_path,
                                                  set_num,
                                                  chunks,
                                                  temperature,
                                                  i
                                              ))
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

        # Step 2: Train on the generated set of data

        data_paths = ["/home/alexander/Data/brandubh/neural_game_records/set_" + str(set_),
                      ]

        train_all(latest_attacker,
                  latest_defender,
                  latest_value,
                  data_paths=data_paths,
                  checkpoint_path="/home/alexander/Data/brandubh/checkpoints",
                  epochs=1,
                  iteration=set_,
                  device='cuda',
                  )

        # Step 3: Run a benchmark to ensure that each agent is still balanced

        bench(f"attacker_cp{set_ + 1}",
              f"defender_cp{set_ + 1}",
              f"value_cp{set_ + 1}",
              num_games=50,
              )

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"------------------------------- Completed Set {set_} --------------------------------------")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"------------------------- Time Taken: {time.time() - set_start} Seconds --------------------------------")
        print()
