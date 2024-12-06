import ai_utils
import episodes
import os


def generate_data(attacker_path: str,
                  defender_path: str,
                  value_path: str,
                  num_iters: int,
                  num_games: int,
                  output_path: str,
                  iteration: int,
                  chunks: int = 1,
                  temperature: float = 1.0,
                  process: int = 0,
                  ):
    if process == 0:
        print(f"Generating data in iteration {iteration}")
    # Step 1: Set up a new folder to store the resulting data
    folder_path = output_path + f"/set_{iteration}"

    # Step 2: Generate and record a batch of game using neural self-play
    for chunk in range(chunks):
        if process == 0:
            print(f"Chunk {chunk} generating...")
        winners = episodes.BatchNeuralSelfPlay(num_iters=num_iters,
                                               num_games=num_games,
                                               attacker_path=attacker_path,
                                               defender_path=defender_path,
                                               value_path=value_path,
                                               temperature=temperature,
                                               deterministic=False,
                                               ).play()
        for i, winner in enumerate(winners):
            ai_utils.NeuralGameRecorder().extract(winner).record().to_csv(folder_path + f"/game_{process}-{chunk}-{i}.csv", index=False)


def bench(attacker_path: str,
          defender_path: str,
          value_path: str,
          num_games: int,
          ):
    print("Benchmarking agents...")

    win_dict = {-1: 0, 0: 0, 1: 0}
    winners = episodes.BatchNeuralSelfPlay(num_iters=200,
                                           num_games=num_games,
                                           attacker_path=attacker_path,
                                           defender_path=defender_path,
                                           value_path=value_path,
                                           deterministic=True,
                                           ).play()
    for winner in winners:
        win_dict[winner.winner] += 1
    print(win_dict)