"""A minimal implementation of (simplified) Brandubh. Created to train RL agents."""

import ai_utils
import sys
import episodes


if __name__ == '__main__':
    num_games = sys.argv[1]
    ui = sys.argv[2]
    save_path = sys.argv[3]
    for i in range(int(num_games)):
        game = episodes.MCTSGame(base_iter=10)
        result = game.play()
        ai_utils.GameRecorder().extract(result).record().to_csv(f"{save_path}/game{ui}-{i}.csv", index=False)
