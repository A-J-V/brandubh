from episodes import MCTSSelfPlay, MCTSVRandom
from tqdm import tqdm


imminent = """\
X..A..X
...AD..
...D..K
AAD.DAA
...D...
...A...
X..A..X"""


def test_mcts_self_play():
    win_counts = {0: 0, 1: 0}
    for _ in tqdm(range(100)):
        game = MCTSSelfPlay(num_iter=25, show=True).play()
        win_counts[game.winner] += 1

    print(win_counts)


def test_mcts_v_random_self_play():
    win_counts = {0: 0, 1: 0}
    for _ in tqdm(range(50)):
        game = MCTSVRandom(base_iter=25, show=True).play()
        win_counts[game.winner] += 1

    print(win_counts)
