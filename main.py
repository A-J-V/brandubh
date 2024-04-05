"""A minimal implementation of (simplified) Brandubh. Created to train RL agents."""

import ai_utils
import sys
import episodes


if __name__ == '__main__':

    game = episodes.HumanVMcts(human=1)
    game.play()