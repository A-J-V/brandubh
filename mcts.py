import copy
import numpy as np
import random

import core


def ucb1(node, c: float = 1.0):
    """Calculate the UCB1 value"""
    if node.visits == 0:
        return float('inf')
    else:
        return (
                (node.value / node.visits) +
                c * (np.log(node.parent.visits) / node.visits) ** 0.5
        )


def select_node(node):
    best_value = -float('inf')
    best_child = None
    for child in node.children:
        value = ucb1(child)
        if value > best_value:
            best_value = value
            best_child = child
    return best_child


def expand_child(node):
    """Take a child from node.unexpanded_children, expand it into a new node, return a reference to it."""
    # Pop the action that will be used to spawn a new child. This is not random currently.
    action = node.unexpanded_children.pop()
    expanded_child = node.step(action)
    return expanded_child


def rollout(node):
    """
    Perform a random rollout from node to termination.
    The intermediate game states between node and termination won't persist beyond the rollout.
    """
    # We don't want to change the actual node that's being used in the game.
    # We clone it to make a dummy game branch and run the rollout from the clone.
    rollout_node = copy.deepcopy(node)
    while rollout_node.terminal == -1:
        action = random.choice(rollout_node.unexpanded_children)
        rollout_node = rollout_node.step(action)

    # The terminal node is the loser's turn always. This is due to the order of operations in the game.
    # A player makes a move which spawns a new node in the game tree, which would be the opponent's turn, and
    # then it is checked for termination. So the terminal node is always the loser's node with value -1.
    rollout_node.backpropagate(value=-1)
