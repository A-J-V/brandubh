import copy
import numpy as np
import random
from tqdm import tqdm

import core


def argmax(lst: list):
    """
    This is returns argmax with ties broken randomly.

    :param lst: List of action scores.
    :return: The argmax of the list of action scores with ties broken randomly.
    """
    if not lst:
        raise Exception("argmax was passed an empty list.")
    max_value = max(lst)
    ties = []
    for i, value in enumerate(lst):
        if value == max_value:
            ties.append(i)
    return random.choice(ties)


def ucb1(node, c: float = 5.0):
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


def rollout(node, caller):
    """
    Perform a random rollout from node to termination.
    The intermediate game states between node and termination won't persist beyond the rollout.
    """
    # We don't want to change the actual node that's being used in the game.
    # We clone it to make a dummy game branch and run the rollout from the clone.
    rollout_node = node.clone()
    while not rollout_node.is_terminal:
        try:
            action = random.choice(rollout_node.unexpanded_children)
        except Exception as e:
            print(np.sum(rollout_node.action_space))
            print(rollout_node.unexpanded_children)
            print(e)

        rollout_node = rollout_node.step(action)

    # The player of the terminal node's node.player attribute is the loser.
    # If the initiating node's player == the terminal node's player, that player lost.
    # Backprop a value accordingly. ??????????????
    if caller == rollout_node.player:
        node.backpropagate(value=-1)
    else:
        node.backpropagate(value=1)


def best_child(node):
    """Return the 'best' child according to number of visits."""
    visit_counts = [child.visits for child in node.children]
    max_visit_index = argmax(visit_counts)
    best = node.children[max_visit_index]
    best.reset_mcts()
    print(visit_counts)
    return best


def run_mcts(root_node, num_iter):
    for iteration in range(num_iter):
        # 1) Selection
        node = root_node
        while not node.is_terminal and node.is_fully_expanded:
            node = select_node(node)

        # 2) Expansion
        if not node.is_terminal and not node.is_fully_expanded:
            node = expand_child(node)

        # 3) Simulation and Backpropagation
        rollout(node, caller=root_node.player)
    return best_child(root_node)
