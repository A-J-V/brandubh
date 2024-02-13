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


def ucb1(node, c: float = 2.0):
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


def check_quiescence_defender(node):
    king_loc = np.where(node.board == 3)
    king_r, king_c = king_loc[0].item(), king_loc[1].item()
    if king_c == 0 or king_c == 6:
        up_to_go = king_r - 1
        down_to_go = 11 - king_r
        if node.action_space[up_to_go*7*7 + king_r * 7 + king_c] == 1:
            print("Quiescence (UP) detected")
            print(node.board)
            return True
        elif node.action_space[down_to_go*7*7 + king_r * 7 + king_c] == 1:
            print("Quiescence (DOWN) detected")
            print(node.board)
            return True

    elif king_r == 0 or king_r == 6:
        left_to_go = 11 + king_c
        right_to_go = 23 - king_c
        if node.action_space[left_to_go * 7 * 7 + king_r * 7 + king_c] == 1:
            print("Quiescence (LEFT) detected")
            print(node.board)
            return True
        elif node.action_space[right_to_go * 7 * 7 + king_r * 7 + king_c] == 1:
            print("Quiescence (RIGHT) detected")
            print(node.board)
            return True
    else:
        return False

def rollout(node, caller):
    """
    Perform a random rollout from node to termination.
    The intermediate game states between node and termination won't persist beyond the rollout.
    """
    # We don't want to change the actual node that's being used in the game.
    # We clone it to make a dummy game branch and run the rollout from the clone.
    rollout_node = node.clone()
    while not rollout_node.is_terminal:
        # if rollout_node.player == 0 and check_quiescence_defender(rollout_node):
        #     rollout_node.winner = 0
        #     continue
        try:
            action = random.choice(rollout_node.unexpanded_children)
        except Exception as e:
            print(np.sum(rollout_node.action_space))
            print(rollout_node.unexpanded_children)
            raise e

        rollout_node = rollout_node.step(action)

    # Backprop the reward up the game tree based on whether the caller who started MCTS won the rollout.
    if caller == rollout_node.winner:
        node.backpropagate(value=1)
    else:
        node.backpropagate(value=-1)


def best_child(node):
    """Return the 'best' child according to number of visits."""
    visit_counts = [child.visits for child in node.children]
    max_visit_index = argmax(visit_counts)
    best = node.children[max_visit_index]
    best.reset_mcts()
    print(visit_counts)
    return best


def run_mcts(root_node, base_iter):
    num_legal_moves = np.sum(root_node.action_space == 1)
    num_iter = base_iter * num_legal_moves
    policy_counts = np.zeros_like(root_node.action_space)
    for iteration in range(num_iter):
        # 1) Selection
        node = root_node
        while not node.is_terminal and node.is_fully_expanded:
            need_policy = True if node == root_node else False
            node = select_node(node)
            if need_policy:
                policy_counts[node.action_index] += 1

        # 2) Expansion
        if not node.is_terminal and not node.is_fully_expanded:
            need_policy = True if node == root_node else False
            node = expand_child(node)
            if need_policy:
                policy_counts[node.action_index] += 1

        # 3) Simulation and Backpropagation
        rollout(node, caller=root_node.player)
    policy = policy_counts / num_iter
    root_node.policy = policy
    return best_child(root_node)
