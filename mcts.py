import numpy as np
import random


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


def expand_all_children(node):
    """Take all children from node.unexpanded_children and expand them into new nodes, return a random one."""
    for i, action in enumerate(node.unexpanded_children[:]):
        node.step(action)
    node.unexpanded_children = []
    return random.choice(node.children)


def rollout(node):
    """
    Perform a random rollout from node to termination.
    The intermediate game states between node and termination won't persist beyond the rollout.
    """
    # We don't want to change the actual node that's being used in the game.
    # We clone it to make a dummy game branch and run the rollout from the clone.
    rollout_node = node.clone()

    # If the node that was selected is already terminal, this loop will be skipped.
    while not rollout_node.is_terminal:
        try:
            action = random.choice(rollout_node.unexpanded_children)
        except Exception as e:
            print(np.sum(rollout_node.action_space))
            print(rollout_node.unexpanded_children)
            raise e

        rollout_node = rollout_node.step(action)

    # Backprop the reward up the game tree based on whether the node from which the rollout began won.
    # NOTE! The value of a node is the estimated value of choosing that node as an action by the PREVIOUS player.
    # Therefore, if we start a rollout from an expanded defender node and the defenders win, that is NEGATIVE,
    # because the attackers don't want to choose that node if the defenders are likely to win from it.
    if node.player == rollout_node.winner:
        node.backpropagate(value=0)
    else:
        node.backpropagate(value=1)


def best_child(node):
    """Return the 'best' child according to number of visits."""
    visit_counts = [child.visits for child in node.children]
    max_visit_index = argmax(visit_counts)
    best = node.children[max_visit_index]
    best.reset_mcts()
    return best


def run_mcts(root_node, base_iter: int):
    """Run the Monte Carlo Tree Search algorithm.

    :param root_node: The root node of the MCTS tree.
    :type root_node: GameNode
    :param base_iter: The number of base iterations.
    :type base_iter: int
    """
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
            node = expand_all_children(node)
            if need_policy:
                policy_counts[node.action_index] += 1

        # 3) Simulation and Backpropagation
        rollout(node)

    root_node.policy = policy_counts
    root_node.legal_actions = root_node.action_space
    return best_child(root_node)
