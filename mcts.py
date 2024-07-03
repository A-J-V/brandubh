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


def check_quiescence_defender(node):
    king_loc = np.where(node.board == 3)
    try:
        king_r, king_c = king_loc[0].item(), king_loc[1].item()
    except Exception as e:
        print(king_loc)
        print(node.board)
        print(node.piece_counts)
        raise e
    if king_c == 0 or king_c == 6:
        up_to_go = king_r - 1
        down_to_go = 11 - king_r
        if node.action_space[up_to_go * 7 * 7 + king_r * 7 + king_c] == 1:
            return True
        elif node.action_space[down_to_go * 7 * 7 + king_r * 7 + king_c] == 1:
            return True

    elif king_r == 0 or king_r == 6:
        left_to_go = 11 + king_c
        right_to_go = 23 - king_c
        if node.action_space[left_to_go * 7 * 7 + king_r * 7 + king_c] == 1:
            return True
        elif node.action_space[right_to_go * 7 * 7 + king_r * 7 + king_c] == 1:
            return True
    else:
        return False


def check_quiescence_attacker(node):
    king_loc = np.where(node.board == 3)
    king_r, king_c = king_loc[0].item(), king_loc[1].item()

    def scan_for_attackers(start_row, start_col):
        for (dr, dc) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            tr, tc = start_row, start_col
            # If we can walk from the blank tile in any direction and bump an attacker, that attacker could
            # capture the King. Therefore, the Attackers have imminent victory, return True.
            while True:
                tr += dr
                tc += dc
                if (0 <= tr <= 6) and (0 <= tc <= 6) and node.board[tr, tc] == 0:
                    continue
                elif (0 <= tr <= 6) and (0 <= tc <= 6) and node.board[tr, tc] == 1:
                    return True
                else:
                    break
        return False

    # The attackers can win immediately if the king has an attacker (or corner) next to him,
    # and the attackers can move an attacker to flank the king.
    if 0 < king_r < 6:
        if node.board[king_r - 1, king_c] == node.ATTACKER or node.board[king_r - 1, king_c] == node.CORNER:
            if node.board[king_r + 1, king_c] == node.BLANK and scan_for_attackers(king_r + 1, king_c):
                return True
        elif node.board[king_r + 1, king_c] == node.ATTACKER or node.board[king_r + 1, king_c] == node.CORNER:
            if node.board[king_r - 1, king_c] == node.BLANK and scan_for_attackers(king_r - 1, king_c):
                return True
    elif 0 < king_c < 6:
        if node.board[king_r, king_c - 1] == node.ATTACKER or node.board[king_r, king_c - 1] == node.CORNER:
            if node.board[king_r, king_c + 1] == node.BLANK and scan_for_attackers(king_r, king_c + 1):
                return True
        elif node.board[king_r, king_c + 1] == node.ATTACKER or node.board[king_r, king_c + 1] == node.CORNER:
            if node.board[king_r, king_c - 1] == node.BLANK and scan_for_attackers(king_r, king_c - 1):
                return True
    else:
        return False


def rollout(node, caller, use_quiescence):
    """
    Perform a random rollout from node to termination.
    The intermediate game states between node and termination won't persist beyond the rollout.
    """
    # We don't want to change the actual node that's being used in the game.
    # We clone it to make a dummy game branch and run the rollout from the clone.
    rollout_node = node.clone()

    # If the node that was selected is already terminal, this loop will be skipped.
    while not rollout_node.is_terminal:
        if use_quiescence:
            if rollout_node.player == 0 and check_quiescence_defender(rollout_node):
                rollout_node.winner = 0
                continue
            elif rollout_node.player == 1 and check_quiescence_attacker(rollout_node):
                rollout_node.winner = 1
                continue
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
        if node.is_terminal:
            use_quiescence = False
        elif node.player == 0 and check_quiescence_defender(node):
            use_quiescence = False
        elif node.player == 1 and check_quiescence_attacker(node):
            use_quiescence = False
        else:
            use_quiescence = True
        rollout(node, caller=root_node.player, use_quiescence=use_quiescence)
    root_node.policy = policy_counts
    root_node.legal_actions = root_node.action_space
    return best_child(root_node)
