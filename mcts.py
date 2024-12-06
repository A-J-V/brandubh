import numpy as np
import random
import torch
from typing import Callable


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
    """Calculate the UCB1 value."""
    if node.visits == 0:
        return float('inf')
    else:
        return (
                (node.value / node.visits) +
                c * (np.log(node.parent.visits) / node.visits) ** 0.5
        )


def PUCT(node, c: float = 2.0):
    """Calculate the PUCT value based on AlphaZero."""
    if node.visits > 0:
        q_value = node.value / node.visits
    else:
        q_value = 0

    puct_value = q_value + c * node.prior * ((node.parent.visits ** 0.5) / (1 + node.visits))

    return puct_value


def select_node(node, func: Callable = ucb1):
    best_value = -float('inf')
    best_child = None
    for child in node.children:
        value = func(child)
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


def expand_predict_all(node, model: torch.nn.Module, device: str):
    """Expand all child nodes, assign prior values using policy prediction, and return the highest value."""
    action_space = node.action_space
    action_tensor = torch.tensor(action_space).float().unsqueeze(0).to(device)
    state = node.board.flatten()
    state_tensor = torch.tensor(state).float().unsqueeze(0).to(device)
    with torch.no_grad():
        pred_prob = model.predict_probs(state_tensor, action_tensor).cpu().squeeze().numpy()
    for i, action in enumerate(node.unexpanded_children[:]):
        child = node.step(action)
        child.prior = pred_prob[action]
    node.unexpanded_children = []
    return random.choice(node.children)


def batch_predict_all(nodes, model0: torch.nn.Module, model1: torch.nn.Module, device: str):
    """Expand all child nodes and run batch inference to assign priors to all."""

    # Separate nodes based on player
    indices0, indices1 = [], []
    nodes0, nodes1 = [], []
    for idx, node in enumerate(nodes):
        if node.player == 0:
            indices0.append(idx)
            nodes0.append(node)
        else:
            indices1.append(idx)
            nodes1.append(node)

    # Prepare a list to hold predictions in the original order
    pred_prob_list = [None] * len(nodes)

    # Process nodes for player 0
    if nodes0:
        action_space0 = np.array([node.action_space for node in nodes0])
        action_tensor0 = torch.tensor(action_space0).float().to(device)
        state0 = np.array([node.board.flatten() for node in nodes0])
        state_tensor0 = torch.tensor(state0).float().to(device)
        with torch.no_grad():
            pred_prob0 = model0.predict_probs(state_tensor0, action_tensor0).cpu().numpy()
        for i, idx in enumerate(indices0):
            pred_prob_list[idx] = pred_prob0[i]

    # Process nodes for player 1
    if nodes1:
        action_space1 = np.array([node.action_space for node in nodes1])
        action_tensor1 = torch.tensor(action_space1).float().to(device)
        state1 = np.array([node.board.flatten() for node in nodes1])
        state_tensor1 = torch.tensor(state1).float().to(device)
        with torch.no_grad():
            pred_prob1 = model1.predict_probs(state_tensor1, action_tensor1).cpu().numpy()
        for i, idx in enumerate(indices1):
            pred_prob_list[idx] = pred_prob1[i]

    # Assign priors and select choices
    choices = []
    for i, node in enumerate(nodes):
        for action in node.unexpanded_children[:]:
            child = node.step(action)
            child.prior = pred_prob_list[i][action]
        node.unexpanded_children = []
        choices.append(random.choice(node.children))
    return choices


def rollout(node):
    """
    Perform a random rollout from node to termination.
    The intermediate game states between node and termination won't persist beyond the rollout.
    """
    # We don't want to change the actual node that's being used in the game.
    # We clone it to make a dummy game branch and run the rollout from the clone.
    rollout_node = node.clone()

    # If the node that was selected is already terminal, this loop will be skipped.
    max_moves = 50000
    move_num = 0
    while not rollout_node.is_terminal:
        if move_num >= max_moves:
            print(f"Rollout exceeded max of {max_moves}!")
            node.backpropagate(value=0)
            return
        try:
            action = random.choice(rollout_node.unexpanded_children)
        except Exception as e:
            print(np.sum(rollout_node.action_space))
            print(rollout_node.unexpanded_children)
            raise e

        rollout_node = rollout_node.step(action)
        move_num += 1

    # Backprop the reward up the game tree based on whether the node from which the rollout began won.
    # NOTE! The value of a node is the estimated value of choosing that node as an action by the PREVIOUS player.
    # Therefore, if we start a rollout from an expanded defender node and the defenders win, that is NEGATIVE,
    # because the attackers don't want to choose that node if the defenders are likely to win from it.
    if node.player == rollout_node.winner:
        node.backpropagate(value=0)
    else:
        node.backpropagate(value=1)


def pseudo_rollout(node, model, device):
    """Use a neural network value function to predict state value"""
    player = node.player
    state = node.board.flatten()
    state_tensor = torch.tensor(state).float().unsqueeze(0).to(device)
    player_tensor = torch.tensor(player).float().to(device)
    with torch.no_grad():
        value_pred = model(state_tensor, player_tensor)
    value_pred = value_pred.item()

    if player == 1:
        node.backpropagate(1 - value_pred)
    elif player == 0:
        node.backpropagate(value_pred)
    else:
        raise Exception("Invalid player")


def batch_pseudo_rollout(nodes, model, device):
    """Use a neural network value function to predict a batch of state values."""
    batch_size = len(nodes)
    player = np.array([node.player for node in nodes])
    state = np.array([node.board.flatten() for node in nodes])
    state_tensor = torch.tensor(state).float().to(device)
    player_tensor = torch.tensor(player).float().unsqueeze(1).to(device)
    with torch.no_grad():
        value_preds = model(state_tensor, player_tensor).squeeze(1)

    for i in range(batch_size):
        if player[i] == 1:
            nodes[i].backpropagate(1 - value_preds[i].item())
        elif player[i] == 0:
            nodes[i].backpropagate(value_preds[i].item())
        else:
            raise Exception("Invalid player")


def best_child(node):
    """Return the 'best' child according to number of visits."""
    visit_counts = [child.visits for child in node.children]
    max_visit_index = argmax(visit_counts)
    best = node.children[max_visit_index]
    best.reset_mcts()
    return best


def probabilistic_child(node, temperature=1.0):
    """Return a child selected probabilistically based on policy."""
    visit_counts = np.array([child.visits for child in node.children])

    # Apply temperature adjustment
    if temperature != 1.0:
        adjusted_counts = visit_counts ** (1 / temperature)
    else:
        adjusted_counts = visit_counts

    visit_probs = adjusted_counts / adjusted_counts.sum()

    index_selected = np.random.choice(len(visit_probs), p=visit_probs)

    best = node.children[index_selected]

    best.reset_mcts()

    return best


def run_mcts(root_node, num_iter: int):
    """Run the Monte Carlo Tree Search algorithm.

    :param root_node: The root node of the MCTS tree.
    :type root_node: GameNode
    :param num_iter: The number of base iterations.
    :type num_iter: int
    """
    num_legal_moves = np.sum(root_node.action_space == 1)
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
        rollout(node)

    root_node.policy = policy_counts
    root_node.legal_actions = root_node.action_space
    return best_child(root_node)


def run_neural_mcts(root_node, policy_function, value_function, device: str, base_iter: int):
    """Run the Monte Carlo Tree Search algorithm with deep learning guidance inspired by AlphaZero.

    NOTE: This does NOT probabilistic select the next node, so it is not suited for generated training data!
    This is the function to use for neural search in deployment!

    :param root_node: The root node of the MCTS tree.
    :type root_node: GameNode
    :param policy_function: The Pytorch policy network.
    :type policy_function: nn.Module
    :param value_function: The Pytorch value network.
    :type value_function: nn.Module
    :type value_function: nn.Module
    :param device: The device on which the value_function should be placed.
    :type device: str
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
            node = select_node(node, func=PUCT)
            if need_policy:
                policy_counts[node.action_index] += 1

        # 2) Expansion
        if not node.is_terminal and not node.is_fully_expanded:
            need_policy = True if node == root_node else False
            node = expand_predict_all(node,
                                      model=policy_function,
                                      device=device)
            if need_policy:
                policy_counts[node.action_index] += 1

        # 3) Simulation and Backpropagation
        pseudo_rollout(node, model=value_function, device=device)

    # Assign some finalized attributes to the root_node before returning the next node selection.
    state = root_node.board.flatten()
    player = root_node.player
    state_tensor = torch.tensor(state).float().unsqueeze(0).to(device)
    player_tensor = torch.tensor(player).float().to(device)
    with torch.no_grad():
        value_pred = value_function(state_tensor, player_tensor)
    value_pred = value_pred.item()

    root_node.value_estimate = value_pred
    root_node.policy = policy_counts
    root_node.legal_actions = root_node.action_space

    next_node = best_child(root_node)
    root_node.selected_action = next_node.action_index
    root_node.selected_action_prob = policy_counts[next_node.action_index] / num_iter

    return next_node


def batch_neural_mcts(root_nodes,
                      attacker_policy_function,
                      defender_policy_function,
                      value_function,
                      device: str,
                      num_iters: int,
                      temperature: float = 1.0,
                      deterministic: bool = False,
                      ):
    """Run the Monte Carlo Tree Search algorithm with deep learning guidance inspired by AlphaZero.

    :param root_nodes: The root node of the MCTS tree.
    :type root_nodes: List[GameNode]
    :param attacker_policy_function: The attacker Pytorch policy network.
    :type attacker_policy_function: nn.Module
    :param defender_policy_function: The defender Pytorch policy network.
    :type defender_policy_function: nn.Module
    :param value_function: The Pytorch value network.
    :type value_function: nn.Module
    :param device: The device on which the value_function should be placed.
    :type device: str
    :param num_iters: The number of MCTS iterations.
    :type num_iters: int
    :param temperature: The temperature applied to action selection
    :type temperature: float
    """

    attacker_policy_function.eval()
    defender_policy_function.eval()
    value_function.eval()

    batch_policy_counts = np.zeros((len(root_nodes), len(root_nodes[0].action_space)))

    for iteration in range(num_iters):

        # 1) Selection
        # Make a list that will contain the selected node from each game
        selected_nodes = []

        # For every game root node, run selection as normal
        for i, root_node in enumerate(root_nodes):
            node = root_node
            while not node.is_terminal and node.is_fully_expanded:
                need_policy = True if node == root_node else False
                node = select_node(node, func=PUCT)
                if need_policy:
                    # Update the policy counts as usual, but be aware of batch number
                    batch_policy_counts[i, node.action_index] += 1
            selected_nodes.append(node)

        # 2) Expansion
        # Make a list of nodes that need expansion and prediction
        expansion_queue = []
        node_info = []
        for i, node in enumerate(selected_nodes):
            if not node.is_terminal and not node.is_fully_expanded:
                need_policy = True if node == root_nodes[i] else False
                expansion_queue.append(node)
                node_info.append((i, need_policy))

        # If there are any nodes that needed to be expanded, expand them now and selected a child
        if expansion_queue:
            choices = batch_predict_all(expansion_queue, model0=defender_policy_function, model1=attacker_policy_function, device=device)

            for i, (idx, need_policy) in enumerate(node_info):
                selected_nodes[idx] = choices[i]
                if need_policy:
                    batch_policy_counts[idx, selected_nodes[idx].action_index] += 1

        # 3) Simulation and Backpropagation
        batch_pseudo_rollout(selected_nodes, model=value_function, device=device)

    # Assign some finalized attributes to the root_nodes before returning the next node selections.
    player = np.array([node.player for node in root_nodes])
    state = np.array([node.board.flatten() for node in root_nodes])
    state_tensor = torch.tensor(state).float().to(device)
    player_tensor = torch.tensor(player).float().unsqueeze(1).to(device)
    with torch.no_grad():
        value_preds = value_function(state_tensor, player_tensor).squeeze(1).cpu()

    next_nodes = []
    for i, root_node in enumerate(root_nodes):
        root_node.value_estimate = value_preds[i].item()
        root_node.policy = batch_policy_counts[i, :]
        root_node.legal_actions = root_node.action_space

        if deterministic:
            next_node = best_child(root_node)
        else:
            next_node = probabilistic_child(root_node, temperature=temperature)
        root_node.selected_action = next_node.action_index
        root_node.selected_action_prob = batch_policy_counts[i, next_node.action_index] / num_iters

        next_nodes.append(next_node)

    return next_nodes
