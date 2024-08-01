import ai
from core import *
from math import isnan
import mcts
import graphics
import pygame
from input import identify_clicked_cell, convert_clicks_to_action
import time
from ai import *


class RandomSelfPlay:
    def __init__(self,
                 attacker=None,
                 defender=None,
                 device='cpu',
                 uid=0,
                 ):
        self.uid = uid
        self.game = GameNode()
        self.device = device
        if attacker is None:
            self.attacker = ai.RandomAI(player=1)
        else:
            self.attacker = attacker.to(self.device)
        if defender is None:
            self.defender = ai.RandomAI(player=0)
        else:
            self.defender = defender.to(self.device)

    def play(self):
        while self.game.winner == -1:
            action_space = self.game.action_space

            if self.game.player == 1:
                action_selected, prob = self.attacker.select_action(self.game.board,
                                                                    action_space,
                                                                    device=self.device)
                _ = self.defender.predict_value(self.game.board, device=self.device)
            elif self.game.player == 0:
                action_selected, prob = self.defender.select_action(self.game.board,
                                                                    action_space,
                                                                    device=self.device)
                _ = self.attacker.predict_value(self.game.board, device=self.device)
            else:
                raise Exception("Unknown player")
            self.game = self.game.step(action_selected)

        return self.game


class MCTSSelfPlay:
    """An episode designed to test and debug MCTS or generating data for network training."""
    def __init__(self, base_iter, show=False, board=None):
        if board is None:
            self.game = GameNode()
        else:
            self.game = GameNode(board=board)
        self.base_iter = base_iter
        self.show = show

    def play(self):
        """Play the game until termination."""
        if self.show:
            display = graphics.initialize()
            graphics.refresh(self.game.board, display)

        while not self.game.is_terminal:
            self.game = mcts.run_mcts(self.game, base_iter=self.base_iter)
            if self.show:
                graphics.refresh(self.game.board, display)
                time.sleep(0.5)

        return self.game


class MCTSVRandom:
    def __init__(self, base_iter, show=False, board=None):
        if board is None:
            self.game = GameNode()
        else:
            self.game = GameNode(board=board)
        self.base_iter = base_iter
        self.show = show
        self.attacker = ai.RandomAI(player=1)
        self.defender = "MCTS"
        self.device = 'cpu'

    def play(self):
        """Play the game until termination."""
        if self.show:
            display = graphics.initialize()
            graphics.refresh(self.game.board, display)

        while not self.game.is_terminal:

            if self.game.player == 1:
                action_space = self.game.action_space
                action_selected, prob = self.attacker.select_action(self.game.board,
                                                                    action_space,
                                                                    device=self.device)
                _ = self.attacker.predict_value(self.game.board, device=self.device)
                self.game = self.game.step(int(action_selected))

            elif self.game.player == 0:
                self.game = mcts.run_mcts(self.game, base_iter=self.base_iter)
            else:
                raise Exception("Unknown player")

            if self.show:
                graphics.refresh(self.game.board, display)
                time.sleep(0.5)

        return self.game


class HumanVMCTS:
    """A game in which a human player will play against an MCTS-based AI."""
    def __init__(self, base_iter=10, human=1):
        self.game = GameNode()
        self.base_iter = base_iter
        self.human = human
        self.piece_to_player = {0: -1,
                                4: -1,
                                1: 1,
                                2: 0,
                                3: 0,
                                }
        print(f"Human is playing as {'attacker' if human == 1 else 'defender'}.")

    def play(self):
        """Play the game until termination, alternating between MCTS and human input."""
        selected = None
        display = graphics.initialize()
        graphics.refresh(self.game.board, display, selected)

        while not self.game.is_terminal:
            print(f"Player: {self.game.player}")

            # The human's turn
            if self.game.player == self.human:
                waiting_for_human = True

                # The game will continuously poll until the human has selected a legal move.
                while waiting_for_human:

                    # Check events for mouse clicks
                    for event in pygame.event.get():

                        # If the user clicks a valid piece, select that piece
                        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and selected is None:
                            x, y = event.pos
                            tile_col, tile_row = identify_clicked_cell(x, y)

                            # Validate that the user clicked in a cell
                            if isnan(tile_col) or isnan(tile_row):
                                print('nan')
                                continue

                            # Check what piece type the user clicked
                            piece = self.game.board[tile_row, tile_col]

                            # Check if that piece is valid (does it belong to the human player?)
                            valid_piece = self.piece_to_player[piece] == self.human

                            # If the piece is valid, select it and highlight the cell to show that it is selected.
                            if valid_piece:
                                selected = (tile_col, tile_row)
                                graphics.refresh(self.game.board, display, selected)

                        # The user already selected a valid piece, now a valid move must be selected.
                        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and selected is not None:
                            x, y = event.pos
                            tile_col, tile_row = identify_clicked_cell(x, y)

                            # Validate that the user clicked in a cell
                            if isnan(tile_col) or isnan(tile_row):
                                print('nan')
                                continue

                            # Convert the cell that the clicked into a move using the game's move encodings
                            move = convert_clicks_to_action(selected[0], selected[1], tile_col, tile_row)

                            # Validate that the move is possible
                            if isnan(move):
                                print('nan')
                                continue

                            # Ravel the move selection to prepare to execute it and step into the next game node
                            action = np.ravel_multi_index([move, selected[1], selected[0]],
                                                          dims=(24, 7, 7),
                                                          ).item()

                            # Validate that the selected action is a legal move in the current action space
                            if self.game.action_space[action] == 1:

                                # The action is legal! Execute it, update the display, and end the human's turn.
                                self.game = self.game.step(action)
                                waiting_for_human = False
                                selected = None
                            else:
                                continue

                        # The user right-clicked, so deselect the piece and let them choose a different one.
                        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3 and selected is not None:
                            selected = None
                            graphics.refresh(self.game.board, display)

            # AI's turn
            else:
                self.game = mcts.run_mcts(self.game, base_iter=self.base_iter)

            # Update the display
            graphics.refresh(self.game.board, display)
            print(self.game.board)

        print(f"Winner: {self.game.winner}")


class MCTSVDeepAgent:
    """An episode designed to test and debug deep RL agents."""
    def __init__(self, base_iter, show=False, board=None, deep_player=1):
        if board is None:
            self.game = GameNode()
        else:
            self.game = GameNode(board=board)
        self.base_iter = base_iter
        self.show = show
        self.deep_player = deep_player
        if deep_player == 1:
            self.model = load_agent(player='attacker')
        else:
            self.model = load_agent(player='defender')

    def play(self):
        """Play the game until termination."""
        if self.show:
            display = graphics.initialize()
            graphics.refresh(self.game.board, display)

        while not self.game.is_terminal:
            if self.game.player == self.deep_player:
                game_state = self.game.board.flatten()
                game_state = torch.Tensor(game_state).float().unsqueeze(0)
                action_space = self.game.action_space
                action_space = torch.Tensor(action_space).float().unsqueeze(0)
                action = torch.argmax(self.model.predict_probs(game_state.to('cuda'), action_space.to('cuda'))).item()
                self.game = self.game.step(action)
            else:
                self.game = mcts.run_mcts(self.game, base_iter=self.base_iter)
            if self.show:
                graphics.refresh(self.game.board, display)
                time.sleep(0.5)

        return self.game

class DeepSelfPlay:
    """An episode designed to test and debug deep RL agents."""
    def __init__(self, show=False, board=None):
        if board is None:
            self.game = GameNode()
        else:
            self.game = GameNode(board=board)
        self.show = show
        self.attacker = load_agent(player='attacker')
        self.defender = load_agent(player='defender')
        self.turn = 0

    def play(self):
        """Play the game until termination."""
        if self.show:
            display = graphics.initialize()
            graphics.refresh(self.game.board, display)

        while not self.game.is_terminal:
            self.turn += 1
            if self.game.player == 1:
                game_state = self.game.board.flatten()
                game_state = torch.Tensor(game_state).float().unsqueeze(0)
                action_space = self.game.action_space
                action_space = torch.Tensor(action_space).float().unsqueeze(0)
                action = torch.argmax(self.attacker.predict_probs(game_state.to('cuda'), action_space.to('cuda'))).item()
                self.game = self.game.step(action)
            else:
                game_state = self.game.board.flatten()
                game_state = torch.Tensor(game_state).float().unsqueeze(0)
                action_space = self.game.action_space
                action_space = torch.Tensor(action_space).float().unsqueeze(0)
                action = torch.argmax(self.defender.predict_probs(game_state.to('cuda'), action_space.to('cuda'))).item()
                self.game = self.game.step(action)
            if self.show:
                graphics.refresh(self.game.board, display)
                time.sleep(0.5)

            # Draw if the game gets stuck
            if self.turn > 100:
                self.game.winner = -1
                return self.game

        return self.game
