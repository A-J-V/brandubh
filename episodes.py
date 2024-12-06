import ai
from core import *
import gc
from math import isnan
import mcts
import graphics
import pygame
from input import identify_clicked_cell, convert_clicks_to_action
import time
from ai import *

class MCTSSelfPlay:
    """An episode designed to test and debug MCTS or generating data for network training."""

    def __init__(self, num_iter, show=False, board=None):
        if board is None:
            self.game = GameNode()
        else:
            self.game = GameNode(board=board)
        self.num_iter = num_iter
        self.show = show

    def play(self):
        """Play the game until termination."""
        if self.show:
            display = graphics.initialize()
            graphics.refresh(self.game.board, display)

        turns = 1
        while not self.game.is_terminal:
            self.game = mcts.run_mcts(self.game, num_iter=self.num_iter)
            if self.show:
                graphics.refresh(self.game.board, display)
                time.sleep(0.5)
            turns += 1

            if turns >= 100:
                return 'stall'

        return self.game


class HumanVMCTS:
    """A game in which a human player will play against an MCTS-based AI."""

    def __init__(self, num_iter=10, human=1):
        self.game = GameNode()
        self.num_iter = num_iter
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
                self.game = mcts.run_mcts(self.game, num_iter=self.num_iter)

            # Update the display
            graphics.refresh(self.game.board, display)
            print(self.game.board)

        print(f"Winner: {self.game.winner}")


class HumanVNeural:
    """A game in which a human player will play against a Deep RL-based AI."""

    def __init__(self, num_iter=100, human=1, set=0):
        self.game = GameNode()
        self.num_iter = num_iter
        self.human = human
        self.piece_to_player = {0: -1,
                                4: -1,
                                1: 1,
                                2: 0,
                                3: 0,
                                }
        self.policy_function = ai.load_agent(f'defender_cp{set}', player=0)
        self.value_function = ai.load_value_function(f'value_cp{set}')
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
                self.game = mcts.run_neural_mcts(self.game,
                                                 policy_function=self.policy_function,
                                                 value_function=self.value_function,
                                                 device='cuda',
                                                 base_iter=100)

            # Update the display
            graphics.refresh(self.game.board, display)
            print(self.game.board)

        print(f"Winner: {self.game.winner}")


class MCTSVDeepAgent:
    """An episode designed to test and debug deep RL agents."""

    def __init__(self, num_iter, show=False, board=None, deep_player=0):
        if board is None:
            self.game = GameNode()
        else:
            self.game = GameNode(board=board)
        self.num_iter = num_iter
        self.show = show
        self.deep_player = deep_player
        if deep_player == 1:
            self.policy_function = load_agent(model_path='attacker_cp27', player=1)
        else:
            self.policy_function = load_agent(model_path='defender_cp27', player=0)
        self.value_function = load_value_function(value_path='value_cp27')

    def play(self):
        """Play the game until termination."""
        if self.show:
            display = graphics.initialize()
            graphics.refresh(self.game.board, display)

        while not self.game.is_terminal:
            if self.game.player == self.deep_player:
                self.game = mcts.run_neural_mcts(self.game,
                                                 policy_function=self.policy_function,
                                                 value_function=self.value_function,
                                                 device='cuda',
                                                 base_iter=self.num_iter)
            else:
                self.game = mcts.run_mcts(self.game, num_iter=self.num_iter)
            if self.show:
                graphics.refresh(self.game.board, display)
                time.sleep(0.5)

        return self.game


class BatchNeuralSelfPlay:
    """Generate bulk Deep RL gameplay data."""

    def __init__(self,
                 num_iters,
                 num_games,
                 attacker_path,
                 defender_path,
                 value_path,
                 show=False,
                 temperature=1.0,
                 deterministic=False,
                 ):

        self.games = [GameNode() for _ in range(num_games)]
        self.num_iters = num_iters
        self.show = show
        self.turn = 0
        self.temperature = temperature
        self.deterministic = deterministic

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.attacker_policy_function = ai.load_agent(attacker_path, player=1)
        self.defender_policy_function = ai.load_agent(defender_path, player=0)
        self.value_function = ai.load_value_function(value_path)

    def play(self):
        """Play the games until they all terminate."""
        if self.show:
            display = graphics.initialize()
            graphics.refresh(self.games[0].board, display)

        live_games = [game for game in self.games if not game.is_terminal]
        terminal_games = []
        while live_games:
            self.turn += 1

            live_games = mcts.batch_neural_mcts(live_games,
                                                attacker_policy_function=self.attacker_policy_function,
                                                defender_policy_function=self.defender_policy_function,
                                                value_function=self.value_function,
                                                device=self.device,
                                                num_iters=self.num_iters,
                                                temperature=self.temperature,
                                                deterministic=self.deterministic,
                                                )
            if self.show:
                graphics.refresh(live_games[0].board, display)
                print(live_games[0].board)
                time.sleep(2.0)

            for game in live_games:
                if game.is_terminal:
                    terminal_games.append(game)
            live_games = [game for game in live_games if not game.is_terminal]

            gc.collect()

            # Draw if the games get stuck
            if self.turn > 100:
                for game in live_games:
                    game.winner = -1

        return terminal_games
