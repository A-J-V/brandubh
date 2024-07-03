import pygame
from graphics import TILE_SCALE, BORDER_ADJ, COL_SPACING, ROW_SPACING


def identify_clicked_cell(x, y):
    """Given the (x, y) coordinates of a mouse click, identify the clicked cell, if any."""
    # Adjust the click to account for the border, then divide by spacing to get the correct tile index.
    x_adj, y_adj = x - BORDER_ADJ, y - BORDER_ADJ

    # If the mouse clicked on the board and not on the surrounding border, return the tile clicked
    if x_adj >= 0.0 and y_adj >= 0.0:
        tile_x, tile_y = int(x_adj / COL_SPACING), int(y_adj / ROW_SPACING)
        if tile_x < 7 and tile_y < 7:
            return tile_x, tile_y

    return float('nan'), float('nan')


def convert_clicks_to_action(x1, y1, x2, y2):
    """Given the (x1, y1) origin tile coords and the (x2, y2) destination coords, return an action value.

    A piece can only move one of 24 ways in this game, and these 24 moves are encoded by assigning a unique
    index to each possible move. The index of a move can be validated against the action space of the current
    game node to determine whether the action is legal.

    The encodings are: 0-5 is up, 6-11 is down, 12-17 is left, 18-23 is right.
    Example: 0 means the piece moves up by 1. 7 means the piece moves down by 2. 23 means the piece moves right by 6.
    """

    y_map = {
        # If y2 - y1 is positive, the movement is down
        1: 6,
        2: 7,
        3: 8,
        4: 9,
        5: 10,
        6: 11,

        # If y2 - y1 is negative, the movement is up
        -1: 0,
        -2: 1,
        -3: 2,
        -4: 3,
        -5: 4,
        -6: 5,
    }

    x_map = {
        # If x2 - x1 is positive, the movement is right
        1: 18,
        2: 19,
        3: 20,
        4: 21,
        5: 22,
        6: 23,

        # If x2 - x1 is negative, the movement is left
        -1: 12,
        -2: 13,
        -3: 14,
        -4: 15,
        -5: 16,
        -6: 17
    }

    # Pieces can only move horizontally or vertically. Run a quick validation.
    if (x1 != x2 and y1 != y2) or (x1 == x2 and y1 == y2):
        # There is diagonal movement or no movement, so this is an illegal move.
        return float('nan')

    elif x1 != x2:
        # Movement is horizontal.
        return x_map[x2 - x1]

    elif y1 != y2:
        # Movement is vertical
        return y_map[y2 - y1]
