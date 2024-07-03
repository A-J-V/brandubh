import pygame
from graphics import TILE_SCALE, BORDER_ADJ, COL_SPACING, ROW_SPACING


def identify_clicked_cell(x, y):
    """Given the (x, y) coordinates of a mouse click, identify the clicked cell, if any."""
    # Assumes that the window is square
    win_width, win_height = pygame.display.get_surface().get_size()
    tile_size = win_width * TILE_SCALE

    # Adjust the click to account for the border, then divide by spacing to get the correct tile index.
    tile_x, tile_y = int((x - BORDER_ADJ) / COL_SPACING), int((y - BORDER_ADJ) / ROW_SPACING)

    return tile_x, tile_y

