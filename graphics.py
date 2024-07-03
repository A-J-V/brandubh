import numpy as np
import pygame

# ROW and COL are reversed sometimes, this should be fixed
WIN_SIZE = 1700
BORDER_SCALE = 0.16
BORDER_ADJ = WIN_SIZE * BORDER_SCALE
TILE_SCALE = 0.15714
TILE_SIZE = WIN_SIZE * TILE_SCALE
KING_SCALE = TILE_SIZE * 1.2727
COL_SPACING = WIN_SIZE * 0.09714
ROW_SPACING = WIN_SIZE * 0.09285
KING_COL_ADJ = WIN_SIZE * 0.02343
KING_ROW_ADJ = WIN_SIZE * 0.04270
TILE_COL_OFFSET = WIN_SIZE * 0.12857
TILE_ROW_OFFSET = WIN_SIZE * 0.09285
RED_HIGHLIGHT = (240, 50, 50, 80)
BLUE_HIGHLIGHT = (50, 50, 240, 80)
RED_TILE = None
BLUE_TILE = None

BLK = (0, 0, 0)
BGCOLOR = BLK
board_image = pygame.image.load('./assets/brandubh_board.png')
board_image = pygame.transform.scale(board_image, (WIN_SIZE, WIN_SIZE))
viking_white = pygame.image.load('./assets/viking_white.png')
viking_white = pygame.transform.scale(viking_white, (TILE_SIZE, TILE_SIZE))
viking_black = pygame.image.load('./assets/viking_black.png')
viking_black = pygame.transform.scale(viking_black, (TILE_SIZE, TILE_SIZE))
viking_king = pygame.image.load('./assets/viking_king.png')
viking_king = pygame.transform.scale(viking_king, (KING_SCALE, KING_SCALE))
encoding_to_img = {1: viking_black,
                   2: viking_white,
                   3: viking_king,
                   }


def initialize():
    """Initialize Pygame and set up the camera and related variables."""
    pygame.init()
    main_display = pygame.display.set_mode(size=(WIN_SIZE, WIN_SIZE))
    pygame.display.set_caption('Hnefatafl')
    main_display.fill(BGCOLOR)
    rect = main_display.get_rect()
    main_display.blit(board_image, rect)
    global RED_TILE
    global BLUE_TILE
    RED_TILE = (pygame.Surface((TILE_SIZE//1.7, TILE_SIZE//1.7)).convert_alpha())
    RED_TILE.fill(RED_HIGHLIGHT)
    BLUE_TILE = pygame.Surface((TILE_SIZE//1.7, TILE_SIZE//1.7)).convert_alpha()
    BLUE_TILE.fill(BLUE_HIGHLIGHT)
    return main_display


def refresh(board: np.array,
            display: pygame.surface,
            ):
    """
    Update the camera.

    This function updates the view of the board that the player sees after each move. It can also be used for debugging
    by highlighting cache info.

    :param board: The 3D NumPy "board" array on which the game is being played.
    :param display: The Pygame surface on which all graphics are drawn.
    """
    display_rect = display.get_rect()
    display.blit(board_image, display_rect)
    piece_rect = pygame.Rect((BORDER_ADJ + 1 * COL_SPACING,
                              BORDER_ADJ + 1 * ROW_SPACING, TILE_SIZE, TILE_SIZE))
    display.blit(RED_TILE, piece_rect)
    for i, row in enumerate(range(board.shape[0])):
        for j, col in enumerate(range(board.shape[1])):
            piece = board[i, j].item()
            cc = TILE_COL_OFFSET
            rc = TILE_ROW_OFFSET
            if piece == 0 or piece == 4:
                continue
            elif piece == 3:
                cc -= KING_COL_ADJ
                rc -= KING_ROW_ADJ

            piece_sprite = encoding_to_img[board[i, j].item()]
            piece_rect = pygame.Rect((col * COL_SPACING + cc,
                                      row * ROW_SPACING + rc,
                                      TILE_SIZE,
                                      TILE_SIZE),
                                     )
            display.blit(piece_sprite, piece_rect)
    pygame.display.update()
