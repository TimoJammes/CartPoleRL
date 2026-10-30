import pygame
import numpy as np

#region colors
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREY = (169, 169, 169)
FIRE = (252, 82, 0)
BROWN = (90, 56, 37)
LIGHT_BLUE = (137, 207, 240)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)
PINK = (255, 192, 203)
CYAN = (0, 255, 255)
DARK_GREEN = (0, 100, 0)
LIGHT_GREEN = (144, 238, 144)
DARK_BLUE = (0, 0, 139)
LIGHT_GREY = (211, 211, 211)
DARK_GREY = (105, 105, 105)
MAGENTA = (255, 0, 255)
NAVY = (0, 0, 128)
TEAL = (0, 128, 128)
GOLD = (255, 215, 0)
SILVER = (192, 192, 192)
MAROON = (128, 0, 0)
OLIVE = (128, 128, 0)
INDIGO = (75, 0, 130)
TURQUOISE = (64, 224, 208)
#endregion

#region pygame vars
WIDTH, HEIGHT = 600, 600
AREA = (WIDTH, HEIGHT)
FPS = 30
screen = pygame.display.set_mode(AREA)
clock = pygame.time.Clock()
#endregion


rng = np.random.default_rng(seed=0)