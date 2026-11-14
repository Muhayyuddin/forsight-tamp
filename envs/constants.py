import os
from collections import namedtuple

# ASSET DIRS
ASSETS_DIR = os.path.join(os.path.join(os.path.dirname(__file__), os.pardir), "assets")


# COLOR
RED = (209/255, 70/255, 70/255, 1)
GREEN = (0, 0.85, 0, 1)
# BLUE = (0, 0, 1, 1)
BLUE = (92/255, 125/255, 255/255, 1)
BLACK = (0, 0, 0, 1)
WHITE = (1, 1, 1, 1)
BROWN = (0.396, 0.263, 0.129, 1)
TAN = (0.824, 0.706, 0.549, 1)
GREY = (0.5, 0.5, 0.5, 1)
YELLOW = (255/255, 199/255, 51/255, 1)
CYAN = (77/255, 224/255, 224/255, 1)

COLOR_MAP = {
    "red": RED,
    "green": GREEN,
    "blue": BLUE,
    "black": BLACK,
    "white": WHITE,
    "brown": BROWN,
    "tan": TAN,
    "grey": GREY,
    "yellow": YELLOW,
    "cyan": CYAN,
}

RGB = namedtuple("RGB", ["red", "green", "blue"])
COLOR_FRANKA = [
    RGB(red=162 / 255, green=192 / 255, blue=222 / 255),
    RGB(red=1, green=1, blue=1),
    RGB(red=1, green=1, blue=1),
    RGB(red=162 / 255, green=192 / 255, blue=222 / 255),
    RGB(red=1, green=1, blue=1),
    RGB(red=1, green=1, blue=1),
    RGB(red=162 / 255, green=192 / 255, blue=222 / 255),
    RGB(red=1, green=1, blue=1),
    RGB(red=162 / 255, green=192 / 255, blue=222 / 255),
    RGB(red=162 / 255, green=192 / 255, blue=222 / 255),
]
