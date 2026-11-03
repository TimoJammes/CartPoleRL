from enum import Enum

class Action(Enum):
    LEFT = -1
    NONE = 0
    RIGHT = 1

    # @property
    # def direction(self) -> int:
    #     """-1, 0 or +1, for the physics."""
    #     return self.value - 1