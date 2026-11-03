from enum import Enum


class EpisodeDoneState(Enum):
    NONE = 0
    FAIL = 1
    STEPS_EXCEEDED = 2