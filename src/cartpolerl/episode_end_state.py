from enum import Enum


class EpisodeEndState(Enum):
    NONE = "NONE"
    FAIL = "FAIL"
    STEPS_EXCEEDED = "STEPS_EXCEEDED"