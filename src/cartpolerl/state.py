from dataclasses import dataclass
import numpy as np

@dataclass
class State:
    x: float
    vel: float
    theta: float
    omega: float

    @property
    def phi(self):
        return self.theta % (2*np.pi) - np.pi

@dataclass(frozen=True)
class Observation:
    pass

@dataclass(frozen=True)
class FullObservation(Observation):
    x: float
    vel: float
    phi: float
    omega: float

@dataclass(frozen=True)
class PartialObservation(Observation):
    pass

OBSERVATIONS: dict[str, type[Observation]] = {
    "full": FullObservation,
}