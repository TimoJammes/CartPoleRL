import vars as V
from dataclasses import dataclass

@dataclass
class Cart:
    x: int
    y: int
    velocity: float
    acc_norm: float
    width: int
    height: int
    color: tuple[3]
    curr_acc: float = 0

@dataclass
class Pole:
    x: int
    y: int
    theta: float
    omega: float
    # omegadot: float
    width: int
    length: int
    color: tuple[3]

