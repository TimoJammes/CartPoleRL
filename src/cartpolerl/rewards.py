from typing import Callable
import numpy as np
from dataclasses import dataclass

from .state import State
from .actions import Action
from .config import TaskConfig, PhysicsConfig

RewardFn = Callable[[PhysicsConfig, TaskConfig, State, Action, State], float]

@dataclass
class Reward:
    fn: RewardFn
    max_val: float

def get_reward(physics_cfg: PhysicsConfig, task_cfg: TaskConfig, state: State, action: Action, next_state: State) -> float:

    if task_cfg.fail_angle_deg is not None:
        verticality_reward = 1 - abs(next_state.phi) / np.deg2rad(task_cfg.fail_angle_deg)
    else:
        verticality_reward = 1 - abs(next_state.phi) / np.deg2rad(180)

    centrality_reward = (physics_cfg.track_width // 2 - abs(next_state.x)) / (physics_cfg.track_width // 2) 
    omega_reward = np.clip((0.1 - abs(next_state.omega)) * 10, 0, 1)

    return verticality_reward + centrality_reward + omega_reward

reward_obj = Reward(get_reward, 3)

REWARDS: dict[str, Reward] = {
    "shaped": reward_obj,
}
