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
    # max_val: float

def center_vert_slow(physics_cfg: PhysicsConfig, task_cfg: TaskConfig, state: State, action: Action, next_state: State) -> float:

    if task_cfg.fail_angle_deg is not None:
        verticality_reward = 1 - abs(next_state.phi) / np.deg2rad(task_cfg.fail_angle_deg)
    else:
        verticality_reward = 1 - abs(next_state.phi) / np.deg2rad(180)

    centrality_reward = (physics_cfg.track_width // 2 - abs(next_state.x)) / (physics_cfg.track_width // 2) 
    omega_reward = np.clip((0.1 - abs(next_state.omega)) * 10, 0, 1)

    return (verticality_reward + centrality_reward + omega_reward) / 3

def center_vert(physics_cfg: PhysicsConfig, task_cfg: TaskConfig, state: State, action: Action, next_state: State) -> float:
    if task_cfg.fail_angle_deg is not None:
        verticality_reward = 1 - abs(next_state.phi) / np.deg2rad(task_cfg.fail_angle_deg)
    else:
        verticality_reward = 1 - abs(next_state.phi) / np.deg2rad(180)

    centrality_reward = (physics_cfg.track_width // 2 - abs(next_state.x)) / (physics_cfg.track_width // 2) 

    return (verticality_reward + centrality_reward) / 2

def survive(physics_cfg: PhysicsConfig, task_cfg: TaskConfig, state: State, action: Action, next_state: State) -> float:
    return 1

# def suicide(physics_cfg: PhysicsConfig, task_cfg: TaskConfig, state: State, action: Action, next_state: State) -> float:
#     return -1


reward_obj_1 = Reward(center_vert_slow)#, 3)
reward_obj_2 = Reward(survive)#, 1)
# reward_obj_3 = Reward(suicide)#, -1)
reward_obj_4 = Reward(center_vert)#, 2)

REWARDS: dict[str, Reward] = {
    "shaped": reward_obj_1,
    "shaped_no_omega": reward_obj_4,
    "survival": reward_obj_2,
    # "suicide": reward_obj_3
}
