import numpy as np
from dataclasses import dataclass, fields

from .config import TaskConfig, PhysicsConfig
from .actions import Action
from .rewards import Reward
from .state import State, Observation, FullObservation

@dataclass(frozen=True)
class VarInfo:
    name: str
    min: float
    max: float

@dataclass(frozen=True)
class EnvSpec:
    variables: tuple[VarInfo, ...]
    actions: tuple[Action, ...]
    max_reward: float

    def has_var(self, name: str) -> bool:
        for var in self.variables:
            if var.name == name:
                return True
        return False

class CartPoleEnv:

    def __init__(self, physics_cfg: PhysicsConfig, task_cfg: TaskConfig, rng: np.random.Generator, reward: Reward, observation_class: type[Observation]):
        self.physics_cfg = physics_cfg
        self.task_cfg = task_cfg
        self.rng = rng
        self.reward = reward

        self.observation_class = observation_class
        self._obs_names = [f.name for f in fields(observation_class)]

        # defines all observable variables and their exact bounds
        self.var_to_bounds = {
            "x": (-self.physics_cfg.track_width/2, self.physics_cfg.track_width/2),
            "vel": (-np.inf, np.inf),
            "phi": (-np.deg2rad(self.task_cfg.fail_angle_deg) if self.task_cfg.fail_angle_deg is not None else -np.inf, np.deg2rad(self.task_cfg.fail_angle_deg) if self.task_cfg.fail_angle_deg is not None else np.inf), #exceeds bounds at last step of episode (if the pole falls)
            "omega": (-np.inf, np.inf)
        }

        for f_name in self._obs_names:
            assert f_name in self.var_to_bounds.keys(), f"variable {f_name} missing from var_to_bounds table in env constructor."

        self.spec = EnvSpec(tuple(VarInfo(f_name, *self.var_to_bounds[f_name]) for f_name in self._obs_names), task_cfg.actions, reward.max_val)


        self.reset() # sets self.state
    
    
    def step(self, action: Action) -> tuple[Observation, float, bool]: #state, reward, done

        next_state = State(self.state.x, self.state.vel, self.state.theta, self.state.omega)

        cart_acceleration = action.value * self.physics_cfg.cart_acceleration_norm - self.physics_cfg.cart_friction_coeff * next_state.vel
        
        next_state.vel += cart_acceleration
    
        next_state.x += next_state.vel
    
        if abs(next_state.x) > self.physics_cfg.track_width / 2:
            next_state.x = self.physics_cfg.track_width / 2 * np.sign(next_state.x)
            next_state.vel = 0
            cart_acceleration = 0
        
        omega_dot = - self.physics_cfg.g / self.physics_cfg.pole_length * np.sin(next_state.theta) + cart_acceleration / self.physics_cfg.pole_length * np.cos(next_state.theta) - self.physics_cfg.pole_friction_coeff * next_state.omega
        next_state.omega += omega_dot
    
        next_state.theta += next_state.omega
    
        next_state.theta = self._wrap(next_state.theta)

        # if abs(pole.theta) < 1e-3 and abs(pole.omega) < 

        reward = self.reward.fn(self.physics_cfg, self.task_cfg, self.state, action, next_state)
        self.state = next_state
        return self.get_obs(self.state), reward, self.is_done(self.state)

    def is_done(self, state: State):
        if self.task_cfg.fail_angle_deg is None:
            return False
        
        return np.rad2deg(abs(state.phi)) > self.task_cfg.fail_angle_deg

    def get_obs(self, state: State | None = None) -> Observation:
        if state is None:
            state = self.state
        return self.observation_class(**{name: getattr(state, name) for name in self._obs_names})
    
    def _wrap(self, theta: float):
        return (theta + np.pi) % (2 * np.pi) - np.pi

    def reset(self):
        self.state = State(0, 0, self._wrap(np.deg2rad(self.rng.uniform(180-self.task_cfg.start_angle_jitter_deg, 180+self.task_cfg.start_angle_jitter_deg))), 0)

        return self.get_obs()

    