from dataclasses import dataclass, field
import numpy as np

from .actions import Action


@dataclass(frozen=True)
class DisplayConfig:
    # fps: int = 30
    window_width: int = 600
    window_height: int = 600

    background_color: tuple[int, int, int] = (255, 255, 255)
    track_color: tuple[int, int, int] = (123, 63, 0) #chocolate
    pole_color: tuple[int, int, int] = (255, 0, 0)
    cart_body_color: tuple[int, int, int] = (0, 0, 0)
    cart_wheel_color: tuple[int, int, int] = (169, 169, 169)

    cart_width: int = 50
    cart_height: int = 25
    cart_wheel_radius: int = 12
    pole_width: int = 5

    track_y: int = 300
    track_thickness: int = 5

@dataclass(frozen=True)
class PhysicsConfig:
    g: float = 3.5

    dt: float = 1/30 #TODO edit physics update to take dt into account

    # action_magnitude: tuple[tuple[Action, float], ...] = (
    #     (Action.LEFT, -1),
    #     (Action.NONE, 0),
    #     (Action.RIGHT, 1)
    # )

    pole_length: int = 200
    pole_friction_coeff: float = 0.005

    track_length: int = 1200

    cart_acceleration_norm: float = 3
    cart_friction_coeff: float = 0.05

    # def get_action_magnitude(self, action: Action) -> float:
    #     for a, m in self.action_magnitude:
    #         if a == action:
    #             return m

    #     raise ValueError(f"Action {action} is not in PhysicsConfig.")

@dataclass(frozen=True)
class BinEdgeConfig:
    name: str
    bin_edges: tuple[float, ...]
    bin_edges_include_endpoints: bool
    n_bins: int

@dataclass(frozen=True)
class BinningConfig:
    edges: tuple[BinEdgeConfig, ...] = (
            # BinEdgeConfig("x",     (100, 500),                     True,   6),
            BinEdgeConfig("vel",   (-36, 36),                      False,  7),
            BinEdgeConfig("phi",   tuple(np.deg2rad((-10, -4, -2, -1, 1, 2, 4, 10))), False, -1),
            BinEdgeConfig("omega", (-0.2, 0.2),                    False,  7),
        )

    def get(self, name: str) -> BinEdgeConfig:
        for e in self.edges:
            if e.name == name:
                return e
        raise KeyError(name)

    def has(self, name: str) -> bool:
        for e in self.edges:
            if e.name == name:
                return True
        return False
    
@dataclass(frozen=True)
class AgentConfig:
    epsilon: float = 0.1 # for greedy exploration
    alpha: float = 0.1 # LR
    gamma: float = 0.99

    bin_config: BinningConfig = field(default_factory=BinningConfig)
    default_n_bins: int = 7
    min_decision_interval_s: float = 0. #seconds, simulation clock (rounds up to nearest #steps) 
    

@dataclass(frozen=True)
class TaskConfig:
    actions: tuple[Action, ...] = (Action.LEFT, Action.NONE, Action.RIGHT)
    reward: str = "shaped_no_omega"          # key in rewards.REWARDS
    observation: str = "full"       # key in state.OBSERVATIONS
    fail_angle_deg: float | None = 30      # None = never terminate (free play)
    # max_steps: int | None = 1000           # cap on episode length
    max_time: float | None = 5 # seconds
    num_episodes: int = 10000
    start_angle_2sigma: float = 1    # reset() samples upright from this as a 2sigma normal distribution
    start_mean_theta_deg: float = 180 #180

@dataclass(frozen=True)
class Config:
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    seed: int = 0