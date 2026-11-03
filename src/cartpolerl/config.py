from dataclasses import dataclass, field
import numpy as np

from .actions import Action


@dataclass(frozen=True)
class DisplayConfig:
    # fps: int = 30
    window_width: int = 600
    window_height: int = 600

    background_color: tuple[int, int, int] = (255, 255, 255)
    pole_color: tuple[int, int, int] = (255, 0, 0)
    cart_color: tuple[int, int, int] = (0, 0, 0)

    cart_width: int = 100
    cart_height: int = 50
    pole_width: int = 10

    track_y: int = 300

@dataclass(frozen=True)
class PhysicsConfig:
    g: float = 3.5

    dt: float = 1/30

    # action_magnitude: tuple[tuple[Action, float], ...] = (
    #     (Action.LEFT, -1),
    #     (Action.NONE, 0),
    #     (Action.RIGHT, 1)
    # )

    pole_length: int = 200 # px
    pole_friction_coeff: float = 0.005

    track_width: int = 600 # px

    # cart_width: int = 100 # px
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

    

@dataclass(frozen=True)
class TaskConfig:
    actions: tuple[Action, ...] = (Action.LEFT, Action.NONE, Action.RIGHT)
    reward: str = "shaped"          # key in rewards.REWARDS
    observation: str = "full"       # key in state.OBSERVATIONS
    fail_angle_deg: float | None = 20      # None = never terminate (free play)
    # max_steps: int | None = 1000           # cap on episode length
    max_time: float | None = 5 # seconds
    num_episodes: int = 1000
    start_angle_jitter_deg: float = 1.0    # reset() samples upright ± this

@dataclass(frozen=True)
class Config:
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    seed: int = 0