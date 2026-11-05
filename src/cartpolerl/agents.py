import numpy as np
from dataclasses import asdict
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any

from .config import AgentConfig, BinEdgeConfig
from .env import EnvSpec
from .state import Observation
from .actions import Action
from .episode_end_state import EpisodeEndState

class Agent(ABC):
    agent_name: str

    @abstractmethod
    def act(self, obs: Observation) -> Action:
        ...

    @abstractmethod
    def update(self, obs: Observation, action: Action, rewards: list[float], obs_prime: Observation, done: EpisodeEndState):
        ...

    @abstractmethod
    def save(self, run_dir: Path) -> None:
        """Write this agent's model into run_dir (the run folder itself is managed by runs.py)."""
        ...

    @property
    @abstractmethod
    def params(self) -> dict[str, Any]:
        ...

    # @abstractmethod
    # def episode_reset(self) -> None:
    #     ...

def _compute_edges(cfg: AgentConfig, env_spec: EnvSpec) -> dict[str, np.ndarray]:
    edges = dict()

    for var in cfg.bin_config.edges:
        assert env_spec.has_var(var.name)

    for var in env_spec.variables:
        if cfg.bin_config.has(var.name):

            cfg_var: BinEdgeConfig = cfg.bin_config.get(var.name)

            if cfg_var.n_bins != -1:
                assert cfg_var.n_bins > 0, "when generating bins from edges, n_bins must be strictly greater than 0."
                assert len(cfg_var.bin_edges) == 2, "when generating bins from edges, bin_edges must have 2 elements (start and end of edges)"
                if cfg_var.bin_edges_include_endpoints:
                    edges[var.name] = np.linspace(cfg_var.bin_edges[0], cfg_var.bin_edges[1], cfg_var.n_bins+1)[1:-1]
                else:
                    assert cfg_var.n_bins >= 3, "when generating bins from edges without including endpoints, n_bins must be >= 3 to ensure adequate binning."
                    edges[var.name] = np.linspace(cfg_var.bin_edges[0], cfg_var.bin_edges[1], cfg_var.n_bins-1)
            else:

                assert np.all(np.diff(cfg_var.bin_edges) > 0)  #check explicit binning is strictly increasing for proper binning
                edges[var.name] = np.array(cfg_var.bin_edges)

        elif var.max == np.inf or var.min == -np.inf:

            raise ValueError("An unbounded variable must have an agent config (or implement initial value learning by agent during training)")
        else:
            if cfg.default_n_bins < 2:
                raise ValueError("cfg.default_n_bins must be at least 2 for adequate binning.")
            edges[var.name] = np.linspace(var.min, var.max, cfg.default_n_bins+1)[1:-1]

    return edges


class QLearningTableAgent(Agent):
    agent_name = "qlearningtable"
    model_file = "model.npz"

    def __init__(self, cfg: AgentConfig, dt: float,
                  actions: tuple[Action, ...], edges: dict[str, np.ndarray], Q: np.ndarray, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng

        self.actions = actions
        self.action_to_index = {
            a: i for i, a in enumerate(self.actions)
        }

        self.edges = edges

        expected_shape = tuple(len(e)+1 for e in self.edges.values()) + (len(self.actions),)
        if Q.shape != expected_shape:
            raise ValueError(f"Q-table shape {Q.shape} does not match bins/actions (expected {expected_shape}).")
        self.Q = Q

        # self.last_action = Action.NONE
        # self.obs_at_last_action: Observation

        # self.min_decision_interval_steps = int(np.ceil(self.cfg.min_decision_interval_s / dt - 1e-9))
        # self.steps_since_action = self.min_decision_interval_steps
        
        # self.rewards_since_action = list()

        

    @classmethod
    def from_spec(cls, cfg: AgentConfig, env_spec: EnvSpec, rng: np.random.Generator) -> "QLearningTableAgent":
        """New agent: bins from the config + env spec, optimistic Q-table."""
        edges = _compute_edges(cfg, env_spec)
        Q = np.full([len(e)+1 for e in edges.values()]+[len(env_spec.actions)], 1 / (1 - cfg.gamma))
        return cls(cfg, env_spec.dt, env_spec.actions, edges, Q, rng)

    @classmethod
    def load(cls, run_dir: Path, cfg: AgentConfig, env_spec: EnvSpec, rng: np.random.Generator) -> "QLearningTableAgent":
        """Agent from a saved run: Q-table and edges come from the file, checked against the env spec."""
        path = Path(run_dir) / cls.model_file
        if not path.exists():
            raise FileNotFoundError(f"No model file at {path}.")

        with np.load(path, allow_pickle=False) as data:
            var_names = [str(n) for n in data["var_names"]]
            action_names = [str(n) for n in data["action_names"]]
            Q = data["Q"]
            edges = {name: data[f"edges__{name}"] for name in var_names}

        spec_var_names = [v.name for v in env_spec.variables]
        if var_names != spec_var_names:
            raise ValueError(f"Saved model variables {var_names} do not match env variables {spec_var_names}.")

        actions = tuple(Action[n] for n in action_names)
        if actions != env_spec.actions:
            raise ValueError(f"Saved model actions {action_names} do not match env actions {[a.name for a in env_spec.actions]}.")

        return cls(cfg, env_spec.dt, actions, edges, Q, rng)

    def save(self, run_dir: Path) -> None:
        arrays = {f"edges__{name}": e for name, e in self.edges.items()}
        np.savez(
            Path(run_dir) / self.model_file,
            Q=self.Q,
            var_names=np.array(list(self.edges.keys())),
            action_names=np.array([a.name for a in self.actions]),
            **arrays, #type: ignore
        )

    def act(self, obs: Observation) -> Action:

        # if self.steps_since_action < self.min_decision_interval_steps:
        #     self.steps_since_action += 1
        #     return self.last_action
        
        obs_discretized = self._discretize(obs)

        if self.rng.random() < self.cfg.epsilon:
            action_num = self.rng.integers(len(self.actions))
        else:
            #this instead of argmax to break ties randomly
            q = self.Q[obs_discretized]
            action_num = self.rng.choice(np.flatnonzero(q == q.max()))


        # self.last_action = self.actions[action_num]
        # self.steps_since_action = 1

        # self.obs_at_last_action = obs

        # return self.last_action

        return self.actions[action_num]
    
    def update(self, obs: Observation, action: Action, rewards: list[float], obs_prime: Observation, done: EpisodeEndState):

        target = sum(r * self.cfg.gamma**k for k, r in enumerate(rewards))

        # target = 0
        # for k, saved_reward in enumerate(self.rewards_since_action):
        #     target += saved_reward * self.cfg.gamma ** k
        if done != EpisodeEndState.FAIL: #bootstrap
            target += self.cfg.gamma ** len(rewards) * self._Qstar(obs_prime)

        idx = self._discretize(obs) + (self.action_to_index[action],)
        self.Q[idx] += self.cfg.alpha * (target - self.Q[idx])

        # self.rewards_since_action.clear()

    def _discretize(self, obs: Observation) -> tuple[int, ...]:
        return tuple(np.digitize(getattr(obs, var_name), self.edges[var_name]) for var_name in self.edges.keys())

    def _Qstar(self, obs: Observation):
        return np.max(self.Q[self._discretize(obs)])

    # def episode_reset(self):
    #     self.steps_since_action = self.min_decision_interval_steps
    #     self.rewards_since_action.clear()

    @property
    def params(self):
        return {
            "alpha": self.cfg.alpha,
            "gamma": self.cfg.gamma,
            "epsilon": self.cfg.epsilon,
            "binning": asdict(self.cfg.bin_config)
        }

    


AGENT_TYPES: dict[str, type[Agent]] = {
    QLearningTableAgent.agent_name: QLearningTableAgent,
}
