import numpy as np
from dataclasses import asdict
from pathlib import Path
from datetime import date
from abc import ABC, abstractmethod
import csv
import json
from typing import Any

from .config import AgentConfig, BinEdgeConfig
from .env import EnvSpec
from .state import Observation
from .actions import Action


class Agent(ABC):
    agent_name: str
    cfg: AgentConfig

    @abstractmethod
    def act(self, obs: Observation):
        ...

    @abstractmethod
    def update(self, obs: Observation, action: Action, reward: float, obs_prime: Observation, done: bool):
        ...

    @property
    @abstractmethod
    def model(self) -> np.ndarray:
        ...

    @property
    def save_model(self):
        return self.cfg.save
    
    def save(self):
        potential_path_base = Path(__file__).resolve().parent.parent.parent / self.cfg.save_folder / f"{date.today()}_{self.agent_name}.npy"

        if potential_path_base.exists():
            ver = 2
            potential_path = potential_path_base.with_stem(potential_path_base.stem + f"_{ver}")

            while potential_path.exists():
                ver += 1
                potential_path = potential_path.with_stem(potential_path_base.stem + f"_{ver}")
        else:
            potential_path = potential_path_base

        np.save(potential_path, self.model)

        self.log_run(potential_path.parent, potential_path.stem)

    def log_run(self, save_folder: Path, run_name: str):
        registry_path = save_folder / "registry.csv"
        write_header = not registry_path.exists()
        with open(registry_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["training_run_name", "params"])
            if write_header:
                w.writeheader()
            w.writerow({"training_run_name": run_name, "params": json.dumps(self.params)})

    @property
    @abstractmethod
    def params(self) -> dict[str, Any]:
        ...

class QLearningTableAgent(Agent):

    def __init__(self, cfg: AgentConfig, env_spec: EnvSpec, rng: np.random.Generator):

        self.agent_name = "qlearningtable"
        self.cfg = cfg
        self.rng = rng

        self.actions = env_spec.actions

        self.action_to_index = {
            a: i for i, a in enumerate(self.actions)
        }

        self.edges = dict()

        for var in cfg.bin_config.edges:
            assert env_spec.has_var(var.name)

        for var in env_spec.variables:
            if cfg.bin_config.has(var.name):

                cfg_var: BinEdgeConfig = cfg.bin_config.get(var.name)

                if cfg_var.n_bins != -1:
                    assert cfg_var.n_bins > 0, "when generating bins from edges, n_bins must be strictly greater than 0."
                    assert len(cfg_var.bin_edges) == 2, "when generating bins from edges, bin_edges must have 2 elements (start and end of edges)"
                    if cfg_var.bin_edges_include_endpoints:
                        self.edges[var.name] = np.linspace(cfg_var.bin_edges[0], cfg_var.bin_edges[1], cfg_var.n_bins+1)[1:-1]
                    else:
                        assert cfg_var.n_bins >= 3, "when generating bins from edges without including endpoints, n_bins must be >= 3 to ensure adequate binning."
                        self.edges[var.name] = np.linspace(cfg_var.bin_edges[0], cfg_var.bin_edges[1], cfg_var.n_bins-1)
                else:
                    
                    assert np.all(np.diff(cfg_var.bin_edges) > 0)  #check explicit binning is strictly increasing for proper binning
                    self.edges[var.name] = np.array(cfg_var.bin_edges)

            elif var.max == np.inf or var.min == -np.inf:

                raise ValueError("An unbounded variable must have an agent config (or implement initial value learning by agent during training)")
            else:
                if cfg.default_n_bins < 2:
                    raise ValueError("cfg.default_n_bins must be at least 2 for adequate binning.")
                self.edges[var.name] = np.linspace(var.min, var.max, cfg.default_n_bins+1)[1:-1]

        self.Q = np.full([len(edges)+1 for edges in self.edges.values()]+[len(self.actions)], env_spec.max_reward / (1 - cfg.gamma))

    def act(self, obs: Observation) -> Action:

        obs_discretized = self._discretize(obs)

        if self.rng.random() < self.cfg.epsilon:
            action_num = self.rng.integers(len(self.actions))
        else:
            #this instead of argmax to break ties randomly
            q = self.Q[obs_discretized]
            action_num = self.rng.choice(np.flatnonzero(q == q.max()))

        return self.actions[action_num]

    def update(self, obs: Observation, action: Action, reward: float, obs_prime: Observation, done: bool):
        obs_discretized = self._discretize(obs)

        target = reward if done else reward + self.cfg.gamma * self._Qstar(obs_prime)
        action_index = self.action_to_index[action]
        idx = obs_discretized + (action_index,)
        self.Q[idx] += self.cfg.alpha * (target - self.Q[idx])

    def _discretize(self, obs: Observation) -> tuple[int, ...]:
        return tuple(np.digitize(getattr(obs, var_name), self.edges[var_name]) for var_name in self.edges.keys())
    
    def _Qstar(self, obs: Observation):
        return np.max(self.Q[self._discretize(obs)])

    @property
    def model(self):
        return self.Q

    @property
    def params(self):
        return {
            "alpha": self.cfg.alpha,
            "gamma": self.cfg.gamma,
            "epsilon": self.cfg.epsilon,
            "binning": asdict(self.cfg.bin_config)
        }

            
            