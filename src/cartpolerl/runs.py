"""Saving and loading training runs.

A run is a folder containing everything needed to interpret, reload and reproduce it:
    model.npz       agent model (written by the agent itself)
    config.json     full Config used for the run
    spec.json       EnvSpec of the env the agent was trained in
    rng_state.json  state of the env and agent random generators at save time
    meta.json       format version, agent type, creation time, parent run, git commit
A registry.csv in the base folder indexes all runs.
"""
import csv
import json
import subprocess
from dataclasses import asdict, dataclass, fields
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import dacite
import numpy as np

from .actions import Action
from .agents import Agent
from .config import Config
from .env import EnvSpec

FORMAT_VERSION = 1

CONFIG_FILE = "config.json"
SPEC_FILE = "spec.json"
RNG_FILE = "rng_state.json"
META_FILE = "meta.json"
REGISTRY_FILE = "registry.csv"

_DACITE_CONFIG = dacite.Config(type_hooks={Action: lambda name: Action[name], Path: Path}, cast=[tuple])


@dataclass(frozen=True)
class LoadedRun:
    path: Path
    cfg: Config
    spec: EnvSpec
    meta: dict[str, Any]
    rng_states: dict[str, dict[str, Any]]


def _to_jsonable(obj: Any) -> Any:
    """default= hook for json.dump: values json can't serialize by itself."""
    if isinstance(obj, Enum):
        return obj.name
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")


def _write_json(path: Path, data: Any) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_to_jsonable)


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing run file: {path}")
    with open(path) as f:
        return json.load(f)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def create_run_dir(base: Path, agent_name: str) -> Path:
    """Create and return a new, unique run folder: <base>/<date>_<agent_name>_<n>."""
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        run_dir = base / f"{date.today()}_{agent_name}_{n}"
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            n += 1


def save_run(run_dir: Path, cfg: Config, spec: EnvSpec, agent: Agent,
             rngs: dict[str, np.random.Generator], parent_run: str | None = None) -> None:
    run_dir = Path(run_dir)

    agent.save(run_dir)
    _write_json(run_dir / CONFIG_FILE, asdict(cfg))
    _write_json(run_dir / SPEC_FILE, asdict(spec))
    _write_json(run_dir / RNG_FILE, {name: rng.bit_generator.state for name, rng in rngs.items()})

    meta = {
        "format_version": FORMAT_VERSION,
        "agent_type": agent.agent_name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "parent_run": parent_run,
        "git_commit": _git_commit(),
    }
    _write_json(run_dir / META_FILE, meta)

    _append_registry(run_dir, meta, agent.params)


def _append_registry(run_dir: Path, meta: dict[str, Any], params: dict[str, Any]) -> None:
    registry_path = run_dir.parent / REGISTRY_FILE
    fieldnames = ["run_name", "agent_type", "created", "parent_run", "params"]
    write_header = not registry_path.exists()
    with open(registry_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow({
            "run_name": run_dir.name,
            "agent_type": meta["agent_type"],
            "created": meta["created"],
            "parent_run": meta["parent_run"] or "",
            "params": json.dumps(params, default=_to_jsonable),
        })


def config_from_dict(d: dict[str, Any]) -> Config:
    return dacite.from_dict(Config, d, config=_DACITE_CONFIG)


def spec_from_dict(d: dict[str, Any]) -> EnvSpec:
    return dacite.from_dict(EnvSpec, d, config=_DACITE_CONFIG)


def load_run(run_dir: Path) -> LoadedRun:
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run folder not found: {run_dir}")

    meta = _read_json(run_dir / META_FILE)
    version = meta.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(f"Run {run_dir.name} has format version {version}, this code reads version {FORMAT_VERSION}.")

    return LoadedRun(
        path=run_dir,
        cfg=config_from_dict(_read_json(run_dir / CONFIG_FILE)),
        spec=spec_from_dict(_read_json(run_dir / SPEC_FILE)),
        meta=meta,
        rng_states=_read_json(run_dir / RNG_FILE),
    )


def check_spec(saved: EnvSpec, current: EnvSpec) -> None:
    """Raise ValueError describing every difference between the saved and the current env spec."""
    if saved == current:
        return

    diffs = []
    saved_vars = {v.name: v for v in saved.variables}
    current_vars = {v.name: v for v in current.variables}

    if list(saved_vars) != list(current_vars):
        diffs.append(f"variables: saved {list(saved_vars)}, current {list(current_vars)}")
    for name in saved_vars.keys() & current_vars.keys():
        s, c = saved_vars[name], current_vars[name]
        if (s.min, s.max) != (c.min, c.max):
            diffs.append(f"bounds of '{name}': saved ({s.min}, {s.max}), current ({c.min}, {c.max})")

    for f in fields(EnvSpec):
        if f.name == "variables":
            continue
        s, c = getattr(saved, f.name), getattr(current, f.name)
        if s != c:
            diffs.append(f"{f.name}: saved {s}, current {c}")

    raise ValueError("Saved run does not match the current env:\n  " + "\n  ".join(diffs))


def restore_rng(state: dict[str, Any]) -> np.random.Generator:
    bit_generator = getattr(np.random, state["bit_generator"])()
    bit_generator.state = state
    return np.random.Generator(bit_generator)
