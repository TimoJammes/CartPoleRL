import numpy as np
import argparse
from dataclasses import replace
from pathlib import Path

from .config import Config, DisplayConfig
from .env import CartPoleEnv
from .rewards import REWARDS
from .state import OBSERVATIONS
from .agents import Agent, QLearningTableAgent
from .pygame_ui import PygameUI, InputCommand
from .human_controller import HumanController
from . import runs

def main():
    args = parse_args()

    loaded = runs.load_run(args.load) if args.load else None

    if loaded is None:
        cfg = Config()
        env_rng, agent_rng = np.random.default_rng(cfg.seed).spawn(2)
    else:
        if loaded.meta["agent_type"] != QLearningTableAgent.agent_name:
            raise SystemExit(f"Run {loaded.path.name} was saved by agent type '{loaded.meta['agent_type']}', expected '{QLearningTableAgent.agent_name}'.")
        # the problem definition comes from the saved run, session settings from now
        cfg = replace(loaded.cfg, display=DisplayConfig())
        env_rng = runs.restore_rng(loaded.rng_states["env"])
        agent_rng = runs.restore_rng(loaded.rng_states["agent"])

    if args.watch:
        cfg = replace(cfg, agent=replace(cfg.agent, epsilon=0.0))
    elif args.epsilon is not None:
        cfg = replace(cfg, agent=replace(cfg.agent, epsilon=args.epsilon))

    env = CartPoleEnv(cfg.physics, cfg.task, env_rng, REWARDS[cfg.task.reward], OBSERVATIONS[cfg.task.observation])

    agent: Agent
    if args.human:
        agent = HumanController(cfg.task.actions)
    elif loaded is not None:
        runs.check_spec(loaded.spec, env.spec)
        agent = QLearningTableAgent.load(loaded.path, cfg.agent, env.spec, agent_rng)
    else:
        agent = QLearningTableAgent.from_spec(cfg.agent, env.spec, agent_rng)

    learning = not (args.watch or args.human)

    ui = None if args.headless else PygameUI(cfg.display, cfg.physics)


    obs = env.reset()

    try:
        while True:
            if ui is not None:
                ui.tick()

                event_commands: set[InputCommand] = ui.process_events()
                if InputCommand.QUIT in event_commands:
                    break

            action = agent.act(obs)
            obs_prime, reward, done = env.step(action)
            if learning:
                agent.update(obs, action, reward, obs_prime, done)

            obs = env.reset() if done else obs_prime

            if ui is not None:
                ui.render(env)

    except KeyboardInterrupt:
        pass
    finally:
        if ui is not None:
            ui.quit()

        if args.save:
            run_dir = runs.create_run_dir(args.run_dir, agent.agent_name)
            runs.save_run(run_dir, cfg, env.spec, agent, {"env": env_rng, "agent": agent_rng},
                          parent_run=loaded.path.name if loaded is not None else None)
            print(f"Saved run to {run_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Q-learning agent")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--human", action="store_true")
    parser.add_argument("--save", action="store_true", help="save the run (model, config, spec, rng state) when it ends")
    parser.add_argument("--run-dir", type=Path, default=Path("training_runs"), help="folder for saved runs, relative to the current directory")
    parser.add_argument("--load", type=Path, metavar="RUN_DIR", help="continue from (or watch) a saved run")
    parser.add_argument("--watch", action="store_true", help="with --load: show the greedy policy (epsilon=0, no learning, no saving)")
    parser.add_argument("--epsilon", type=float, help="override the agent's exploration rate")
    args = parser.parse_args()

    if args.headless and args.human:
        parser.error("--headless and --human cannot be used together (no GUI).")
    if args.watch and args.load is None:
        parser.error("--watch requires --load.")
    if args.watch and (args.headless or args.save):
        parser.error("--watch cannot be combined with --headless or --save.")
    if args.load is not None and args.human:
        parser.error("--load cannot be combined with --human.")
    if args.human and args.save:
        parser.error("--human has no model to save.")

    return args

if __name__ == "__main__":
    main()
