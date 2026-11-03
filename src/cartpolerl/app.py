import numpy as np
import argparse

from .config import Config
from .env import CartPoleEnv
from .rewards import reward_obj
from .state import FullObservation
from .agents import QLearningTableAgent
from .pygame_ui import PygameUI, InputCommand
from .human_controller import HumanController

def main():
    args = parse_args()

    cfg = Config()
    env_rng, agent_rng = np.random.default_rng(cfg.seed).spawn(2)

    env = CartPoleEnv(cfg.physics, cfg.task, env_rng, reward_obj, FullObservation)
    
    if args.human:
        agent = HumanController(cfg.task.actions)
    else:
        agent = QLearningTableAgent(cfg.agent, env.spec, agent_rng)
    
    ui = None if args.headless else PygameUI(cfg.display, cfg.physics)


    obs = env.reset()
    mainloop=True

    try:
        while mainloop:
            if ui is not None:
                ui.tick()
                
                event_commands: set[InputCommand] = ui.process_events()
                if InputCommand.QUIT in event_commands:
                    break

            action = agent.act(obs)
            obs_prime, reward, done = env.step(action)
            agent.update(obs, action, reward, obs_prime, done)

            obs = env.reset() if done else obs_prime
            
            if ui is not None:
                ui.render(env)

    except KeyboardInterrupt:
        pass
    finally:
        if ui is not None:
            ui.quit()

        if agent.save_model:
            agent.save()

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Q-learning agent")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--human", action="store_true")
    args = parser.parse_args()

    if args.headless and args.human:
        parser.error("--headless and --human cannot be used together (no GUI).")

    return args

if __name__ == "__main__":
    main()