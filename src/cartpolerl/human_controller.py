import pygame

from .state import Observation
from .actions import Action
from .agents import Agent
from .done import EpisodeDoneState

class HumanController(Agent):
    agent_name = "human"

    def __init__(self, env_actions: tuple[Action, ...]):

        self.defined_actions = (Action.LEFT, Action.RIGHT, Action.NONE)

        assert set(env_actions) == set(self.defined_actions), "UI does not define the same human_controller actions as the env accepts."

    @property
    def params(self):
        return {}
    
    def act(self, obs: Observation) -> Action:
        return self.get_action()


    def get_action(self) -> Action:
        keys = pygame.key.get_pressed()

        if keys[pygame.K_RIGHT] and keys[pygame.K_LEFT]:
            return Action.NONE
        
        if keys[pygame.K_RIGHT]:
            return Action.RIGHT
        if keys[pygame.K_LEFT]:
            return Action.LEFT

        return Action.NONE

    def update(self, obs: Observation, action: Action, reward: float, obs_prime: Observation, done: EpisodeDoneState):
        pass
    def save(self, run_dir):
        pass

    def episode_reset(self):
        pass