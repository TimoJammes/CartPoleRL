import numpy as np

from .agents import Agent
from .state import Observation
from .actions import Action
from .episode_end_state import EpisodeEndState

class AppAgentWrapper:


    def __init__(self, agent_min_decision_interval_s, dt):

        
        self.last_action = Action.NONE
        self.obs_at_last_action: Observation

        self.min_decision_interval_steps = int(np.ceil(agent_min_decision_interval_s / dt - 1e-9))
        self.steps_since_action = self.min_decision_interval_steps
        
        self.rewards_since_action: list[float] = list()

        self.episode_rewards: list[float] = list()
        self.episode_rewards_sum: float = 0

    def act(self, agent: Agent, obs: Observation) -> Action:
        if self.steps_since_action < self.min_decision_interval_steps:
            self.steps_since_action += 1
            return self.last_action

        self.last_action = agent.act(obs)
        self.steps_since_action = 1
        self.obs_at_last_action = obs

        return self.last_action


    def update(self, agent: Agent, obs: Observation, action: Action,
                reward: float, obs_prime: Observation, end_state: EpisodeEndState) -> None:

        self.rewards_since_action.append(reward)
        self.episode_rewards.append(reward)
        self.episode_rewards_sum += reward

        
        if end_state == EpisodeEndState.NONE and self.steps_since_action < self.min_decision_interval_steps:
            return


        obs = self.obs_at_last_action

        

        agent.update(obs, action, self.rewards_since_action, obs_prime, end_state)

        self.rewards_since_action.clear()

    def episode_reset(self):
        self.steps_since_action = self.min_decision_interval_steps
        self.rewards_since_action.clear()
        self.episode_rewards.clear()
        self.episode_rewards_sum = 0