from typing import Dict

import supersuit as ss
from gym.spaces import Dict, MultiDiscrete, Tuple
from pettingzoo.butterfly import pistonball_v5, prison_v4
from ray.rllib.env import PettingZooEnv
from ray.rllib.env.multi_agent_env import ENV_STATE
from ray.tune.registry import register_env


class Environment(object):
    def __init__(self):
        self.env = None

    def get_env(self):
        return self.env

    def env_creator(self):
        self.env = prison_v4.env(
            vector_observation=True,
            continuous=False,
            synchronized_start=True,
            identical_aliens=True,
            max_cycles=150,
            num_floors=2,
            random_aliens=False,
        )

    def registerenv(self):
        self.env_creator()

        register_env("prison_v4", lambda config: PettingZooEnv(self.env))
