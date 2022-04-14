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

    def t(self):
        return "test"

    def env_creator(self):
        # self.env = prison_v3.env(num_floors=2, vector_observation=True)

        self.env = prison_v4.env(
            vector_observation=True,
            continuous=False,
            synchronized_start=True,
            identical_aliens=True,
            max_cycles=150,
            num_floors=2,
            random_aliens=False,
        )

        # self.env = ss.color_reduction_v0(self.env, mode='B')
        # self.env = ss.dtype_v0(self.env, 'float32')
        # self.env = ss.resize_v0(self.env, x_size=84, y_size=84)
        # self.env = ss.normalize_obs_v0(self.env, env_min=0, env_max=1)
        # self.env = ss.frame_stack_v1(self.env, 3)

    pass

    def registerenv(self):
        self.env_creator()

        register_env("prison_v4", lambda config: PettingZooEnv(self.env))
        pass
