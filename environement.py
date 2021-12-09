from ray.tune.registry import register_env
from pettingzoo.butterfly import prison_v3
from ray.rllib.env import PettingZooEnv
import supersuit as ss
from typing import Dict



class Environment(object):
   

    def env_creator(self):
        env = prison_v3.env(num_floors=4)

        env = ss.color_reduction_v0(env, mode='B')
        env = ss.dtype_v0(env, 'float32')
        env = ss.resize_v0(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 3)
        env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
        return env
    
    def registerenv(self):
        register_env("prison", lambda config: PettingZooEnv(self.env_creator()))
        pass