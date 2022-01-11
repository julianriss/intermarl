from ray.tune.registry import register_env
from pettingzoo.butterfly import prison_v3
from ray.rllib.env import PettingZooEnv
import supersuit as ss
from typing import Dict



class Environment(object):
    def __init__(self):
        self.env = None

    def env(self):
        return self.env

    def env_creator(self):
        self.env = prison_v3.env(num_floors=2, vector_observation=True)
       
        #self.env = ss.color_reduction_v0(self.env, mode='B')
        #self.env = ss.dtype_v0(self.env, 'float32')
        #self.env = ss.resize_v0(self.env, x_size=84, y_size=84)
        #self.env = ss.frame_stack_v1(self.env, 3)
        #self.env = ss.normalize_obs_v0(self.env, env_min=0, env_max=1)
        pass
    
    def registerenv(self):
        self.env_creator()
        register_env("prison", lambda config: PettingZooEnv(self.env))
        pass