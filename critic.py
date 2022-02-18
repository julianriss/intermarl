from ast import Mult
from cgi import test
from numpy import float32
from ray.rllib.agents.dqn.dqn_torch_policy import DQNTorchPolicy
from ray.rllib.agents.dqn.dqn_torch_policy import ComputeTDErrorMixin
from ray.rllib.agents.dqn.dqn_torch_policy import QLoss
from ray.rllib.agents.dqn.dqn_torch_policy import compute_q_values
from ray.rllib.policy.sample_batch import SampleBatch
import mpu
from gym.spaces import MultiDiscrete, Discrete, Box
import numpy as np
import ray
from types import MethodType

dir = "/Users/julian/Desktop/"



class Critic(object):

    def test(self):
        print("test")
        return "z"

    def __init__(self):

        # environment.env.reset()

        #self.os = environment.env.observation_space(environment.env.agents[0])
        self.os = Box(-300., 300., (4,), float32)

        #self.acs = environment.get_env().action_space(environment.get_env().agents[0])
        
        self.acs = Discrete(81)
        self.dqn = DQNTorchPolicy(self.os, self.acs, {"num_gpus": 0,
            "num_workers": 1,
            })
        pass 

    def get_q_values(self, obs):
        q_tp1, _, _, _ = compute_q_values(
        self.dqn,
        self.dqn.target_models[self.dqn.model],
        {"obs": obs},
        explore=False,
        is_training=False)
        return q_tp1

    def feedDQN(self, batch, agentnumber):
        print(batch)
        self.dqn.learn_on_batch(batch)

       
        print("agentnumber: ", agentnumber)
        print("observations:", batch[SampleBatch.CUR_OBS])
        print("observations_new:", batch[SampleBatch.NEXT_OBS])
        print("actions: ", batch[SampleBatch.ACTIONS] )
        print("reward:", batch[SampleBatch.REWARDS])
        print("Q-Values:",  self.get_q_values(batch[SampleBatch.NEXT_OBS]))
        print("\n")
        pass

