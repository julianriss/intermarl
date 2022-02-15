from cgi import test
from numpy import float32
from ray.rllib.agents.dqn.dqn_torch_policy import DQNTorchPolicy
from ray.rllib.agents.dqn.dqn_torch_policy import ComputeTDErrorMixin
from ray.rllib.agents.dqn.dqn_torch_policy import QLoss

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

    def __init__(self, environment):

        # environment.env.reset()

        #self.os = environment.env.observation_space(environment.env.agents[0])
        self.os = Box(-300., 300., (4,), float32)

        #self.acs = environment.get_env().action_space(environment.get_env().agents[0])
        
        self.acs = Discrete(81)
        self.dqn = DQNTorchPolicy(self.os, self.acs, {"num_gpus": 0,
            "num_workers": 1,
            })
        pass 
    def feedDQN(self, batch):
        #print("observations: ", batch["obs"])
        #print("actions:", batch["actions"])

        #repeator = 1
        #repeated_batch = SampleBatch.concat_samples([batch for k in range(repeator)])
        #random_obs = np.random.rand(repeator,4)
        #random_actions = np.random.randint(low=0, high=80, size=repeator)
        #repeated_batch['obs'] = random_obs
        #repeated_batch["new_obs"] = random_obs
        # repeated_batch["actions"] = random_actions
        #repeated_batch["dones"] = np.array([False for k in range(256)])
        #repeated_batch["rewards"] = np.array([0 for k in range(256)])

        #print(self.dqn.get_weights())
        #print()
        self.dqn.learn_on_batch(batch)
        
        print("saved:",  mpu.io.read(dir + "qvals.pickle"))
        #print(self.dqn.get_tower_stats("q_values"))

         
        #mpu.io.write('learn_on_batch.pickle', self.dqn.learn_on_batch(batch))
        #print(batch)
        pass

