from numpy import float32
from ray.rllib.agents.dqn.dqn_torch_policy import DQNTorchPolicy
from ray.rllib.agents.dqn.dqn_torch_policy import ComputeTDErrorMixin
import mpu
from gym.spaces import MultiDiscrete, Discrete, Box





class Critic(object):

    def test(self):
        return "z"

    def __init__(self, environment):

        environment.env.reset()

        #self.os = environment.env.observation_space(environment.env.agents[0])
        self.os = Box(-300., 300., (1,), float32)

        #self.acs = environment.get_env().action_space(environment.get_env().agents[0])
        
        self.acs = Discrete(81)
        self.dqn = DQNTorchPolicy(self.os, self.acs, {"num_gpus": 0,
            "num_workers": 1})
        pass 
    def feedDQN(self, batch):
        print("observations: ", batch["obs"])
        print("actions:", batch["actions"])

        #print(self.dqn.get_weights())
        print(self.dqn.learn_on_batch(batch))
        #mpu.io.write('learn_on_batch.pickle', self.dqn.learn_on_batch(batch))
        #print(batch)
        pass

