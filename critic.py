from numpy import float32
from ray.rllib.agents.dqn.dqn_torch_policy import DQNTorchPolicy
from ray.rllib.agents.dqn.dqn_torch_policy import compute_q_values
from ray.rllib.policy.sample_batch import SampleBatch
from gym.spaces import Discrete, Box
import torch
import src.utils_folder.array_utils as ar_ut
from typing import Tuple
import numpy as np


dir = "/Users/julian/Desktop/"



class Critic(object):

    def test(self):
        print("test")
        return "z"

    def __init__(self):

        # environment.env.reset()
        self.action_space = (3, 3, 3, 3)  # TODO: make this dynamic from the specific environment!

        #self.os = environment.env.observation_space(environment.env.agents[0])
        self.os = Box(-300., 300., (4,), float32)

        #self.acs = environment.get_env().action_space(environment.get_env().agents[0])
        
        self.acs = Discrete(np.prod(self.action_space))
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


    def get_impact_samples_for_batch(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Computes the impact samples for a given batch.
        Args:
            obs (torch.Tensor): batch of observations (shape=(#samples, #features))
            actions (torch.Tensor): batch of one-hot encoded actions
        Returns:
            torch.Tensor: shape=(#samples, #agents)
        """
        q_values = self.get_q_values(obs)
        decoded_actions = ar_ut.decode_discrete_actions(actions, self.action_space, ret_as_joint_actions=True)
        neighbors = ar_ut.get_neighbors_to_actions(decoded_actions, self.action_space)
        return ar_ut.get_impact_samples_from_q_values(q_values, neighbors)

