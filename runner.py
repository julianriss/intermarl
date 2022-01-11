import functools
from logging import config, critical
from os import ctermid
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
import hydra
from typing import Dict, Iterable
from ray.tune.trial import Trial
from ray.tune.ray_trial_executor import RayTrialExecutor
from typing import List

from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.tune import Callback
import numpy as np

from ray.rllib import TorchPolicy
from critic import Critic
from functools import partial

import time


class MyCallback(DefaultCallbacks):
  
  

    def __init__(self, environment, legacy_callbacks_dict: Dict[str, callable] = None):
        self.critic = Critic(environment)
        super().__init__(legacy_callbacks_dict=legacy_callbacks_dict)


    def on_learn_on_batch(self, *, policy, train_batch, result, **kwargs):
        #print("observation: ", train_batch["obs"], "actions: ", train_batch["actions"], "rewards: ", train_batch["rewards"], "agentindex: ", train_batch["agent_index"])
        agentindex = train_batch["agent_index"]
        observations = train_batch["obs"]
        actions = train_batch["actions"]
        rewards = train_batch["rewards"]
        batchsize = len(agentindex)
        number_of_agents = len(set(agentindex))
        print(number_of_agents)
        #observations = np.split(observations, number_of_agents)
        #actions = np.split(actions, number_of_agents)
        #rewards = np.split(rewards, number_of_agents)
        
        #agents = np.empty(4, 0)

        #for i in range(batchsize):
            
         #   print("\n\n")
        
        self.critic.feedDQN(train_batch)
        
        pass

class Runner(object):
    def __init__(self, config: Dict, environment) -> None:
        self.config = config
        self.environement = environment
        pass

    def test():
        print("tsgvhbjnk")

    
    def run(self):
        strhhj = "tzujh"
        tune.run(self.config['algorithm'], checkpoint_freq=1, config={
            "framework": self.config['framework'],
            "env": self.config['env'],
            #"rollout_fragment_length": 40,
           #"train_batch_size": 800,
            "callbacks": partial(MyCallback, self.environement),
           "num_cpus_per_worker":1
           
        })
        print("This is where stuff happens")