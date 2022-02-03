import functools
from logging import config, critical
from os import ctermid
from pdb import post_mortem
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.qmix import QMixTrainer
from ray.rllib.models import ModelCatalog
import hydra
from typing import Dict, Iterable
from ray.tune.trial import Trial
from ray.tune.ray_trial_executor import RayTrialExecutor
from typing import List

from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.tune import Callback
import numpy as np
from time import sleep
from ray.rllib import TorchPolicy
from critic import Critic
from functools import partial
from ray.rllib.policy.sample_batch import SampleBatch


import time
from gym.spaces import MultiDiscrete, Discrete

from numpy import asarray, dtype, float32, float64
from numpy import save
import mpu
import torch
from gym.spaces import MultiDiscrete, Discrete, Box


class MyCallback(DefaultCallbacks):
  
  

    def __init__(self, environment, legacy_callbacks_dict: Dict[str, callable] = None):
        self.environemnt = environment
        self.critic = Critic(environment)
        self.batch = np.array([])
        self.i = None
        self.e = None
        self.z = 0

        super().__init__(legacy_callbacks_dict=legacy_callbacks_dict)


    #def on_learn_on_batch(self, *, policy, train_batch, result, **kwargs):
   #     self.i += 1
   #     self.train_batch = train_batch
        #print("observation: ", train_batch["obs"], "actions: ", train_batch["actions"], "rewards: ", train_batch["rewards"], "agentindex: ", train_batch["agent_index"])
   #     agentindex = train_batch["agent_index"]
   #     observations = train_batch["obs"]
   #     actions = train_batch["actions"]
   #     rewards = train_batch["rewards"]
   #     batchsize = len(agentindex)
   #     number_of_agents = len(set(agentindex))
        #print(number_of_agents)
   #     print(agentindex)
   #     print("\n")
        #observations = np.split(observations, number_of_agents)
        #actions = np.split(actions, number_of_agents)
        #rewards = np.split(rewards, number_of_agents)
        
        #agents = np.empty(4, 0)

        #for i in range(batchsize):
            
         #   print("\n\n")
        #mpu.io.write('ppo.pickle', train_batch)
        #self.critic.feedDQN(train_batch)
        
        #print("train_batch")
        #print(train_batch)
   #     pass

    def on_postprocess_trajectory(self, *, worker, episode, agent_id, policy_id, policies, postprocessed_batch, original_batches, **kwargs):
        #print("before: ")
        #print(postprocessed_batch["obs"])

        #print("after:")
        #SampleBatch.__setitem__(postprocessed_batch, "obs", Discrete(350.0))
        #print(SampleBatch.CUR_OBS)
        #print(postprocessed_batch[SampleBatch.ACTIONS])
        self.batch = np.append(self.batch, postprocessed_batch)
        if(agent_id == "prisoner_3"):

            o0 = self.batch[0][SampleBatch.CUR_OBS][0][0]
            o1 = self.batch[1][SampleBatch.CUR_OBS][0][0]
            o2 = self.batch[2][SampleBatch.CUR_OBS][0][0]
            o3 = self.batch[3][SampleBatch.CUR_OBS][0][0]


            a0 = self.batch[0]["actions"][0]
            a1 = self.batch[1]["actions"][0]
            a2 = self.batch[2]["actions"][0]
            a3 = self.batch[3]["actions"][0]

            pb = postprocessed_batch.copy()
            


            #Preprozessor zur Prüfung der Gültigkeit der Daten
            action_encoder = ModelCatalog.get_preprocessor_for_space(Box(-300.0, 300.0, (4,), dtype=np.float32))
            
           
            #Erzeugung von validen Array mit Observations/Actions aller Agenten
            new_obs = action_encoder.transform(np.array([o0, o1, o2, o3], dtype=float32))
            new_actions = action_encoder.transform(np.array([a0, a1, a2, a3], dtype=float32))


            #zusammengefasste Observations und Actions werden in kopierten postprocessed_batch geschrieben
            SampleBatch.__setitem__(pb, SampleBatch.CUR_OBS, new_obs)
            SampleBatch.__setitem__(pb, SampleBatch.ACTIONS, new_actions)

            
            #mpu.io.write("/Users/julian/Desktop/ub.pickle", pb)
            #mpu.io.write("/Users/julian/Desktop/postprocessed.pickle", postprocessed_batch)

            #Übergabe des neuen batches an Critic
            self.critic.feedDQN(pb)


            self.batch = np.array([])

       

       
        pass

class Runner(object):
    def __init__(self, config: Dict, environment) -> None:
        self.config = config
        self.environement = environment
        pass

    

    
    def run(self):
        tune.run(self.config['algorithm'], checkpoint_freq=1, config={
            "framework": self.config['framework'],
            "env": self.config['env'],
            "rollout_fragment_length": 1,
            #"sgd_minibatch_size": 32,
            "train_batch_size":32,
            #"prioritized_replay": False,
            #"batch_mode": "complete_episodes",
            "callbacks": partial(MyCallback, self.environement),
            "num_gpus": 0,
            "num_workers": 1
            #"multiagent": {
            #"replay_mode": "lockstep",
            # },
            
           #"num_cpus_per_worker":1
           
        })
        print("This is where stuff happens")