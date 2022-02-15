import functools
from logging import config, critical
from os import ctermid
import os
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
from random import randrange

import gym
import time
from gym.spaces import MultiDiscrete, Discrete

from numpy import asarray, dtype, float32, float64, int32
from numpy import save
import mpu
import torch
from gym.spaces import MultiDiscrete, Discrete, Box
import copy









class MyCallback(DefaultCallbacks):
  
  

    def __init__(self, environment, num_agents, legacy_callbacks_dict: Dict[str, callable] = None):
        self.environemnt = environment
        self.critic = Critic(environment)
        self.batch_list = []
        self.batch = np.array([])
        self.i = None
        self.e = None
        self.z = 0
        self.num_agents = num_agents
        self.criticsarray = []

        for i in range(0, num_agents):
            critic = Critic(environment)
            self.criticsarray.append(critic)

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
        #print("this is one original_batch")
        #print(original_batches)
        #print("this was one original batch")
        self.batch = np.append(self.batch, postprocessed_batch)
        #print(self.batch)

        
            
        
        if(agent_id == "prisoner_" + str(self.num_agents-1)):
        
            observations = np.array([])
            observations_next = np.array([])
            actions = np.array([], dtype=int)

            for i in range(0,4):
                observations = np.append(observations, self.batch[i][SampleBatch.CUR_OBS][0][0])
                observations_next = np.append(observations_next, self.batch[i][SampleBatch.NEXT_OBS][0][0])
                actions = np.append(actions, self.batch[i][SampleBatch.ACTIONS][0])
           
            pb = copy.deepcopy(postprocessed_batch)
            

           
            #Preprozessor zur Prüfung der Gültigkeit der Daten
            state_encoder = ModelCatalog.get_preprocessor_for_space(Box(-300.0, 300.0, (4,), dtype=np.float32))
            action_encoder = ModelCatalog.get_preprocessor_for_space(MultiDiscrete([3,3,3,3]))
            
           
            #Erzeugung von validen Array mit Observations/Actions aller Agenten
            combined_obs = state_encoder.transform(observations)
            combined_next_obs = state_encoder.transform(observations_next)
            

            combined_actions = action_encoder.transform(actions)
            #print(combined_actions)
            
            #zusammengefasste Observations und Actions werden in kopierten postprocessed_batch geschrieben
            SampleBatch.__setitem__(pb, SampleBatch.CUR_OBS, combined_obs[np.newaxis, :])
            SampleBatch.__setitem__(pb, SampleBatch.NEXT_OBS, combined_next_obs[np.newaxis, :])
            SampleBatch.__setitem__(pb, SampleBatch.ACTIONS, combined_actions)


            rewardbatches = []
            for i in range(0,self.num_agents):
                rewardbatches.append(pb)
                SampleBatch.__setitem__(rewardbatches[i], SampleBatch.REWARDS, self.batch[i][SampleBatch.REWARDS])


            for i in range(0, self.num_agents):
                self.criticsarray[i].feedDQN(rewardbatches[i], i)

            

            self.batch = np.array([])

       

       
        pass

class Runner(object):
    def __init__(self, config: Dict, environment, num_agents) -> None:
        self.config = config
        self.environement = environment
        self.num_agents = num_agents
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
            "callbacks": partial(MyCallback, self.environement, self.num_agents),
            "num_gpus": 0,
            "num_workers": 1
            #"multiagent": {
            #"replay_mode": "lockstep",
            # },
            
           #"num_cpus_per_worker":1
           
        })
        print("This is where stuff happens")