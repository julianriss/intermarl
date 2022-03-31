import copy
from functools import partial
from logging import config, critical
from os import ctermid
from pdb import post_mortem
from random import randrange
from time import sleep
from typing import Dict, List

from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.integration.wandb import WandbLoggerCallback

import numpy as np
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch

import src.utils_folder.array_utils as ar_ut
from impact_approximator import ImpactApproximator
from src.data_loader.replay_buffer import ReplayBuffer

import mpu


def discretizeactions(
    input, base
):  # TODO: This one is deprecated, switch to actions_to_nodes from array_utils
    stringinput = "".join(map(str, input))
    return int(stringinput, base)


def getBatchSize(batch):
    return int(str(batch).split("SampleBatch(", 1)[1].split(":", 1)[0])


class MyCallback(DefaultCallbacks):

    def __init__(self, config: Dict, legacy_callbacks_dict: Dict[str, callable] = None):
        self.batch_list = []
        self.batch = np.array([])
        self.collected_postprocessed_batch = np.array([])
        self.concatenatedbatch = []  
        self.batchcounter = 0
        self.i = None
        self.e = None
        self.z = 0
        self.config = config
        self.env_config = config["rl_env"]
        self.num_agents = self.env_config["num_agents"]
        self.action_space_sizes = self.env_config["action_space_sizes"]
        self.criticsarray = self._init_critics()
        self.replay_buffer = self._init_replay_buffer()
        self.count = 0
        self.average = [0,0,0,0]

        super().__init__(legacy_callbacks_dict=legacy_callbacks_dict)

    def _init_critics(self) -> List[ImpactApproximator]:
        return [ImpactApproximator(self.config, i) for i in range(self.num_agents)]

    def _init_replay_buffer(self) -> ReplayBuffer:
        return ReplayBuffer(self.config)

    def on_postprocess_trajectory(self, *, worker, episode, agent_id, policy_id, policies, postprocessed_batch, original_batches, **kwargs):
        self.count += 1

        if(postprocessed_batch.__len__() == 1):
            self.collected_postprocessed_batch = np.append(self.collected_postprocessed_batch, postprocessed_batch)

            #Wenn self.collected_postprocessed_batch batches von allen 4 agenten enthält, wird er an den replay buffer übergeben und dort weiter verarbeitet  
            if agent_id == "prisoner_" + str(self.num_agents - 1):
                self.replay_buffer.add_data_to_buffer(self.collected_postprocessed_batch, postprocessed_batch, self.action_space_sizes)
                self.collected_postprocessed_batch = np.array([])


            if self.replay_buffer.push_count > 1:
                for i in range(0, 4):

                    #Sample wird aus Replaybuffer für agenten i geholt
                    sample = self.replay_buffer.sample_batch()[0][i]

                    #mpu.io.write("logs/sample" + str(i) + ".pickle", sample)

                    #critic für Agenten i mit sample für Agenten i trainieren
                    self.criticsarray[i].train_critic(sample)
                    print("Agent ", i, " :", self.criticsarray[i].get_tim_approximation(), " Sum: ",  self.criticsarray[i].get_tim_approximation().sum())
                    
                
                    self.criticsarray[i].update_impact_measurement(
                        sample[SampleBatch.OBS],
                        sample[SampleBatch.ACTIONS],
                    )
        return 


class Runner(object):
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.env_config = config["rl_env"]
        self.num_agents = self.env_config["num_agents"]

    def run(self):
        tune.run(
            self.config["algorithm"]["name"],
            checkpoint_freq=1,
            # callbacks=[WandbLoggerCallback(
            #    project="Optimization_Project",
            #    api_key="",
            #    log_config=True)],
            config={
                "framework": self.config["framework"],
                "env": self.env_config["name"],
                # "rollout_fragment_length": 81,
                # "sgd_minibatch_size": 32,
                # "train_batch_size": self.config["algorithm"]["train_batch_size"],
                # "prioritized_replay": False,
                # "batch_mode": "complete_episodes",
                "callbacks": partial(MyCallback, self.config),
                # "num_gpus": 0,
                "num_workers": 1,



            },
        )
