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
from collections import deque

import src.utils_folder.array_utils as ar_ut
from impact_approximator import ImpactApproximator
from src.data_loader.replay_buffer import ReplayBuffer
import torch
import mpu
import matplotlib.pyplot as plt


def discretizeactions(
    input, base
):  # TODO: This one is deprecated, switch to actions_to_nodes from array_utils
    stringinput = "".join(map(str, input))
    return int(stringinput, base)


def getBatchSize(batch):
    return int(str(batch).split("SampleBatch(", 1)[1].split(":", 1)[0])


def print_shape(batch):
    print("\n\nObs:")
    print(batch[SampleBatch.OBS].shape)

    print("\n\nObs_next:")
    print(batch[SampleBatch.NEXT_OBS].shape)

    print("\n\nAction:")
    print(batch[SampleBatch.ACTIONS].shape)

    print("\n\nReward:")
    print(batch[SampleBatch.REWARDS].shape)


def not_hot_encoded_batch_to_hot_encoded_batch(batch):
    cbatch = copy.deepcopy(batch)
    actionint = int(cbatch[SampleBatch.ACTIONS][0])
    hot_encoded = np.zeros(81)
    hot_encoded[actionint] = 1
    SampleBatch.__setitem__(
        cbatch, SampleBatch.ACTIONS, hot_encoded[np.newaxis, :]
    )
    return cbatch


class MyCallback(DefaultCallbacks):

    def __init__(self, config: Dict, legacy_callbacks_dict: Dict[str, callable] = None):
        self.one_hot_encoding = False
        self.batch_list = []
        self.batchsize = 0
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
        self.average = [0, 0, 0, 0]
        self.one = deque(maxlen=100)
        self.two = deque(maxlen=100)
        self.three = deque(maxlen=100)
        self.four = deque(maxlen=100)
        if self.one_hot_encoding == True:
            self.batchsize = 81
        if self.one_hot_encoding == False:
            self.batchsize = 256

        super().__init__(legacy_callbacks_dict=legacy_callbacks_dict)

    def _init_critics(self) -> List[ImpactApproximator]:
        return [ImpactApproximator(self.config, i) for i in range(self.num_agents)]

    def _init_replay_buffer(self) -> ReplayBuffer:
        return ReplayBuffer(self.config)

    def on_postprocess_trajectory(self, *, worker, episode, agent_id, policy_id, policies, postprocessed_batch, original_batches, **kwargs):
        self.count += 1

    
        if(postprocessed_batch.__len__() == 1):
            self.collected_postprocessed_batch = np.append(
                self.collected_postprocessed_batch, postprocessed_batch)

            # Wenn self.collected_postprocessed_batch batches von allen 4 agenten enthält, wird er an den replay buffer übergeben und dort weiter verarbeitet
            if agent_id == "prisoner_" + str(self.num_agents - 1):
                self.replay_buffer.add_data_to_buffer(
                    self.collected_postprocessed_batch, postprocessed_batch, self.action_space_sizes, one_hot_encoding=self.one_hot_encoding)
                self.collected_postprocessed_batch = np.array([])

            if self.replay_buffer.buffer[0].__len__() > 10:
                for i in range(0, 4):

                    #verkettetes Sample wird aus Replaybuffer für agenten i geholt
                    sample = self.replay_buffer.sample_batch(self.batchsize, i)

                    # critic für Agenten i mit sample für Agenten i trainieren
                    self.criticsarray[i].train_critic(sample)


                    #batch der größe 1 aus dem replay buffer holfen Impact Measurements upzudaten
                    batchfortim = self.replay_buffer.sample_batch(1, i)

                    #Implementierung des Impact Measurement updated nimmt nur hot encoded Actions an. deswegen manuelle umwandlung
                    if self.one_hot_encoding == False:
                        hotbatchfortim = not_hot_encoded_batch_to_hot_encoded_batch(
                            batchfortim)
                        self.criticsarray[i].update_impact_measurement(
                            torch.tensor(hotbatchfortim[SampleBatch.OBS]),
                            torch.tensor(hotbatchfortim[SampleBatch.ACTIONS])
                        )
                    if self.one_hot_encoding == True:
                        self.criticsarray[i].update_impact_measurement(
                            torch.tensor(batchfortim[SampleBatch.OBS]),
                            torch.tensor(batchfortim[SampleBatch.ACTIONS])
                        )

                    #print(self.criticsarray[i].get_q_values(torch.tensor(hotbatchfortim[SampleBatch.OBS])))
                    print("Agent ", i, ": ",self.criticsarray[i].get_tim_approximation())
                #criticdata = self.criticsarray[0].get_tim_approximation().detach().numpy()

    
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
                # "env": "cooperative_pong_v4",
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
