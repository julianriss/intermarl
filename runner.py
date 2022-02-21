import copy
import functools
import os
import time
from functools import partial
from logging import config, critical
from os import ctermid
from pdb import post_mortem
from random import randrange
from time import sleep
from typing import Dict, Iterable, List

import gym
import hydra
import mpu
import numpy as np
import torch
from gym.spaces import Box, Discrete, MultiDiscrete
from numpy import asarray, dtype, float32, float64, int32, save
from ray import tune
from ray.rllib import TorchPolicy
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.qmix import QMixTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.tune import Callback
from ray.tune.ray_trial_executor import RayTrialExecutor
from ray.tune.trial import Trial

from critic import Critic


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
        self.concatenatedbatch = []
        self.batchsize = 81  # TODO: Why is this needed? Makes no sense
        self.batchcounter = 0
        self.i = None
        self.e = None
        self.z = 0
        self.config = config
        self.env_config = config["rl_env"]
        self.num_agents = self.env_config["num_agents"]
        self.criticsarray = self._init_critics()

        super().__init__(legacy_callbacks_dict=legacy_callbacks_dict)

    def _init_critics(self) -> List[Critic]:
        return [Critic(self.config, i) for i in range(self.num_agents)]

    def on_postprocess_trajectory(
        self,
        *,
        worker,
        episode,
        agent_id,
        policy_id,
        policies,
        postprocessed_batch,
        original_batches,
        **kwargs
    ):
        self.batch = np.append(self.batch, postprocessed_batch)

        if agent_id == "prisoner_" + str(
            self.num_agents - 1
        ):  # TODO: this is only done for the last agent, generalize this!

            observations = np.array(
                []
            )  # TODO: Should be made dynamic, and should be stored where the batches are needed. Not needed in the callback class but rather in the critic, maybe even write a new class that handles data.
            observations_next = np.array([])
            actions = np.array([], dtype=int)

            for i in range(0, 4):  # TODO: Stuff like this should be in an own method
                observations = np.append(
                    observations, self.batch[i][SampleBatch.CUR_OBS][0][0]
                )
                observations_next = np.append(
                    observations_next, self.batch[i][SampleBatch.NEXT_OBS][0][0]
                )
                actions = np.append(actions, self.batch[i][SampleBatch.ACTIONS][0])

            pb = copy.deepcopy(postprocessed_batch)

            # TODO: Stuff like this should be in an own method
            # Preprozessor zur Prüfung der Gültigkeit der Daten
            state_encoder = ModelCatalog.get_preprocessor_for_space(
                Box(-300.0, 300.0, (4,), dtype=np.float32)
            )
            action_encoder = ModelCatalog.get_preprocessor_for_space(Discrete(81))

            # Erzeugung von validen Array mit Observations/Actions aller Agenten
            combined_obs = state_encoder.transform(observations)
            combined_next_obs = state_encoder.transform(observations_next)

            combined_actions = action_encoder.transform(discretizeactions(actions, 3))

            # zusammengefasste Observations und Actions werden in kopierten postprocessed_batch geschrieben
            SampleBatch.__setitem__(
                pb, SampleBatch.CUR_OBS, combined_obs[np.newaxis, :]
            )
            SampleBatch.__setitem__(
                pb, SampleBatch.NEXT_OBS, combined_next_obs[np.newaxis, :]
            )
            SampleBatch.__setitem__(
                pb, SampleBatch.ACTIONS, combined_actions[np.newaxis, :]
            )

            rewardbatches = []
            # TODO: Stuff like this should be in an own method
            for i in range(0, self.num_agents):
                rewardbatches.append(pb)
                SampleBatch.__setitem__(
                    rewardbatches[i],
                    SampleBatch.REWARDS,
                    self.batch[i][SampleBatch.REWARDS],
                )

                # verarbeitete Samples werden verkettet und in concatenatedbatch geschrieben. Bei vier agenten hat
                # concatbatch dann die größe vier und an jeder Position einen Batch in der größe der vorher definierten
                # batch size
                if len(self.concatenatedbatch) < 4:
                    self.concatenatedbatch.append(rewardbatches[i])
                self.concatenatedbatch[i] = SampleBatch.concat(
                    self.concatenatedbatch[i], rewardbatches[i]
                )

            # TODO: Stuff like this should be in an own method
            if getBatchSize(self.concatenatedbatch[0]) == self.batchsize:
                for i in range(0, self.num_agents):
                    self.criticsarray[i].train_critic(self.concatenatedbatch[i])
                    impact_samples = self.criticsarray[i].get_impact_samples_for_batch(
                        self.concatenatedbatch[i][SampleBatch.OBS],
                        self.concatenatedbatch[i]["actions"],
                    )  # This returns the impact samples! Can be used to train the tim estimation. I would say, you simply track the moving average. TBD.
                    print("Impact Samples i", impact_samples)
                self.concatenatedbatch = []
            self.batch = np.array([])
        pass


class Runner(object):
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.env_config = config["rl_env"]
        self.num_agents = self.env_config["num_agents"]

    def run(self):
        tune.run(
            self.config["algorithm"]["name"],
            checkpoint_freq=1,
            config={
                "framework": self.config["framework"],
                "env": self.env_config["name"],
                "rollout_fragment_length": 1,
                # "sgd_minibatch_size": 32,
                "train_batch_size": self.config["algorithm"]["train_batch_size"],
                # "prioritized_replay": False,
                # "batch_mode": "complete_episodes",
                "callbacks": partial(MyCallback, self.config),
                "num_gpus": 0,
                "num_workers": 1
                # "multiagent": {
                # "replay_mode": "lockstep",
                # },
                # "num_cpus_per_worker":1
            },
        )
