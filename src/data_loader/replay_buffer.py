import random
from collections import deque, namedtuple
from typing import Dict

import numpy as np
import torch
from numpy.random import choice

from src.data_loader.data_handler import BaseDataHandler, PrisonDataHandler
from ray.rllib.policy.sample_batch import SampleBatch


class ReplayBuffer(object):
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.data_handler = self._init_data_handler(config)
        self.im_config = config["impact_measurement"]
        self.buffer_size = self.im_config["buffer_size"]
        self.sampling_size = self.im_config["sampling_size"]
        self.buffer = []
        self.push_count = 0

    def _init_data_handler(self, config) -> BaseDataHandler:
        if self.config["rl_env"]["name"] == "prison_v3":
            return PrisonDataHandler(config)
        elif self.config["rl_env"]["name"] == "pistonball_v5":
            raise NotImplementedError
        else:
            raise ValueError("No data_handler for specified env found!")

    def add_data_to_buffer(self, data, pb_structure, action_space_size):
        rewarded_batches = self.data_handler.transform_postprocessed_batch(
            data, pb_structure, action_space_size)

        if len(self.buffer) < self.buffer_size:
            self.buffer.append(rewarded_batches)
        else:
            self.buffer.append[self.push_count % self.buffer_size] = rewarded_batches
        self.push_count += 1
    pass

    def sample_batch(self):
        return random.sample(self.buffer, 1)
        
