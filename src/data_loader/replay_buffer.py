import random
from collections import deque, namedtuple
from typing import Dict

import numpy as np
import torch
from numpy.random import choice

from src.data_loader.data_handler import BaseDataHandler, PrisonDataHandler


class ReplayBuffer(object):
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.data_handler = self._init_data_handler(config)
        self.im_config = config["impact_measurement"]
        self.buffer_size = self.im_config["buffer_size"]
        self.sampling_size = self.im_config["sampling_size"]

    def _init_data_handler(self, config) -> BaseDataHandler:
        if self.config["rl_env"]["name"] == "prison_v3":
            return PrisonDataHandler(config)
        elif self.config["rl_env"]["name"] == "pistonball_v5":
            raise NotImplementedError
        else:
            raise ValueError("No data_handler for specified env found!")

    def add_data_to_buffer(self, data):
        # - transform data via transformed_data = self.data_handler.transform_postprocessed_batch(data)
        # - add transformed data to buffer
        pass

    def sample_batch(self):
        # return batch with size self.sampling_size
        pass
