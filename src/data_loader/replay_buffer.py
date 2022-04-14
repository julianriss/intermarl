import random
from collections import deque, namedtuple
from typing import Dict

import numpy as np
import torch
from numpy.random import choice

from src.data_loader.data_handler import BaseDataHandler, PrisonDataHandler
from ray.rllib.policy.sample_batch import SampleBatch
from collections import deque
import mpu

class ReplayBuffer(object):
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.data_handler = self._init_data_handler(config)
        self.im_config = config["impact_measurement"]
        self.buffer_size = self.im_config["buffer_size"]
        self.sampling_size = self.im_config["sampling_size"]
        self.buffer = [deque(maxlen=self.buffer_size)] * 4
        self.push_count = 0

    def _init_data_handler(self, config) -> BaseDataHandler:
        if self.config["rl_env"]["name"] == "prison_v4":
            return PrisonDataHandler(config)
        elif self.config["rl_env"]["name"] == "pistonball_v5":
            raise NotImplementedError
        else:
            raise ValueError("No data_handler for specified env found!")

    def add_data_to_buffer(self, data, pb_structure, action_space_size, one_hot_encoding):
        rewarded_batch = self.data_handler.transform_postprocessed_batch(
            data, pb_structure, action_space_size, one_hot_encoding)

        
        for i in range(0, rewarded_batch.__len__()):
            self.buffer[i].append(rewarded_batch[i])

    pass

    def sample_batch(self, size, agentid):
        samples = random.choices(self.buffer[agentid], k=size)
       
        samples = SampleBatch.concat_samples(samples)
        
        #mpu.io.write("logs/samplesnew.pickle", samples)
        
        return samples
        
        
