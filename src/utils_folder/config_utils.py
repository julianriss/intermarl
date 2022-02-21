from typing import Dict
from gym.spaces import Discrete, Box
from numpy import float32
import numpy as np

def enrich_config_file(config: Dict):
    enricht_config_by_env_specifications(config)


def enricht_config_by_env_specifications(config: Dict):
    if config["rl_env"]["name"] == "prison_v3":
        config["rl_env"]["action_space_sizes"] = tuple([3 for k in range(config["rl_env"]["num_agents"])])
        config["rl_env"]["observation_space"] = Box(-300., 300., (config["rl_env"]["num_agents"],), float32)
        config["rl_env"]["action_space"] = Discrete(np.prod(config["rl_env"]["action_space_sizes"]))

