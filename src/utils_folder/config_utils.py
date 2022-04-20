from typing import Dict

import numpy as np
from gym.spaces import Box, Discrete
from numpy import float32


def enrich_config_file(config: Dict):
    enricht_config_by_env_specifications(config)


def enricht_config_by_env_specifications(config: Dict):
    if config["rl_env"]["name"] == "prison_v4":
        config["rl_env"]["action_space_sizes"] = tuple(
            [3 for k in range(config["rl_env"]["num_agents"])]
        )
        config["rl_env"]["observation_space"] = Box(
            0.0, 300.0, (config["rl_env"]["num_agents"],), float32
        )
        config["rl_env"]["action_space"] = Discrete(
            np.prod(config["rl_env"]["action_space_sizes"])
        )

        config["rl_env"]["sa_observation_spaces"] = [
            Box(0.0, 300.0, (1,), float32)
            for i in range(config["rl_env"]["num_agents"])
        ]
        config["rl_env"]["sa_action_spaces"] = [
            Discrete(3) for i in range(config["rl_env"]["num_agents"])
        ]
