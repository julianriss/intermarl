from typing import Dict

import numpy as np
import torch

from src.models.impact_networks import SimpleFF


def get_impact_network(config: Dict) -> torch.nn.Module:
    if (
        config["rl_env"]["name"] == "prison_v3"
        or config["rl_env"]["name"] == "custom_prison"
    ):
        return SimpleFF(
            in_features=np.prod(config["rl_env"]["observation_space"].shape),
            out_features=config["rl_env"]["num_agents"],
            hidden_layer_size=config["impact_measurement"]["hidden_layer_size"],
        )
    elif False:
        raise NotImplementedError
    else:
        raise ValueError(
            "There is no impact network specified for the given environment!"
        )
