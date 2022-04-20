from typing import Tuple

import numpy as np
from gym.spaces import Box, Discrete


class RandomPolicy(object):
    def __init__(self, config, agent_id) -> None:
        self.config = config
        self.agent_id = agent_id
        self.action_space_type = self._get_action_space_type()
        self.lower_bound, self.upper_bound = self._get_action_space_bounds()

    def _get_action_space_type(self) -> str:
        if isinstance(self.config["rl_env"]["action_space"], Discrete):
            return "discrete"
        elif isinstance(self.config["rl_env"]["action_space"], Box):
            return "box"
        else:
            raise ValueError("No valid action space type for random policy chosen!")

    def _get_action_space_bounds(self) -> Tuple[float, float]:
        return 0, self.config["rl_env"]["action_space_sizes"][self.agent_id]

    def get_action(self, obs):
        if self.action_space_type == "discrete":
            return np.random.randint(self.lower_bound, self.upper_bound)
        elif self.action_space_type == "box":
            return np.random.uniform(self.lower_bound, self.upper_bound)
        else:
            return None

    def learn(self, actions, new_obs, rewards, dones, infos):
        pass
