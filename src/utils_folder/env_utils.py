from typing import Dict, Tuple

import gym
from gym import spaces
from pettingzoo.butterfly import prison_v4


def get_simulation_env(config: Dict):
    rl_config = config["rl_env"]
    if rl_config["name"] == "prison_v4":
        assert (
            rl_config["num_agents"] % 2 == 0
        ), "Prison can only have an even number of agents!"
        return prison_v4.env(
            vector_observation=True,
            continuous=False,
            synchronized_start=False,
            identical_aliens=False,
            max_cycles=150,
            num_floors=int(rl_config["num_agents"] / 2),
            random_aliens=False,
        )
    else:
        raise ValueError("No valid environment chosen!")


def get_impact_dqn_env(config: Dict, agent_id: int):
    return DummyEnvForSpaces(config["rl_env"]["name"], agent_id, config)


class DummyEnvForSpaces(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render.modes": ["human"]}

    def __init__(self, env_type: str, agent_id: int, info_dict: Dict = None):
        super(DummyEnvForSpaces, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.agent_id = agent_id
        action_space, observation_space = get_env_single_agent_impact_action_and_observation_spaces(
            env_type, agent_id, info_dict
        )
        self.action_space = action_space
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = observation_space

    def step(self, action):
        return None

    def reset(self):
        return None

    def render(self, mode="human"):
        return None

    def close(self):
        return None


def get_env_single_agent_impact_action_and_observation_spaces(
    env_type: str, agent_id: int, info_dict: Dict = None
) -> Tuple[spaces.Space]:
    if env_type == "prison_v4":
        action_space = get_prison_v4_im_action_space(info_dict)
        observation_space = get_prison_v4_im_observation_space(info_dict)
        return action_space, observation_space
    elif env_type == "pistonball":
        raise NotImplementedError()
    else:
        raise ValueError("No valid env-type specified!")


def get_prison_v4_im_action_space(info_dict) -> spaces.Discrete:
    return info_dict["rl_env"]["action_space"]


def get_prison_v4_im_observation_space(info_dict) -> spaces.Box:
    return info_dict["rl_env"]["observation_space"]
