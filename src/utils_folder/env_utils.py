from typing import Dict, Tuple

import gym
from gym import spaces
from pettingzoo.butterfly import prison_v3

from src.custom_envs.prison_add_obs import init_custom_prison_env


def get_simulation_env(config: Dict):
    rl_config = config["rl_env"]
    if rl_config["name"] == "prison_v3":
        assert (
            rl_config["num_agents"] % 2 == 0
        ), "Prison can only have an even number of agents!"
        return prison_v3.env(
            vector_observation=True,
            continuous=False,
            synchronized_start=False,
            identical_aliens=False,
            max_cycles=150,
            num_floors=int(rl_config["num_agents"] / 2),
            random_aliens=False,
        )
    elif rl_config["name"] == "custom_prison":
        assert (
            rl_config["num_agents"] % 2 == 0
        ), "Prison can only have an even number of agents!"
        return init_custom_prison_env(
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


class DummyEnvForSpaces(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render.modes": ["human"]}

    def __init__(self, action_space, observation_space, agent_id):
        super(DummyEnvForSpaces, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.agent_id = agent_id
        self.action_space = action_space
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
    if env_type == "prison_v3" or env_type == "custom_prison":
        action_space = get_prison_v3_im_action_space(info_dict) 
        observation_space = get_prison_v3_im_observation_space(info_dict)
        return action_space, observation_space
    elif env_type == "pistonball":
        raise NotImplementedError()
    else:
        raise ValueError("No valid env-type specified!")


def get_prison_v3_im_action_space(info_dict) -> spaces.Discrete:
    return info_dict["rl_env"]["action_space"]


def get_prison_v3_im_observation_space(info_dict) -> spaces.Box:
    return info_dict["rl_env"]["observation_space"]


def get_impact_dqn_env(config: Dict, agent_id: int):
    action_space, observation_space = get_env_single_agent_impact_action_and_observation_spaces(
        config["rl_env"]["name"], agent_id, config
    )
    return DummyEnvForSpaces(action_space, observation_space, agent_id)


def get_single_agent_dummy_env(config: Dict, agent_id: int) -> DummyEnvForSpaces:
    action_space = config["rl_env"]["sa_action_spaces"][agent_id]
    observation_space = config["rl_env"]["sa_observation_spaces"][agent_id]
    return DummyEnvForSpaces(action_space, observation_space, agent_id)
