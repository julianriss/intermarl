from typing import Dict, List

from src.models.custom_dqn import NonRolloutDQN
from src.models.custom_policies import RandomPolicy


def get_policy_dict(config: Dict) -> Dict:
    if config["algo_model_sharing"]:
        shared_policy = get_policy_for_agent(config, 0)
        return {
            agent_id: shared_policy
            for agent_id in range(config["rl_env"]["num_agents"])
        }
    else:
        return {
            agent_id: get_policy_for_agent(config, agent_id)
            for agent_id in range(config["rl_env"]["num_agents"])
        }


def get_policy_for_agent(config, agent_id):
    algo_type = config["algorithm"]["name"]
    if algo_type == "DQN":
        pass
    elif algo_type == "random":
        return RandomPolicy(config, agent_id)
    else:
        raise ValueError("No valid algorithm type chosen!")
