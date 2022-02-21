from email.policy import Policy
from ray.rllib.agents.dqn.dqn_torch_policy import DQNTorchPolicy
from ray.rllib.agents.dqn.dqn_torch_policy import compute_q_values
import torch
import src.utils_folder.array_utils as ar_ut
from typing import Dict


dir = "/Users/julian/Desktop/"



class Critic(object):

    def test(self):
        print("test")
        return "z"

    def __init__(self, config: Dict, agent_id: int):
        self.agent_id = agent_id
        self.config = config
        self.env_config = config["rl_env"]
        self.num_agents = self.env_config["num_agents"]
        self.action_space_sizes = self.env_config["action_space_sizes"]

        self.observation_space = self.env_config["observation_space"]
        self.action_space = self.env_config["action_space"]

        self.critic = self._init_critic()

    def _init_critic(self) -> Policy:
        return DQNTorchPolicy(self.observation_space, self.action_space, {"num_gpus": 0,
            "num_workers": 1,
            })

    def get_q_values(self, obs):
        q_tp1, _, _, _ = compute_q_values(
        self.critic,
        self.critic.target_models[self.critic.model],
        {"obs": obs},
        explore=False,
        is_training=False)
        return q_tp1

    def train_critic(self, batch):
        self.critic.learn_on_batch(batch)

    def get_impact_samples_for_batch(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Computes the impact samples for a given batch.
        Args:
            obs (torch.Tensor): batch of observations (shape=(#samples, #features))
            actions (torch.Tensor): batch of one-hot encoded actions
        Returns:
            torch.Tensor: shape=(#samples, #agents)
        """
        q_values = self.get_q_values(obs)
        decoded_actions = ar_ut.decode_discrete_actions(actions, self.action_space_sizes, ret_as_joint_actions=True)
        neighbors = ar_ut.get_neighbors_to_actions(decoded_actions, self.action_space_sizes)
        return ar_ut.get_impact_samples_from_q_values(q_values, neighbors)

