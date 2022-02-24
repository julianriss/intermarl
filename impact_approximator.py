from email.policy import Policy
from typing import Dict

import torch
from ray.rllib.agents.dqn.dqn_torch_policy import DQNTorchPolicy, compute_q_values

import src.utils_folder.array_utils as ar_ut
import src.utils_folder.network_utils as nw_ut

dir = "/Users/julian/Desktop/"


class ImpactApproximator(object):
    def __init__(self, config: Dict, agent_id: int):
        self.agent_id = agent_id
        self.config = config
        self.env_config = config["rl_env"]
        self.num_agents = self.env_config["num_agents"]
        self.action_space_sizes = self.env_config["action_space_sizes"]
        self.tim_step_size = config["impact_measurement"]["tim_initial_step_size"]
        self.tim_step_size_decay = config["impact_measurement"]["tim_step_size_decay"]
        self.sim_learning_rate = config["impact_measurement"]["sim_learning_rate"]

        self.observation_space = self.env_config["observation_space"]
        self.action_space = self.env_config["action_space"]

        self.tim_measurement = self._init_tim_measurement()
        self.sim_measurement = (
            self._init_sim_measurement()
        )  # TODO: Would be cleaner as an own class!
        self.sim_criterion = torch.nn.MSELoss()
        self.sim_optimizer = torch.optim.Adam(
            self.sim_measurement.parameters(),
            lr=config["impact_measurement"]["sim_learning_rate"],
        )

        self.critic = self._init_critic()

    def _init_critic(self) -> Policy:
        return DQNTorchPolicy(
            self.observation_space, self.action_space, {"num_gpus": 0, "num_workers": 1}
        )

    def _init_tim_measurement(self) -> torch.Tensor:
        return torch.ones(self.num_agents) / self.num_agents

    def _init_sim_measurement(self) -> torch.nn.Module:
        return nw_ut.get_impact_network(self.config)

    def get_q_values(self, obs):
        q_tp1, _, _, _ = compute_q_values(
            self.critic,
            self.critic.target_models[self.critic.model],
            {"obs": obs},
            explore=False,
            is_training=False,
        )
        return q_tp1

    def train_critic(self, batch):
        self.critic.learn_on_batch(batch)

    def update_impact_measurement(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        impact_samples = self.get_impact_samples_for_batch(obs, actions)
        self._update_tim(impact_samples)
        self._update_sim(obs, impact_samples)
        return impact_samples

    def _update_tim(self, impact_samples: torch.Tensor):
        self.tim_measurement = (
            1.0 - self.tim_step_size
        ) * self.tim_measurement + self.tim_step_size * torch.mean(impact_samples, 0)
        self.tim_step_size *= (
            self.tim_step_size_decay
        )  # TODO: The learning rate and its way to be updated is something to play with!

    def _update_sim(self, obs: torch.Tensor, impact_samples: torch.Tensor):
        self.sim_optimizer.zero_grad()
        pred_sim = self.sim_measurement(obs)
        loss = self.sim_criterion(pred_sim, impact_samples)
        loss.backward()
        self.sim_optimizer.step()

    def get_tim_approximation(self) -> torch.Tensor:
        return self.tim_measurement

    def get_sim_approximation(self, obs: torch.Tensor):
        with torch.no_grad():
            return self.sim_measurement(obs)

    def get_impact_samples_for_batch(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Computes the impact samples for a given batch.
        Args:
            obs (torch.Tensor): batch of observations (shape=(#samples, #features))
            actions (torch.Tensor): batch of one-hot encoded actions
        Returns:
            torch.Tensor: shape=(#samples, #agents)
        """
        q_values = self.get_q_values(obs)
        decoded_actions = ar_ut.decode_discrete_actions(
            actions, self.action_space_sizes, ret_as_joint_actions=True
        )
        neighbors = ar_ut.get_neighbors_to_actions(
            decoded_actions, self.action_space_sizes
        )
        return ar_ut.get_impact_samples_from_q_values(q_values, neighbors)
