from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.dqn.policies import DQNPolicy
from torch.nn import functional as F


class NonRolloutDQN(DQN):
    def __init__(
        self,
        policy: Union[str, Type[DQNPolicy]],
        env: Union[GymEnv, str],
        config: Dict,
        total_time_steps: int = 1_000_000,
        learning_rate: Union[float, Schedule] = 0.0001,
        buffer_size: int = 1000000,
        learning_starts: int = 50000,
        batch_size: int = 32,
        tau: float = 1,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
        callback=None,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "run",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        update_rate: int = 1000,
        agent_id: int = 0,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            replay_buffer_class,
            replay_buffer_kwargs,
            optimize_memory_usage,
            target_update_interval,
            exploration_fraction,
            exploration_initial_eps,
            exploration_final_eps,
            max_grad_norm,
            tensorboard_log,
            create_eval_env,
            policy_kwargs,
            verbose,
            seed,
            device,
            _init_setup_model,
        )
        self.num_collected_steps = 0
        self._total_timesteps = total_time_steps
        self.update_rate = update_rate
        self.config = config
        self.agent_id = agent_id
        _, callback = self._setup_learn(
            self._total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            tb_log_name,
        )

    def learn(self, actions, new_obs, rewards, dones, infos) -> "OffPolicyAlgorithm":

        self.num_collected_steps += 1

        # Retrieve reward and episode length if using Monitor wrapper
        self._update_info_buffer(infos, dones)

        # Store data in replay buffer (normalized action and unnormalized observation)
        self._store_transition(
            self.replay_buffer, actions, new_obs, rewards, dones, infos
        )

        self._update_current_progress_remaining(
            self.num_collected_steps, self._total_timesteps
        )

        # For DQN, check if the target network should be updated
        # and update the exploration schedule
        # For SAC/TD3, the update is dones as the same time as the gradient update
        # see https://github.com/hill-a/stable-baselines/issues/900
        self._on_step()

        if (
            self.num_collected_steps > 0
            and self.num_collected_steps > self.learning_starts
        ):
            if self.num_collected_steps % 1000 == 0:
                if self.agent_id == 0:
                    obs_to_track = torch.tensor(
                        [[10.0, 10.0, 10.0, 10.0], [150.0, 10.0, 10.0, 150.0]]
                    )

                    q_values = self.get_q_values(obs_to_track)
                    print("Q-values: ")
                    print(
                        "Train Net close left, action left: "
                        + str(float(q_values[0, 0]))
                    )
                    print(
                        "Train Net center, action left: " + str(float(q_values[1, 0]))
                    )
                self.train(
                    batch_size=self.batch_size, gradient_steps=self.gradient_steps
                )
        return self

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )

            if torch.isnan(replay_data.observations).any().item():
                obs_non_missing_data = torch.any(
                    ~torch.isnan(replay_data.observations), -1
                )
                replay_data = replay_data._replace(
                    observations=replay_data.observations[obs_non_missing_data],
                    actions=replay_data.actions[obs_non_missing_data],
                    next_observations=replay_data.next_observations[
                        obs_non_missing_data
                    ],
                    dones=replay_data.dones[obs_non_missing_data],
                    rewards=replay_data.rewards[obs_non_missing_data],
                )

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * self.gamma * next_q_values
                )

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(
                current_q_values, dim=1, index=replay_data.actions.long()
            )

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

    def get_q_values(self, obs):
        with torch.no_grad():
            q_values = self.q_net(obs)
        return q_values
