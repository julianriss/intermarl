import copy
from functools import partial
from typing import Dict, List

import mpu
import numpy as np
import torch
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.dqn.dqn_torch_policy import compute_q_values
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.tune.integration.wandb import WandbLoggerCallback

import src.utils_folder.array_utils as ar_ut
from impact_approximator import ImpactApproximator
from src.data_loader.replay_buffer import ReplayBuffer


def getBatchSize(batch):
    return int(str(batch).split("SampleBatch(", 1)[1].split(":", 1)[0])


def print_shape(batch):
    print("\n\nObs:")
    print(batch[SampleBatch.OBS].shape)

    print("\n\nObs_next:")
    print(batch[SampleBatch.NEXT_OBS].shape)

    print("\n\nAction:")
    print(batch[SampleBatch.ACTIONS].shape)

    print("\n\nReward:")
    print(batch[SampleBatch.REWARDS].shape)


def not_hot_encoded_batch_to_hot_encoded_batch(batch):
    cbatch = copy.deepcopy(batch)
    actionint = int(cbatch[SampleBatch.ACTIONS][0])
    hot_encoded = np.zeros(81)
    hot_encoded[actionint] = 1
    SampleBatch.__setitem__(cbatch, SampleBatch.ACTIONS, hot_encoded[np.newaxis, :])
    return cbatch


class MyCallback(DefaultCallbacks):
    def __init__(self, config: Dict, legacy_callbacks_dict: Dict[str, callable] = None):
        self.one_hot_encoding = False
        self.collected_postprocessed_batch = np.array([])
        self.config = config
        self.env_config = config["rl_env"]
        self.im_config = config["impact_measurement"]
        self.num_agents = self.env_config["num_agents"]
        self.action_space_sizes = self.env_config["action_space_sizes"]
        self.critic_batch_size = self.im_config["critic_batch_size"]
        self.criticsarray = self._init_critics()
        self.replay_buffer = self._init_replay_buffer()
        self.count = 0
        if self.one_hot_encoding == True:
            self.critic_batch_size = 81

        super().__init__(legacy_callbacks_dict=legacy_callbacks_dict)

    def _init_critics(self) -> List[ImpactApproximator]:
        return [ImpactApproximator(self.config, i) for i in range(self.num_agents)]

    def _init_replay_buffer(self) -> ReplayBuffer:
        return ReplayBuffer(self.config)

    def on_postprocess_trajectory(
        self,
        *,
        worker,
        episode,
        agent_id,
        policy_id,
        policies,
        postprocessed_batch,
        original_batches,
        **kwargs
    ):
        self.count += 1
        # print("Count of data: " + str(self.count))
        if postprocessed_batch.__len__() > 1:
            # TODO: This is when the episode ends. Needs to be handled and not skipped!
            mpu.io.write(
                "logs/postprocessed_batch_too_long.pickle", postprocessed_batch
            )
        if postprocessed_batch.__len__() == 1:
            self.collected_postprocessed_batch = np.append(
                self.collected_postprocessed_batch, postprocessed_batch
            )
            if (
                postprocessed_batch["rewards"][0] != 0
                and postprocessed_batch["new_obs"][0][0] > 0.0
            ):
                mpu.io.write("logs/postprocessed_batch.pickle", postprocessed_batch)

            # Wenn self.collected_postprocessed_batch batches von allen 4 agenten enthält, wird er an den replay buffer übergeben und dort weiter verarbeitet
            if int(agent_id.split("_")[-1]) == (self.num_agents - 1):
                # TODO: This does not work if some agents die or the order changes at some point
                self.replay_buffer.add_data_to_buffer(
                    self.collected_postprocessed_batch,
                    postprocessed_batch,
                    self.action_space_sizes,
                    one_hot_encoding=self.one_hot_encoding,
                )
                self.collected_postprocessed_batch = np.array([])

            self._update_critics()
            self._update_impact_measurements()
            # criticdata = self.criticsarray[0].get_tim_approximation().detach().numpy()
        return

    def _update_impact_measurements(self):
        if (self.count / self.num_agents) % self.im_config["impact_update_rate"] == 0:
            for agent in range(self.num_agents):
                batch_for_tim = self.replay_buffer.draw_latest_data(
                    self.im_config["impact_batch_size"], agent
                )
                if self.one_hot_encoding == False:
                    batch_for_tim = not_hot_encoded_batch_to_hot_encoded_batch(
                        batch_for_tim
                    )
                self.criticsarray[agent].update_impact_measurement(
                    torch.tensor(batch_for_tim[SampleBatch.OBS]),
                    torch.tensor(batch_for_tim[SampleBatch.ACTIONS]),
                )

    def on_learn_on_batch(
        self, *, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs
    ) -> None:
        # TODO: Only valid for four agents and the current state of prison_v4!
        multi_agent_0_critic_obs_to_track = {
            "close_left_1": [10.0, 10.0, 10.0, 10.0],
            "close_left_2": [10.0, 90.0, 90.0, 90.0],
            "center_1": [120.0, 10.0, 10.0, 10.0],
            "close_right_1": [245.0, 90.0, 90.0, 90.0],
            "small_random": [0.1, -0.1, 0.2, -0.11],
        }
        multi_agent_0_critic_actions_to_track = {
            "left_1": [0, 0, 0, 0],
            "left_2": [0, 1, 1, 1],
            "idle": [1, 2, 2, 2],
            "right_1": [2, 1, 1, 1],
        }
        for ma_state_desc, ma_obs in multi_agent_0_critic_obs_to_track.items():
            ma_obs_tensor = torch.tensor([ma_obs])
            q_values_to_track = self.criticsarray[0].get_q_values(ma_obs_tensor)
            for (
                ma_action_desc,
                ma_actions,
            ) in multi_agent_0_critic_actions_to_track.items():
                action_index = ar_ut.actions_to_nodes(
                    np.array([ma_actions]), self.env_config["action_space_sizes"]
                )
                q_value = q_values_to_track.index_select(1, torch.tensor(action_index))
                result[
                    "ma_Q-value/obs type: "
                    + ma_state_desc
                    + "; action type: "
                    + ma_action_desc
                ] = float(q_value)

        single_agent_obs_to_track = {
            "close_left_1": [10.0],
            "center_1": [120.0],
            "close_right_1": [245.0],
            "small_random": [0.1],
        }
        single_agent_actions_to_track = {"left": [0], "idle": [1], "right": [2]}
        for sa_state_desc, sa_obs in single_agent_obs_to_track.items():
            sa_obs_tensor = torch.tensor([sa_obs])
            sa_q_values, _, _, _ = compute_q_values(
                policy,
                policy.target_models[policy.model],
                {"obs": sa_obs_tensor},
                explore=False,
                is_training=False,
            )
            for action_desc, sa_actions in single_agent_actions_to_track.items():
                q_value = sa_q_values.index_select(1, torch.tensor(sa_actions))
                result[
                    "single_agent_Q-value/obs type: "
                    + sa_state_desc
                    + "; action type: "
                    + action_desc
                ] = float(q_value)

    def _update_critics(self):
        if (self.count / self.num_agents) % self.im_config["critic_update_rate"] == 0:
            print(
                "Update_Critics step: "
                + str(
                    (self.count / self.num_agents)
                    / self.im_config["critic_update_rate"]
                )
            )
            for i in range(self.num_agents):
                # verkettetes Sample wird aus Replaybuffer für agenten i geholt
                sample = self.replay_buffer.sample_batch(self.critic_batch_size, i)

                # critic für Agenten i mit sample für Agenten i trainieren
                self.criticsarray[i].train_critic(sample)


class Runner(object):
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.env_config = config["rl_env"]
        self.num_agents = self.env_config["num_agents"]

    def run(self):
        tune.run(
            self.config["algorithm"]["name"],
            checkpoint_freq=1,
            callbacks=[
                WandbLoggerCallback(
                    project="riss-intermarl", entity="fpieroth", log_config=True
                )
            ],
            config={
                "framework": self.config["framework"],
                "env": self.env_config["name"],
                # "env": "cooperative_pong_v4",
                # "rollout_fragment_length": 81,
                # "sgd_minibatch_size": 32,
                # "train_batch_size": self.config["algorithm"]["train_batch_size"],
                # "prioritized_replay": False,
                # "batch_mode": "complete_episodes",
                "callbacks": partial(MyCallback, self.config),
                # "num_gpus": 0,
                "num_workers": 1,
                "logger_config": {
                    "wandb": {"project": "riss-intermarl", "entity": "fpieroth"}
                },
            },
        )
