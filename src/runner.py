import copy
from typing import Any, Dict

import numpy as np
import torch
import wandb

import src.utils_folder.algo_utils as algo_ut
import src.utils_folder.array_utils as ar_ut
import src.utils_folder.env_utils as env_ut
from impact_approximator import ImpactApproximator


class Runner(object):
    def __init__(self, config) -> None:
        self.config = config
        self.env = env_ut.get_simulation_env(config)
        self.num_agents = self.config["rl_env"]["num_agents"]
        self.total_num_simulations = self.config["total_num_simulations"]
        self.agent_policies = self._init_agent_policies()
        self.impact_measurers = self._init_impact_measurers()
        self.step_info_dict = {agent_id: None for agent_id in range(self.num_agents)}
        self.impact_batch_dict = self._get_empty_impact_batch_dict()
        self.impact_data_batch_size = 0
        self.reward = 0
        wandb.init()

    def _init_impact_measurers(self) -> Dict[int, ImpactApproximator]:
        return {i: ImpactApproximator(self.config, i) for i in range(self.num_agents)}

    def _get_empty_impact_batch_dict(self):
        data_dict = {"obs": [], "actions": [], "rewards": [], "dones": []}
        return {
            agent_id: copy.deepcopy(data_dict) for agent_id in range(self.num_agents)
        }

    def _init_agent_policies(self) -> Dict:
        return algo_ut.get_policy_dict(self.config)

    def _get_action_from_policy(self, obs: np.ndarray, done: bool, agent_id: int):
        action_for_buffer = self.agent_policies[agent_id].get_action(obs)
        action_for_env = action_for_buffer
        if done:
            action_for_env = None
        return action_for_env, action_for_buffer

    def _store_transition_data(self, observation, action, reward, done, info, agent_id):
        self.step_info_dict[agent_id] = {
            "obs": observation,
            "actions": np.array([action]),
            "rewards": np.array([reward]),
            "dones": np.array([done]),
        }
        # we neglect info at this point!

    def _update_policies(self):
        for agent_id in range(self.num_agents):
            agent_step_data = self.step_info_dict[agent_id]
            self.agent_policies[agent_id].learn(
                actions=agent_step_data["actions"],
                new_obs=agent_step_data["obs"],
                rewards=agent_step_data["rewards"],
                dones=agent_step_data["dones"],
                infos=[{}],
            )

    def _train_critics_of_impact_measurers(self):
        impact_transition_data = self._get_impact_transition_data()
        for agent_id in range(self.num_agents):
            agent_impact_transition_data = impact_transition_data[agent_id]
            self.impact_measurers[agent_id].train_critic(
                actions=agent_impact_transition_data["actions"],
                new_obs=agent_impact_transition_data["obs"],
                rewards=agent_impact_transition_data["rewards"],
                dones=agent_impact_transition_data["dones"],
                infos=[{}],
            )
        self._add_data_to_impact_batch_dict(impact_transition_data)

    def _add_data_to_impact_batch_dict(self, impact_transition_data):
        self.impact_data_batch_size += 1
        for agent_id, agent_data in impact_transition_data.items():
            for data_key, data in agent_data.items():
                self.impact_batch_dict[agent_id][data_key].append(data)

    def _get_impact_transition_data(self) -> Dict:
        impact_transition_data = {}
        for agent_id in range(self.num_agents):
            agent_impact_transition_data = {}
            agent_impact_transition_data["obs"] = np.array(
                self.ret_list_of_nested_dict_by_common_key(self.step_info_dict, "obs")
            ).T
            agent_impact_transition_data["actions"] = np.array(
                [
                    ar_ut.actions_to_nodes(
                        np.array(
                            self.ret_list_of_nested_dict_by_common_key(
                                self.step_info_dict, "actions"
                            )
                        ).squeeze(),
                        self.config["rl_env"]["action_space_sizes"],
                    )
                ]
            )
            agent_impact_transition_data["rewards"] = self.step_info_dict[agent_id][
                "rewards"
            ]
            agent_impact_transition_data["dones"] = self.step_info_dict[agent_id][
                "dones"
            ]
            impact_transition_data[agent_id] = agent_impact_transition_data
        return impact_transition_data

    def _update_impact_measures(self):
        for agent_id, impact_measurer in self.impact_measurers.items():
            impact_measurer.update_impact_measurement(
                torch.tensor(
                    np.array(self.impact_batch_dict[agent_id]["obs"])
                ).squeeze(),
                torch.tensor(np.array(self.impact_batch_dict[agent_id]["actions"])),
            )
        self.impact_batch_dict = self._get_empty_impact_batch_dict()
        self.impact_data_batch_size = 0

    @staticmethod
    def ret_list_of_nested_dict_by_common_key(nested_dict: Dict[Any, Dict], common_key):
        return [nested_dict[outer_key][common_key] for outer_key in nested_dict.keys()]

    def run(self):
        self.env.reset()
      
        for i in range(self.total_num_simulations):
            #if i % 2_000 == 0:
                #print("On step: " + str(i))
            done_list = []
            for agent_id in range(self.num_agents):
                obs, _, done, _ = self.env.last(agent_id = agent_id)
                done_list.append(done)
                action_for_env, action_for_buffer = self._get_action_from_policy(
                    obs, done, agent_id
                )
                self.env.step(action_for_env)
                if len(done_list) == self.num_agents and all(done_list):
                    self.env.reset()
                new_obs, new_reward, _, new_info = self.env.last(agent_id = agent_id)
                self._store_transition_data(
                    new_obs, action_for_buffer, new_reward, done, new_info, agent_id
                )
                self.reward += self.step_info_dict[agent_id]["rewards"][0]
                #print(self.reward)
            self._update_policies()
            #self._train_critics_of_impact_measurers()
            #if (
            #    self.impact_data_batch_size
            #    % self.config["impact_measurement"]["impact_batch_size"]
            #    == 0
            #    and self.impact_data_batch_size > 0
            #):
            #    self._update_impact_measures()

            ## Maybe some logging from this point onward ##
            if i % 10000 == 0 and i > 0:
                print("Step: ", i)
                ar = self.reward / 4
                print("Average reward last rollout: " + str(ar))
                wandb.log({"Average Reward": ar}, step=i)

                self.reward = 0

            import torch

            if i % 2_000 == 0:
            #    ma_obs_to_track = torch.tensor(
            #        [[10.0, 10.0, 10.0, 10.0]]
            #    )

            #    sa_obs_to_track = torch.tensor(
            #        [[10.0]]
            #    )

                sa_obs_to_track = torch.tensor([[10.0, 0]])
                #ma_q_values = self.impact_measurers[0].get_q_values(ma_obs_to_track)
                #sa_q_values = self.agent_policies[0].get_q_values(sa_obs_to_track)
                #q_ma = self.impact_measurers[0].get_q_values(ma_obs_to_track)[0]
                q_sa = self.agent_policies[0].get_q_values(sa_obs_to_track)[0]
                #print(q_ma, q_sa)
                #wandb.log({"ma0": q_ma[0]}, step=i)
                #wandb.log({"ma1": q_ma[1]}, step=i)
                #wandb.log({"ma2": q_ma[2]}, step=i)
                #wandb.log({"ma3": q_ma[3]}, step=i)

                wandb.log({"sa0": q_sa[0]}, step=i)
                wandb.log({"sa1": q_sa[1]}, step=i)
                wandb.log({"sa2": q_sa[2]}, step=i)


                
               # print(
               #     sa_q_values[0]
               # )
               