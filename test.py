from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from pettingzoo.butterfly import pistonball_v5
import supersuit as ss
import torch
from torch import nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray import shutdown


class CNNModelV2(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space,
                              num_outputs, *args, **kwargs)
        nn.Module.__init__(self)
        self.model = nn.Sequential(
            nn.Conv2d(
                3,
                32,
                [8, 8],
                stride=(4, 4)),
            nn.ReLU(),
            nn.Conv2d(
                32,
                64,
                [4, 4],
                stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(
                64,
                64,
                [3, 3],
                stride=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(3136, 512)),
            nn.ReLU(),
        )
        self.policy_fn = nn.Linear(512, num_outputs)
        self.value_fn = nn.Linear(512, 1)

    def forward(self, input_dict, state, seq_lens):
        model_out = self.model(input_dict["obs"].permute(0, 3, 1, 2))
        self._value_out = self.value_fn(model_out)
        return self.policy_fn(model_out), state

    def value_function(self):
        return self._value_out.flatten()


def env_creator(args):
    env = prison_v3.env(vector_observation=True, continuous=False, synchronized_start=False,
                        identical_aliens=False, max_cycles=150, num_floors=2, random_aliens=False)

    #env = ss.color_reduction_v0(env, mode='B')
    #env = ss.dtype_v0(env, 'float32')
    #env = ss.resize_v0(env, x_size=84, y_size=84)
    #env = ss.frame_stack_v1(env, 3)
    #env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    return env


if __name__ == "__main__":
    shutdown()

    env_name = "pistonball_v5"

    register_env(env_name, lambda config: ParallelPettingZooEnv(
        env_creator(config)))

    test_env = ParallelPettingZooEnv(env_creator({}))
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    ModelCatalog.register_custom_model("CNNModelV2", CNNModelV2)

    def gen_policy(i):
        config = {
            "model": {
                "custom_model": "CNNModelV2",
            },
            "gamma": 0.99,
        }
        return (None, obs_space, act_space, config)

    policies = {"policy_0": gen_policy(0)}

    policy_ids = list(policies.keys())

    tune.run(
        "DQN",
        name="PPO",
        stop={"timesteps_total": 5000000},
        checkpoint_freq=10,
        local_dir="~/ray_results/"+env_name,
        config={
            # Environment specific
            "env": env_name,
            # General
            "log_level": "ERROR",
            "framework": "torch",
            "num_gpus": 0,
            "num_workers": 4,
            "num_envs_per_worker": 1,
            "compress_observations": False,
            "batch_mode": 'truncate_episodes',

            # 'use_critic': True,


            # Method specific
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": (
                    lambda agent_id: policy_ids[0]),
            },
        },
    )
