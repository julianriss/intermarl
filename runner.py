from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
import hydra
from typing import Dict, Iterable
from ray.tune import Callback
from ray.tune.trial import Trial
from ray.tune.ray_trial_executor import RayTrialExecutor
from typing import List

from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.tune import Callback


class MyCallback(DefaultCallbacks):
   def on_learn_on_batch(self, *, policy, train_batch, result, **kwargs):
        print("observation: ", train_batch["obs"], "actions: ", train_batch["actions"], "rewards: ", train_batch["rewards"], "agentindex: ", train_batch["agent_index"])
        pass
class Runner(object):
    def __init__(self, config: Dict) -> None:
        self.config = config
        pass

    def run(self):
        tune.run(self.config['algorithm'], checkpoint_freq=1, config={
            "framework": self.config['framework'],
            "env": self.config['env'],
            #"rollout_fragment_length": 40,
           #"train_batch_size": 800,
            "callbacks": MyCallback
           
           
        })
        print("This is where stuff happens")