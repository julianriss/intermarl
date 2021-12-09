from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
import hydra
from typing import Dict

class Runner(object):
    def __init__(self, config: Dict) -> None:
        self.config = config
        pass

    def run(self):
        tune.run("PPO", config=self.config)
        print("This is where stuff happens")