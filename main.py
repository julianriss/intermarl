from typing import Dict

import hydra
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer


class Runner(object):
    def __init__(self, config: Dict) -> None:
        pass

    def run(self):
        print("This is where stuff happens")


def get_config():
    hydra.initialize(config_path="configs", job_name="run")
    cfg = hydra.compose(config_name="config")
    return cfg


def main():
    config = get_config()
    runner = Runner(config)
    runner.run()


if __name__ == "__main__":
    main()
    print("Done!")
