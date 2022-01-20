

import hydra
from critic import Critic
import mpu

from environement import Environment
from runner import Runner
from numpy import load
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

def get_config():
    hydra.initialize(config_path="configs", job_name="run")
    cfg = hydra.compose(config_name="config")
    transform = hydra.utils.instantiate(cfg, _convert_="partial")
    return transform




def main():
    configg = get_config()

    
    environment = Environment()
    environment.registerenv()
    


    
    ppo = mpu.io.read('/Users/julian/ray_results/PPO/PPO_prison_21269_00000_0_2022-01-18_09-32-10/ppo.pickle')
    dqn = mpu.io.read('/Users/julian/ray_results/DQN/DQN_prison_aad07_00000_0_2022-01-18_09-21-42/dqn.pickle')
    #critic = Critic(environment)
    #critic.feedDQN(data)
    
    runner = Runner(configg, environment)
    runner.run()


if __name__ == "__main__":
    main()
    print("Done!")
