

import hydra
from critic import Critic


from environement import Environment
from runner import Runner

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
    


    
    

    
    runner = Runner(configg, environment)

    runner.run()


if __name__ == "__main__":
    main()
    print("Done!")
