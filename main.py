

import hydra

from environement import Environment
from runner import Runner
import src.utils_folder.config_utils as cf_ut

def get_config():
    hydra.initialize(config_path="configs", job_name="run")
    cfg = hydra.compose(config_name="config")
    transform = hydra.utils.instantiate(cfg, _convert_="partial")
    return transform




def main():
    config = get_config()

    
    environment = Environment()
    environment.registerenv()
    

    # path = '/Users/julian/ray_results/'
    #path = './logs/'
    
    #ppo = mpu.io.read(path + 'PPO_prison_21269_00000_0_2022-01-18_09-32-10/ppo.pickle')
    #dqn = mpu.io.read(path + 'DQN_prison_aad07_00000_0_2022-01-18_09-21-42/dqn.pickle')
    #dqnsamplebatch = mpu.io.read(path + 'DQN/DQN_prison_48052_00000_0_2022-01-20_16-11-45/dqnsamplebatch.pickle')
    #postprocessed = mpu.io.read(path + 'postprocessed.pickle')
    #new_batch_format = mpu.io.read(path + 'new_batch_format.pickle')
    # original_batches = mpu.io.read(path + 'original_batches.pickle')
    #ub = mpu.io.read(path + "ub.pickle")


    #learnonbatch = mpu.io.read(path + 'learn_on_batch.pickle')
    #critic = Critic(environment)
    #critic.feedDQN(new_batch_format)

    cf_ut.enrich_config_file(config)
    
    runner = Runner(config)
    runner.run()


if __name__ == "__main__":
    main()
    print("Done!")
