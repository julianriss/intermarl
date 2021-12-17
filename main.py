

import hydra


from environement import Environment
from runner import Runner



def get_config():
    hydra.initialize(config_path="configs", job_name="run")
    cfg = hydra.compose(config_name="config")
    transform = hydra.utils.instantiate(cfg, _convert_="partial")
    return transform




def main():
    config = get_config()

    runner = Runner(config)
    
    environment = Environment()
    environment.registerenv()
    

    runner.run()


if __name__ == "__main__":
    main()
    print("Done!")
