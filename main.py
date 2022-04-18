import hydra

import src.utils_folder.config_utils as cf_ut
from environement import Environment
from runner import Runner


def get_config():
    hydra.initialize(config_path="configs", job_name="run")
    cfg = hydra.compose(config_name="config")
    transform = hydra.utils.instantiate(cfg, _convert_="partial")
    return transform


def main():
    config = get_config()

    environment = Environment()
    environment.registerenv()

    cf_ut.enrich_config_file(config)
    runner = Runner(config)
    runner.run()


if __name__ == "__main__":
    main()
    print("Done!")
