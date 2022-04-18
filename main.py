import hydra
import mpu
import numpy as np
import torch

import src.utils_folder.array_utils as ar_ut
import src.utils_folder.config_utils as cf_ut
from environement import Environment
from impact_approximator import ImpactApproximator
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

    single_postprocessed_batch = mpu.io.read("./logs/postprocessed_batch.pickle")
    postprocessed_batch_too_long = mpu.io.read(
        "./logs/postprocessed_batch_too_long.pickle"
    )
    ma_train_batch = mpu.io.read("./logs/multi_agent_learn_batch.pickle")
    # critic = Critic(environment)
    # critic.feedDQN(new_batch_format)

    cf_ut.enrich_config_file(config)

    impact_approximator_0 = ImpactApproximator(config, 0)

    obs_to_track = torch.tensor([[10.0, 10.0, 10.0, 10.0]])
    actions_to_track = ar_ut.actions_to_nodes(np.array([[0, 0, 0, 0]]), (3, 3, 3, 3))
    q_values_to_track = impact_approximator_0.get_q_values(obs_to_track).index_select(
        1, torch.tensor(actions_to_track)
    )

    runner = Runner(config)
    runner.run()


if __name__ == "__main__":
    main()
    print("Done!")
