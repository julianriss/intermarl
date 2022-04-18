import copy
from random import randrange
from typing import Dict, Tuple

import numpy as np
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch

import src.utils_folder.array_utils as ar_ut


class BaseDataHandler(object):
    def __init__(self, config: Dict) -> None:
        self.config = config

    def transform_postprocessed_batch(self, collected_batch):
        """Takes the data from all agents and transforms them into the right format to train with.
        This includes:
        - concatenating the observations from each agent
        - transforming the joint-action into a trainable format
        - appending the correct reward with the correct agent-index

        Args:
            experience (Tuple): 
        """
        raise NotImplementedError


class PrisonDataHandler(BaseDataHandler):
    """This one handles the data transformations for the prison environment.

    Args:
        BaseDataHandler (_type_): _description_
    """

    def __init__(self, config: Dict) -> None:
        super().__init__(config)

    def transform_postprocessed_batch(
        self, data, pb_structure, action_space_size, one_hot_encoding
    ):
        observations = np.array([])
        observations_next = np.array([])
        actions = np.array([], dtype=int)

        for i in range(
            self.config["rl_env"]["num_agents"]
        ):  # TODO: Stuff like this should be in an own method
            observations = np.append(  # TODO: Enrich the observations with the moving direction
                observations, data[i][SampleBatch.CUR_OBS][0][0]
            )
            observations_next = np.append(
                observations_next, data[i][SampleBatch.NEXT_OBS][0][0]
            )
            actions = np.append(actions, data[i][SampleBatch.ACTIONS][0])

        pb = copy.deepcopy(pb_structure)

        state_encoder = ModelCatalog.get_preprocessor_for_space(
            self.config["rl_env"]["observation_space"]
        )
        #

        # Erzeugung von validen Array mit Observations/Actions aller Agenten
        combined_obs = state_encoder.transform(observations)
        combined_next_obs = state_encoder.transform(observations_next)
        # Ursprünglich auch Actions aus .transorf. Daher kam das one hot encoding

        if one_hot_encoding == True:
            action_encoder = ModelCatalog.get_preprocessor_for_space(
                self.config["rl_env"]["action_space"]
            )

            combined_actions = action_encoder.transform(
                ar_ut.actions_to_nodes(actions, action_space_size)
            )

        else:
            combined_actions = ar_ut.actions_to_nodes(actions, action_space_size)
        # zusammengefasste Observations und Actions werden in kopierten postprocessed_batch geschrieben
        SampleBatch.__setitem__(pb, SampleBatch.CUR_OBS, combined_obs[np.newaxis, :])
        SampleBatch.__setitem__(
            pb, SampleBatch.NEXT_OBS, combined_next_obs[np.newaxis, :]
        )
        SampleBatch.__setitem__(pb, SampleBatch.ACTIONS, np.array([combined_actions]))
        rewardbatches = []
        # TODO: Stuff like this should be in an own method
        for i in range(self.config["rl_env"]["num_agents"]):
            rewardbatches.append(pb)
            SampleBatch.__setitem__(
                rewardbatches[i], SampleBatch.REWARDS, data[i][SampleBatch.REWARDS]
            )

        return rewardbatches
