from typing import Dict, Tuple


class BaseDataHandler(object):
    def __init__(self, config: Dict) -> None:
        self.config = config

    def transform_postprocessed_batch(self, experience: Tuple):
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
