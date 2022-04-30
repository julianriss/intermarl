import numpy as np
from pettingzoo.butterfly.prison.prison import raw_env
from pettingzoo.utils import wrappers


def init_custom_prison_env(**kwargs):
    env = PrisonAddObs(**kwargs)

    #if env.continuous:
    #    env = wrappers.ClipOutOfBoundsWrapper(env)
    #else:
    #    env = wrappers.AssertOutOfBoundsWrapper(env)

    #env = wrappers.OrderEnforcingWrapper(env)
  
    return env


class PrisonAddObs(raw_env):
    def __init__(
        self,
        continuous=False,
        vector_observation=False,
        max_cycles=150,
        num_floors=4,
        synchronized_start=False,
        identical_aliens=False,
        random_aliens=False,
    ):
        super().__init__(
            continuous,
            vector_observation,
            max_cycles,
            num_floors,
            synchronized_start,
            identical_aliens,
            random_aliens,
        )
        self.reached_left_side = self._init_reached_left_side()

    def _init_reached_left_side(self) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0, 0.0])

    def reset(self):
        self.reached_left_side = self._init_reached_left_side()
        return super().reset()
    

    def last(self, agent_id):
        obs, reward, done, info = super().last()
        
        if(obs == 0.0):
            self.reached_left_side[agent_id] = 1.0
        if(obs == 300.0):
            #print("wll right")
            self.reached_left_side[agent_id] = 0.0


        #print(self.reached_left_side)
        # TODO: adapt obs:
        """ 
       1. last() is called after every step, so you can assume that it is coherent in time
       2. look at the obs for each agent (you need to somehow get the information which agents is currently moving)
       3. if he touched the left wall, set self.reached_left_side to 1.0 for this agent
       4. if he touches the right wall, set it to 0.0
       5. add reached_left_side to the observation array
       6. adapt the dimensions in the env_utils (so in the config)
       7. see if everything runs through
       """
        obs = np.array([float(obs), self.reached_left_side[agent_id]])
        return obs, reward, done, info
