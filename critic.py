from ray.rllib.agents.dqn.dqn_torch_policy import DQNTorchPolicy


class Critic(object):

    def test(self):
        return "z"

    def __init__(self, environment):

        environment.env.reset()


        self.os = environment.env.observation_space(environment.env.agents[0])
        self.acs = environment.env.action_space(environment.env.agents[0])
        self.dqn = DQNTorchPolicy(self.os,  self.acs , {"num_workers": 1, "num_gpus": 0})
    
    def feedDQN(self, batch):
        self.dqn.learn_on_batch(batch)
        pass

