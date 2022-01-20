from ray.rllib.agents.dqn.dqn_torch_policy import DQNTorchPolicy


class Critic(object):

    def test(self):
        return "z"

    def __init__(self, environment):

        environment.env.reset()


        self.os = environment.env.observation_space(environment.env.agents[0])
        self.acs = environment.env.action_space(environment.env.agents[0])
        self.dqn = DQNTorchPolicy(self.os,  self.acs , {})
    
    def feedDQN(self, batch):
        
        #print(self.dqn.get_weights())
        self.dqn.learn_on_batch(batch)
        print("learn on batch")
        pass

