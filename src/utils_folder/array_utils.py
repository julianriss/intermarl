from ntpath import join
import numpy as np
from typing import List, Tuple
import torch


def nodes_to_actions(nodes: np.ndarray, array_shape: Tuple[int]) -> np.ndarray:
    """This is the array version of node_to_actions.
    Args:
        nodes (np.ndarray): shape=(x, )
        array_shape (Tuple[int]): the multi-index to ravel into

    Returns:
        np.ndarray: shape(x, len(array_shape))
    """
    return np.array(np.unravel_index(nodes, array_shape)).T


def actions_to_nodes(actions: np.ndarray, array_shape: Tuple[int]) -> np.ndarray:
    """Array of shape (batch_size, num_agents) is turned into a raveled multi-index by array_shape shape (y,)
    Args:
        actions (np.ndarray): shape (batch_size, num_agents)
        array_shape (Tuple[int]): shape(batch_size, )

    Returns:
        np.ndarray: shape(batch_size, )
    """
    return np.ravel_multi_index(actions.T, array_shape)


def get_neighbors_to_actions(batched_actions: np.ndarray, action_space: Tuple[int]) -> List[np.ndarray]:
    """Computes all single deviating neighbors to a given array of joint-actions.

    Args:
        batched_actions (np.ndarray): shape(batch_size, num_agents)
        action_space (Tuple[int]): shape(num_agents) where the values are the action-space sizes
    Returns:
        List[np.ndarray]: returns a list of arrays #agents[array.shape=(#samples, #action_space(agent))]
    """
    neighbors = []
    for deviating_agent in range(len(action_space)):
        neighbors_to_deviating_agent = np.empty((batched_actions.shape[0], action_space[deviating_agent]), dtype=int)
        for deviating_action in range(action_space[deviating_agent]):
            copied_actions = batched_actions.copy()
            copied_actions[:, deviating_agent] = deviating_action
            neighbors_to_deviating_agent[:, deviating_action] = actions_to_nodes(copied_actions, action_space)
        neighbors.append(neighbors_to_deviating_agent)
    return neighbors


def get_impact_samples_for_batch(obs: torch.Tensor, actions: torch.Tensor, action_space: Tuple[int]) -> torch.Tensor:
    """Computes the impact samples for a given batch.
    Args:
        obs (torch.Tensor): batch of observations (shape=(#samples, #features))
        actions (torch.Tensor): batch of joint actions
        action_space (Tuple[int]): action space (shape=(#agents, ))

    Returns:
        torch.Tensor: shape=(#samples, #agents)
    """
    q_values = torch.arange(2*3*4*5*6).view(2,3*4*5*6)  # TODO: Calculate q-values
    decoded_actions = decode_discrete_actions(actions, action_space, ret_as_joint_actions=True)
    neighbors = get_neighbors_to_actions(decoded_actions, action_space)
    return get_impact_samples_from_q_values(q_values, neighbors)


def get_impact_samples_from_q_values(q_values: torch.Tensor, neighbors: List[np.ndarray]) -> torch.Tensor:
    impact_samples = torch.empty(q_values.shape[0], len(neighbors))
    for agent, deviating_actions_for_agent in enumerate(neighbors):
        slice_tensor = torch.LongTensor(deviating_actions_for_agent)
        sliced_q_values = torch.gather(q_values, dim=1, index=slice_tensor)
        impact_samples[:, agent] = (sliced_q_values.max(dim=1)[0] - sliced_q_values.min(dim=1)[0])
    return impact_samples


def decode_discrete_actions(encoded_actions: torch.Tensor, action_space: Tuple[int], ret_as_joint_actions:bool=False) -> np.ndarray:
    """Inverts the transform method from Discrete gym space encoder.
    Args:
        encoded_actions (torch.Tensor): tensor of shape=(#samples, size action_space)
        action_space (Tuple[int]): tuple of indiviual action space sizes shape=(#agents, )
        ret_as_joint_actions (bool, optional): If True return actions a joint-action. Defaults to False.

    Returns:
        np.ndarray: shape=(#samples, ) or shape=(#samples, #agents)
    """
    decoded_actions = encoded_actions.max(dim=1)[1]
    decoded_actions = decoded_actions.detach().cpu().numpy()
    if ret_as_joint_actions:
        decoded_actions = nodes_to_actions(decoded_actions, action_space)
    return decoded_actions

def main():
    action_space = (3, 4, 5, 6)
    nodes_to_actions(np.array([1, 2]), action_space)
    actions_to_nodes(np.array([[1, 2, 0, 1], [0, 1, 2, 1]]), action_space)
    neighbors = get_neighbors_to_actions(np.array([[2, 2, 0, 1], [0, 0, 2, 1]]),
     action_space)
    
    from ray.rllib.models import ModelCatalog
    from gym.spaces import MultiDiscrete, Discrete, Box
    action_encoder = ModelCatalog.get_preprocessor_for_space(Discrete(3*4*5*6))
    encoded_actions = torch.tensor([action_encoder.transform(i) for i in [34,  110]])
    decoded_actions = decode_discrete_actions(encoded_actions, action_space, True)

    impact_samples = get_impact_samples_for_batch(None, encoded_actions,
     action_space)
    
    

    print('test')


if __name__ == "__main__":
    main()
    print("Done!")