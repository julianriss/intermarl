from ntpath import join
import numpy as np
from typing import List, Tuple
import torch
import networkx as nx
from tqdm import tqdm


def build_response_graph(action_space: Tuple[int]) -> nx.DiGraph:
    response_graph = nx.DiGraph()
    max_flat_index = calc_maximimum_flat_index(action_space)
    response_graph.add_nodes_from([node for node in range(max_flat_index)])
    print("Building the response graph! This step is costly, so be patient.")
    for node in tqdm(range(max_flat_index)):
        neighbors = get_neighbors_of_node(node, action_space)
        for neighbor in neighbors:
            response_graph.add_edge(node, neighbor)
    return response_graph


def calc_maximimum_flat_index(array_shape: Tuple[int]) -> int:
    return np.prod(array_shape)


def get_neighbors_of_node(node: int, action_space: Tuple[int]) -> List[int]:
    """Enumerates all neighbors of a given node in the action-graph.
    Another action is a neighbor iff only one agent deviates in its action.
    Args:
        node: a flat index of a joint-action
    Returns:
        neighbors: a list of indices that represent single deviating joint-actions from the given node
    """
    base_actions = node_to_actions(node, action_space)
    neighbors = []
    for deviating_agent, base_action in enumerate(base_actions):
        for action in range(action_space[deviating_agent]):
            if action != base_action:
                deviating_actions = replace_single_action_in_actions(
                    base_actions, action, deviating_agent
                )
                neighbors.append(actions_to_nodes(deviating_actions, action_space))
    return neighbors


def node_to_actions(node: int, array_shape: Tuple[int]) -> Tuple[int]:
    return np.unravel_index(node, array_shape)


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


def replace_single_action_in_actions(
    base_actions: Tuple[int], deviating_action: int, deviating_agent: int
) -> Tuple[int]:
    changed_actions = list(base_actions)
    changed_actions[deviating_agent] = deviating_action
    return changed_actions


def get_neighbors_to_actions(batched_actions: np.ndarray, action_space: Tuple[int]) -> List[List[int]]:
    """Computes all single deviating neighbors to a given array of joint-actions.

    Args:
        batched_actions (np.ndarray): shape(batch_size, num_agents)
        action_space (Tuple[int]): shape(num_agents) where the values are the action-space sizes

    Returns:
        List[List[int]]: For every given action, it contains a list of length num_agents, 
        where each there contains a list of num_neighbors (=action_space_size for this agent)
    """
    return [get_all_neighbors_to_joint_action(action_space, actions) for actions in batched_actions]

def get_all_neighbors_to_joint_action(action_space, actions):
    return [get_neighbors_to_deviating_agent(action_space, actions, deviating_agent) for deviating_agent in range(actions.shape[0])]

def get_neighbors_to_deviating_agent(action_space, actions, deviating_agent):
    return [replace_single_action_in_actions(
                        actions, deviating_action, deviating_agent
                    ) for deviating_action in range(action_space[deviating_agent])]


def main():
    action_space = (3, 3, 3, 3)
    nodes_to_actions(np.array([1, 2]), action_space)
    actions_to_nodes(np.array([[1, 2, 0, 1], [0, 1, 2, 1]]), action_space)
    neighbors = get_neighbors_to_actions(np.array([[1, 2, 0, 1], [0, 1, 2, 1]]), action_space)


    q_values = torch.rand(4, 81)


if __name__ == "__main__":
    main()
    print("Done!")