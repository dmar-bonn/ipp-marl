import logging
from typing import Dict

import numpy as np
import torch
import cv2
from matplotlib import pyplot as plt

from agent.state_space import AgentStateSpace
from utils.state import get_w_entropy_map

from agent.action_space import AgentActionSpace

logger = logging.getLogger(__name__)


def get_network_input(
    t: int,
    global_information: Dict,
    accumulated_map_knowledge,
    batch_memory,
    agent_id,
    simulated_map: np.array,
    params: Dict,
):
    agent_state_space = AgentStateSpace(params)
    agent_action_space = AgentActionSpace(params)
    network_type = params["networks"]["type"]
    n_agents = params["experiment"]["missions"]["n_agents"]

    position_map = get_position_feature_map(
        network_type,
        batch_memory,
        global_information,
        agent_state_space,
        agent_action_space,
        n_agents,
    )
    w_entropy_map, weightings_map, entropy_map, _, prob_map = get_w_entropy_map(
        None, accumulated_map_knowledge, simulated_map, "global", agent_state_space
    )

    other_actions_map = get_other_actions_map(
        batch_memory, global_information, agent_id, params, agent_state_space
    )
    footprint_map = get_footprint_map(agent_state_space, global_information)
    actor_network_input = batch_memory.get(-1, agent_id, "observation")

    # COMA input
    total_input_map = torch.tensor(
        np.dstack(
            (
                actor_network_input.cpu().detach().numpy(),
                np.expand_dims(position_map, 2),
                w_entropy_map,
                prob_map,
                np.expand_dims(footprint_map, 2),
                np.expand_dims(other_actions_map, 2),
            )
        )
    ).float()

    # IAC input
    # total_input_map = actor_network_input.float()

    batch_memory.insert(-1, agent_id, state=total_input_map)
    return total_input_map


def get_position_feature_map(
    network_type,
    batch_memory,
    global_information,
    agent_state_space,
    agent_action_space,
    n_agents,
):
    position_map = np.zeros(
        (agent_state_space.space_x_dim, agent_state_space.space_y_dim)
    )
    for agent in range(len(global_information)):
        position = global_information[agent]["position"]
        position_idx = agent_state_space.position_to_index(position)
        position_map[position_idx[0], position_idx[1]] = (
            position_idx[2] + 1
        ) / agent_state_space.space_z_dim  #

    return position_map


def get_footprint_map(agent_state_space, global_information):
    footprint_map = global_information[0]["map2communicate"].copy()
    footprint_map[footprint_map < 0.49] = 1
    footprint_map[footprint_map > 0.51] = 1
    for agent in global_information:
        if agent == 0:
            pass
        else:
            new_map = global_information[agent]["map2communicate"].copy()
            footprint_map[new_map < 0.49] = 1
            footprint_map[new_map > 0.51] = 1

    footprint_map = cv2.resize(
        footprint_map,
        (agent_state_space.space_dim[1], agent_state_space.space_dim[0]),
        interpolation=cv2.INTER_AREA,
    )
    return footprint_map


def get_other_actions_map(
    batch_memory, global_information, agent_id, params: Dict, agent_state_space
):
    n_actions = params["experiment"]["constraints"]["num_actions"]
    n_agents = params["experiment"]["missions"]["n_agents"]

    action_map = np.zeros(
        (agent_state_space.space_x_dim, agent_state_space.space_y_dim)
    )
    for agent in range(n_agents):
        if agent == agent_id:  #
            pass  #
        else:  #
            position = global_information[agent]["position"]
            position_idx = agent_state_space.position_to_index(position)
            try:
                action_map[position_idx[0], position_idx[1]] = (
                    batch_memory.get(-1, agent, "action").item() + 1
                ) / n_actions
            except:
                logger.info(f"position bug")
    return action_map


def get_previous_actions_map(t, batch_memory, params: Dict):
    n_actions = params["experiment"]["constraints"]["num_actions"]
    n_agents = params["experiment"]["missions"]["n_agents"]

    if t == 0:
        prev_actions_map = np.expand_dims(np.zeros(n_agents), axis=1)
    else:
        prev_actions_map = np.zeros(n_agents)
        for agent_id in range(n_agents):
            action = batch_memory.get(-2, agent_id, "action") / (n_actions - 1)
            prev_actions_map[agent_id] = action.cpu()
        prev_actions_map = np.expand_dims(prev_actions_map, axis=1)
    return prev_actions_map
