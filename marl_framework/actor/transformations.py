import logging
from typing import Dict, List

import numpy as np
import cv2
import torch

from agent.state_space import AgentStateSpace
from utils.state import get_w_entropy_map

logger = logging.getLogger(__name__)


def get_network_input(
    local_information,
    fused_local_map,
    simulated_map,
    agent_id,
    t,
    params,
    batch_memory,
    agent_state_space,
):
    total_budget = params["experiment"]["constraints"]["budget"]
    spacing = params["experiment"]["constraints"]["spacing"]

    position_map, position = get_position_feature_map(
        local_information, agent_id, agent_state_space, params
    )

    w_entropy_map, weightings_map, entropy_map, local_w_entropy_map, prob_map = get_w_entropy_map(
        local_information[agent_id]["footprint_img"],
        fused_local_map,
        simulated_map,
        "actor",
        agent_state_space,
    )

    budget_map = get_budget_feature_map(total_budget - t, position_map, params)
    altitude_map = get_altitude_map(
        local_information, agent_id, agent_state_space, position_map, spacing
    )
    agent_id_map = get_agent_id_map(agent_id, position_map, params)
    footprint_map = get_footprint_map(local_information, agent_id, agent_state_space, t)
    observation_map = torch.tensor(
        np.dstack(
            [
                budget_map,
                agent_id_map,
                position_map,
                w_entropy_map,
                local_w_entropy_map,
                prob_map,
                footprint_map,
            ]
        )
    )

    return observation_map


def get_footprint_map(local_information, agent_id, agent_state_space, t):
    footprint_map = local_information[agent_id]["map2communicate"].copy()
    footprint_map[footprint_map < 0.49] = 1
    footprint_map[footprint_map > 0.51] = 1
    for agent in local_information:
        if agent == agent_id:
            pass
        else:
            new_map = local_information[agent]["map2communicate"].copy()
            footprint_map[new_map < 0.49] = 0
            footprint_map[new_map > 0.51] = 0
    new_map = local_information[agent_id]["map2communicate"].copy()
    footprint_map[new_map < 0.49] = 1
    footprint_map[new_map > 0.51] = 1

    footprint_map = cv2.resize(
        footprint_map,
        (agent_state_space.space_dim[1], agent_state_space.space_dim[0]),
        interpolation=cv2.INTER_AREA,
    )

    return footprint_map


def get_agent_id_map(agent_id, position_map, params: Dict) -> np.array:
    n_agents = params["experiment"]["missions"]["n_agents"]
    return np.ones_like(position_map) * ((agent_id + 1) / n_agents)


def get_budget_feature_map(remaining_budget, position_map, params: Dict):
    total_budget = params["experiment"]["constraints"]["budget"]
    return np.ones_like(position_map) * (remaining_budget / total_budget)


def get_altitude_map(
    local_information, agent_id, agent_state_space, position_map, spacing
):
    altitude = None
    for idx in local_information:
        if idx == agent_id:
            own_position = local_information[idx]["position"]
            altitude = own_position[2]
            break
    return np.ones_like(position_map) * (
        (altitude // spacing) / agent_state_space.space_z_dim
    )


def get_position_feature_map(
    local_information: Dict,
    agent_id: int,
    agent_state_space: AgentStateSpace,
    params: Dict,
) -> np.array:
    n_agents = params["experiment"]["missions"]["n_agents"]

    own_position = None
    relative_indexes = None
    other_positions = []
    position_map = np.ones(
        (agent_state_space.space_x_dim, agent_state_space.space_y_dim)
    )

    for idx in local_information:
        if idx == agent_id:
            own_position = agent_state_space.position_to_index(
                local_information[idx]["position"]
            )
            relative_indexes = [
                [
                    idx,
                    [5, 5, (own_position[2] + 1) / (agent_state_space.space_z_dim + 1)],
                ]
            ]
            if own_position[0] < 5:
                position_map[0 : 5 - own_position[0], :] = 0
            if own_position[1] < 5:
                position_map[:, 0 : 5 - own_position[1]] = 0
            if own_position[0] > 5:
                position_map[
                    agent_state_space.space_x_dim - 1 - (own_position[0] - 6) :, :
                ] = 0
            if own_position[1] > 5:
                position_map[
                    :, agent_state_space.space_y_dim - 1 - (own_position[1] - 6) :
                ] = 0
        else:
            other_position_idx = agent_state_space.position_to_index(
                local_information[idx]["position"]
            )
            other_positions.append([idx, other_position_idx])

    for other_position in other_positions:
        relative_indexes.append(
            [
                other_position[0],
                [
                    other_position[1][0] - own_position[0] + 5,
                    other_position[1][1] - own_position[1] + 5,
                    (other_position[1][2] + 1) / (agent_state_space.space_z_dim + 1),
                ],
            ]
        )
    for relative_index in relative_indexes:
        if (
            relative_index[1][0] >= 0
            and relative_index[1][0] < agent_state_space.space_x_dim
            and relative_index[1][1] >= 0
            and relative_index[1][1] < agent_state_space.space_x_dim
        ):
            position_map[
                int(relative_index[1][0]), int(relative_index[1][1])
            ] = relative_index[1][2]

    return position_map, own_position


def get_previous_action_map(agent_id, batch_memory, t, params: Dict):
    n_agents = params["experiment"]["missions"]["n_agents"]
    n_actions = params["experiment"]["constraints"]["num_actions"]

    if t == 0:
        action = 0
    else:
        for agent in range(n_agents):
            if agent_id == agent_id:
                action = np.float64(batch_memory.get(-1, agent_id, "action").item()) / (
                    n_actions - 1
                )
    return action
