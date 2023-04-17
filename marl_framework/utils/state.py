import copy
import time
from typing import List

import cv2
import numpy as np
import torch

from marl_framework.agent.communication_log import CommunicationLog
from marl_framework.agent.state_space import AgentStateSpace
from matplotlib import pyplot as plt


def get_w_entropy_map(
    map_footprint: np.array,
    local_map: np.array,
    simulated_map: np.array,
    observability: str,
    agent_state_space: AgentStateSpace,
):
    if observability != "reward" and observability != "eval":
        grid_map = cv2.resize(
            local_map,
            (agent_state_space.space_dim[1], agent_state_space.space_dim[0]),
            interpolation=cv2.INTER_AREA,
        )
        if observability == "actor":
            try:
                map_footprint = cv2.resize(
                    map_footprint,
                    (agent_state_space.space_dim[1], agent_state_space.space_dim[0]),
                    interpolation=cv2.INTER_AREA,
                )
            except:
                print(f"map_footprint: {map_footprint}")

        simulated_map = cv2.resize(
            simulated_map,
            (agent_state_space.space_dim[1], agent_state_space.space_dim[0]),
            interpolation=cv2.INTER_AREA,
        )

    else:
        grid_map = local_map.copy()

    feature_map = calculate_w_entropy(
        grid_map, map_footprint, simulated_map, observability, agent_state_space
    )

    return feature_map


def calculate_w_entropy(
    grid_map: np.array,
    map_footprint: np.array,
    simulated_map: np.array,
    observability: str,
    agent_state_space: AgentStateSpace,
):
    class_weighting = [0, 1]

    if observability == "eval":  # use ground truth for evaluation
        target = copy.deepcopy(simulated_map)
    else:
        target = copy.deepcopy(grid_map)

    target[target > 0.501] = 1  # map is binary
    target[target < 0.499] = 0

    weightings = target.copy()
    weightings[np.round(weightings, 2) == 0] = class_weighting[0]
    weightings[np.round(weightings, 2) == 1] = class_weighting[1]
    weightings[np.round(weightings, 2) == 0.5] = 0.5

    se = get_shannon_entropy(grid_map)
    w_entropy_map = weightings * se

    # if observability != "reward":
    #     weightings = cv2.resize(
    #         weightings,
    #         (agent_state_space.space_dim[1], agent_state_space.space_dim[0]),
    #         interpolation=cv2.INTER_AREA,
    #     )
    #     se = cv2.resize(
    #         se,
    #         (agent_state_space.space_dim[1], agent_state_space.space_dim[0]),
    #         interpolation = cv2.INTER_AREA,
    #     )
    #     w_entropy_map = cv2.resize(
    #         w_entropy_map,
    #         (agent_state_space.space_dim[1], agent_state_space.space_dim[0]),
    #         interpolation = cv2.INTER_AREA,
    #     )

    if observability == "actor":

        target_footprint = copy.deepcopy(map_footprint)
        target_footprint[target_footprint > 0.501] = 1
        target_footprint[target_footprint < 0.499] = 0

        weightings_footprint = target_footprint.copy()
        weightings_footprint[np.round(weightings_footprint, 2) == 0] = class_weighting[
            0
        ]
        weightings_footprint[np.round(weightings_footprint, 2) == 1] = class_weighting[
            1
        ]
        weightings_footprint[np.round(weightings_footprint, 2) == 0.5] = 0.5

        se_footprint = get_shannon_entropy(map_footprint)
        w_entropy_map_footprint = weightings_footprint * se_footprint
    else:
        w_entropy_map_footprint = None

    return w_entropy_map, weightings, se, w_entropy_map_footprint, grid_map


def get_shannon_entropy(p):
    p[0.0001 > p] = 0.0001
    p[0.9999 < p] = 0.9999
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def append_global_information_to_timestep(
    data_timestep, mapping, communication_log: CommunicationLog
) -> List:
    data_timestep[0]["global_positions"] = communication_log.get_global_positions()
    other_agents_maps = {
        agent_id: agent_info for agent_id, agent_info in enumerate(data_timestep[1:])
    }
    data_timestep[0]["global_map"] = mapping.fuse_map(
        data_timestep[0]["local_map"], other_agents_maps
    )
    return data_timestep
