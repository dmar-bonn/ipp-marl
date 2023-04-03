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

    other_actions_map = get_other_actions_map(batch_memory, global_information, agent_id, params, agent_state_space)
    footprint_map = get_footprint_map(agent_state_space, global_information)
    actor_network_input = batch_memory.get(-1, agent_id, "observation")

    # COMA input
    total_input_map = torch.tensor(np.dstack((actor_network_input.cpu().detach().numpy(),
                                              np.expand_dims(position_map, 2), w_entropy_map, prob_map,
                                              np.expand_dims(footprint_map, 2) , np.expand_dims(other_actions_map, 2)))).float()


    # IAC input
    # total_input_map = actor_network_input.float()

    # np.dstack(
    #     [
    #         other_actions_map,
    #         position_map,
    #         w_entropy_map,
    #         actor_network_input,
    #         #         previous_action_map,
    #         #     ]
    #     ]
    # ).float())
    # ).float(),
    # torch.tensor(w_entropy_map)
    # torch.tensor(accumulated_map_knowledge),
    # torch.tensor(simulated_map),
    # t
    # ]

    # plt.imshow(total_input_map[:, :, 0])
    # plt.title(f"t={t}")
    # plt.clim(0, 1)
    # plt.savefig(f"/home/penguin2/Documents/plots/critic_input_0_t{t}_agent{agent_id}.png")
    #
    # plt.imshow(total_input_map[:, :, 1])
    # plt.title(f"t={t}")
    # plt.clim(0, 1)
    # plt.savefig(f"/home/penguin2/Documents/plots/critic_input_1_t{t}_agent{agent_id}.png")
    #
    # plt.imshow(total_input_map[:, :, 2])
    # plt.title(f"t={t}")
    # plt.clim(0, 1)
    # plt.savefig(f"/home/penguin2/Documents/plots/critic_input_2_t{t}_agent{agent_id}.png")
    #
    # plt.imshow(total_input_map[:, :, 3])
    # plt.title(f"t={t}")
    # plt.clim(0, 1)
    # plt.savefig(f"/home/penguin2/Documents/plots/critic_input_3_t{t}_agent{agent_id}.png")
    #
    # plt.imshow(total_input_map[:, :, 4])
    # plt.title(f"t={t}")
    # plt.clim(0, 1)
    # plt.savefig(f"/home/penguin2/Documents/plots/critic_input_4_t{t}_agent{agent_id}.png")
    #
    # plt.imshow(total_input_map[:, :, 5])
    # plt.title(f"t={t}")
    # plt.clim(0, 1)
    # plt.savefig(f"/home/penguin2/Documents/plots/critic_input_5_t{t}_agent{agent_id}.png")
    #
    # plt.imshow(total_input_map[:, :, 6])
    # plt.title(f"t={t}")
    # plt.clim(0, 1)
    # plt.savefig(f"/home/penguin2/Documents/plots/critic_input_6_t{t}_agent{agent_id}.png")
    #
    # plt.imshow(total_input_map[:, :, 7])
    # plt.title(f"t={t}")
    # plt.clim(0, 1)
    # plt.savefig(f"/home/penguin2/Documents/plots/critic_input_7_t{t}_agent{agent_id}.png")
    #
    # plt.imshow(total_input_map[:, :, 8])
    # plt.title(f"t={t}")
    # plt.clim(0, 1)
    # plt.savefig(f"/home/penguin2/Documents/plots/critic_input_8_t{t}_agent{agent_id}.png")
    #
    # plt.imshow(total_input_map[:, :, 9])
    # plt.title(f"t={t}")
    # plt.clim(0, 1)
    # plt.savefig(f"/home/penguin2/Documents/plots/critic_input_9_t{t}_agent{agent_id}.png")
    #
    # plt.imshow(total_input_map[:, :, 10])
    # plt.title(f"t={t}")
    # plt.clim(0, 1)
    # plt.savefig(f"/home/penguin2/Documents/plots/critic_input_10_t{t}_agent{agent_id}.png")
    #
    # plt.imshow(total_input_map[:, :, 11])
    # plt.title(f"t={t}")
    # plt.clim(0, 1)
    # plt.savefig(f"/home/penguin2/Documents/plots/critic_input_11_t{t}_agent{agent_id}.png")

    batch_memory.insert(-1, agent_id, q_state=total_input_map)
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
        position_map[position_idx[0], position_idx[1]] = (position_idx[2] + 1) / agent_state_space.space_z_dim  #

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


def get_other_actions_map(batch_memory, global_information, agent_id, params: Dict, agent_state_space):
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
                action_map[position_idx[0], position_idx[1]] = (batch_memory.get(-1, agent,
                                                                                 "action").item() + 1) / n_actions
            except:
                logger.info(f"position bug")
    return action_map


