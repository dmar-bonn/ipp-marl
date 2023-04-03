import numpy as np
import matplotlib.pyplot as plt
import torch

from marl_framework.agent.state_space import AgentStateSpace
from marl_framework.utils.state import get_w_entropy_map

from utils.utils import compute_euclidean_distance


def get_global_reward(
        last_map,
        next_map,
        mission_type,
        footprints,
        simulated_map: np.array,
        agent_state_space: AgentStateSpace,
        actions,
        agent_id,
        t,
        budget
):
    done = False
    reward = 0
    scale = 10  # 40
    offset = 0.17  # 1.05

    o_min = 0
    o_max = 0.02
    p_max = 1
    fp_factor = 1

    absolute_utility_reward, relative_utility_reward = get_utility_reward(last_map, next_map, simulated_map,
                                                                          agent_state_space)
    # print(f"absolute_utility_reward: {absolute_utility_reward}")
    absolute_reward = scale * absolute_utility_reward - offset
    relative_reward = 22 * relative_utility_reward - 0.5 # / cost_factor - 0.35       # 22, 0.5

    # if mission_type == "DeepQ":
    #     # footprint_penalty = get_footprint_penalty(footprints, agent_id, simulated_map, o_min, o_max, p_max)
    #     absolute_utility_reward, relative_utility_reward = get_utility_reward(last_map, next_map, simulated_map,
    #                                                                           agent_state_space)
    #     absolute_reward = scale * absolute_utility_reward - offset
    #     reward += 34 * relative_utility_reward - 0.25

    return done, relative_reward, absolute_reward   # done, relative_reward, absolute_reward


def get_collision_reward(next_positions, done):
    for agent1 in range(len(next_positions)):
        for agent2 in range(agent1):
            done = is_collided(
                next_positions[agent1],
                next_positions[agent2],
            )
            if done:
                break
        if done:
            break

    return done, -1 if done else 0


def get_utility_reward(
        state: np.array,
        state_: np.array,
        simulated_map: np.array,
        agent_state_space: AgentStateSpace,
):
    entropy_before = get_w_entropy_map(None, state, simulated_map, "reward", agent_state_space)[2]
    output = get_w_entropy_map(None, state_, simulated_map, "reward", agent_state_space)
    entropy_after = output[2]
    entropy_reduction = entropy_before - entropy_after

    absolute_reward = np.mean(output[1] * entropy_reduction)
    relative_reward = absolute_reward / (np.mean(output[1] * entropy_before))

    # plt.imshow(state)
    # plt.title(f"state before")
    # plt.clim(0, 1)
    # # plt.colorbar()
    # plt.savefig(f"/home/penguin2/Documents/plots/state_before.png")
    #
    # plt.imshow(state_)
    # plt.title(f"state after")
    # plt.clim(0, 1)
    # # plt.colorbar()
    # plt.savefig(f"/home/penguin2/Documents/plots/state_after.png")
    #
    # plt.imshow(entropy_before)
    # plt.title(f"entropy_before")
    # plt.clim(0, 1)
    # plt.colorbar()
    # plt.savefig(f"/home/penguin2/Documents/plots/entropy_before.png")
    #
    # plt.imshow(entropy_after)
    # plt.title(f"entropy_after")
    # plt.clim(0, 1)
    # # plt.colorbar()
    # plt.savefig(f"/home/penguin2/Documents/plots/entropy_after.png")
    #
    # plt.imshow(entropy_reduction)
    # plt.title(f"entropy_reduction")
    # plt.clim(0, 1)
    # # plt.colorbar()
    # plt.savefig(f"/home/penguin2/Documents/plots/entropy_reduction.png")
    #
    # plt.imshow(output[1])
    # plt.title(f"weightings")
    # plt.clim(0, 1)
    # # plt.colorbar()
    # plt.savefig(f"/home/penguin2/Documents/plots/weightings.png")
    #
    # plt.imshow(output[1] * entropy_reduction)
    # plt.title(f"weighted reduction")
    # plt.clim(0, 1)
    # # plt.colorbar()
    # plt.savefig(f"/home/penguin2/Documents/plots/weighted_reduction.png")

    # print(f"absolute_reward: {absolute_reward}")
    # print(f"relative_reward: {relative_reward}")

    return absolute_reward, relative_reward


def is_collided(next_position_1, next_position_2):
    if np.array_equal(next_position_1, next_position_2):
        return True
    return False


def get_footprint_penalty(footprints, agent_id, simulated_map, o_min, o_max, p_max):
    own_footprint = footprints[agent_id]
    overlap = []
    for fp in range(len(footprints)):
        if fp == agent_id:
            pass
        else:
            overlap.append(compute_overlap(own_footprint, footprints[fp], simulated_map))
    mean_overlap = sum(overlap) / len(overlap)
    if mean_overlap > o_max:
        return 0
    elif mean_overlap < o_min:
        return p_max
    else:
        return p_max - ((mean_overlap - o_min) / (o_max - o_min)) / p_max


def compute_overlap(footprint1, footprint2, simulated_map):
    yu = max(footprint1[0], footprint2[0])
    yd = min(footprint1[1], footprint2[1])
    xl = max(footprint1[2], footprint2[2])
    xr = min(footprint1[3], footprint2[3])

    if yu > yd:
        return 0
    if xl > xr:
        return 0
    return ((yd - yu + 1) * (xr - xl + 1)) / np.size(simulated_map)
