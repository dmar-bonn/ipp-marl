import copy
from collections import namedtuple
from sklearn.metrics import f1_score, classification_report
import numpy as np
import torch
from matplotlib import pyplot as plt

TransitionCOMA = namedtuple(
    "TransitionPPO",
    (
        "state",
        "observation",
        "action",
        "mask",
        "reward",
        "done",
        "td_target",
        "discounted_return",
    ),
)


def normalize(x: np.array) -> np.array:
    min_value = np.min(x)
    max_value = np.max(x)
    if min_value == max_value:
        return x / max_value
    return (x - min_value) / (max_value - min_value)


def compute_euclidean_distance(start: np.array, goal: np.array) -> float:
    return np.linalg.norm(start - goal, ord=2)


def clip_gradients(network: torch.nn.Module, gradient_norm: float) -> torch.nn.Module:
    for param in network.parameters():
        if param.grad is not None:
            param.grad.data.clamp_(-gradient_norm, gradient_norm)

    return network


def get_wrmse(map_state, map_simulation):

    # target = copy.deepcopy(map_simulation)
    # target[target > 0.501] = 1
    # target[target < 0.499] = 0

    class_weighting = [0, 1]

    weightings = map_simulation.copy()
    weightings[np.round(weightings, 2) == 0] = class_weighting[0]
    weightings[np.round(weightings, 2) == 1] = class_weighting[1]
    weightings[np.round(weightings, 2) == 0.5] = 0.5

    # rmse = np.sqrt((map_state-map_simulation)**2)
    rmse_map = (map_state-map_simulation)**2
    map_unique, map_counts = np.unique(map_simulation, return_counts=True)
    target_counts = map_counts[-1]
    rmse_masked = rmse_map.copy()
    rmse_masked[map_simulation == 0] = 0
    wrmse = np.sqrt(np.sum(rmse_masked) / target_counts)



    rounded_map_state = map_state.copy()
    rounded_map_state[rounded_map_state > 0.5] = 1
    rounded_map_state[rounded_map_state <= 0.5] = 0

    f1 = f1_score(map_simulation.flatten(), rounded_map_state.flatten(), average=None)

    # w_f1 = weightings * f1
    # w_f1 = np.sum(w_f1)
    # w_f1 = w_f1 / target_counts

    w_f1 = f1[1]

    return w_f1 # wrmse


def get_fixed_footprint_coordinates(footprint, footprint_clipped):
    yu = 0
    yd = footprint[1] - footprint[0]
    xl = 0
    xr = footprint[3] - footprint[2]

    if footprint_clipped[0] > footprint[0]:
        yu = (footprint[1] - footprint[0]) - (footprint_clipped[1] - footprint_clipped[0])
    if footprint_clipped[1] < footprint[1]:
        yd = footprint_clipped[1] - footprint_clipped[0]
    if footprint_clipped[3] < footprint[3]:
        xr = footprint_clipped[3] - footprint_clipped[2]
    if footprint_clipped[2] > footprint[2]:
        xl = (footprint[3] - footprint[2]) - (footprint_clipped[3] - footprint_clipped[2])

    return int(yu), int(yd), int(xl), int(xr)





