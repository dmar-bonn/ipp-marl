import copy
import json
import time
from typing import Dict

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from utils.plotting import plot_trajectories
from logger import setup_logger
import constants
from agent.action_space import AgentActionSpace
from agent.agent import Agent
from agent.state_space import AgentStateSpace
from batch_memory import BatchMemory
from coma_wrapper import COMAWrapper
from mapping.grid_maps import GridMap
from mapping.mappings import Mapping
from mapping.simulations import Simulation
from params import load_params
from sensors import Sensor
from sensors.cameras import Camera
from sensors.models import SensorModel
from sensors.models.sensor_models import AltitudeSensorModel
from utils.plotting import plot_trajectories
from utils.reward import get_global_reward
from utils.state import get_shannon_entropy, get_w_entropy_map
from utils.utils import get_wrmse


class IG_baseline:
    def __init__(self, params: Dict, writer: SummaryWriter, num_episode):
        self.params = params
        self.num_episode = num_episode
        self.budget = params["experiment"]["constraints"]["budget"]
        self.n_agents = params["experiment"]["missions"]["n_agents"]
        self.class_weighting = params["experiment"]["missions"]["class_weighting"]
        self.communication = params["experiment"]["baselines"]["information_gain"][
            "communication"
        ]
        self.coma_wrapper = COMAWrapper(params, writer)
        self.grid_map = GridMap(params)
        self.sensor_model = AltitudeSensorModel(params)
        self.sensor = Sensor(self.sensor_model, self.grid_map)
        self.mapping = Mapping(self.grid_map, self.sensor, self.params, num_episode)
        self.agent_state_space = AgentStateSpace(params)
        self.action_space = AgentActionSpace(params)
        self.batch_memory = BatchMemory(params, self.coma_wrapper)
        self.camera = Camera(params, self.sensor_model, self.grid_map)
        self.simulation = Simulation(
            params, self.sensor, num_episode, self.sensor_model
        )
        self.writer = writer

    def execute(self):
        agents = []
        for agent_id in range(self.n_agents):
            agents.append(
                Agent(
                    self.coma_wrapper.actor_network,
                    self.params,
                    self.mapping,
                    agent_id,
                    self.agent_state_space,
                )
            )
        # next_positions = []
        agent_positions = []
        agent_altitudes = []
        relative_rewards = []
        absolute_rewards = []
        entropies = []
        rmses = []
        init_maps = []

        for agent_id in range(self.n_agents):
            init_maps.append(agents[agent_id].local_map)
        current_global_map = init_maps[0].copy()

        entropy_map = get_w_entropy_map(
            None,
            current_global_map,
            self.mapping.simulated_map,
            "eval",
            self.agent_state_space,
        )[0]
        map_unique, map_counts = np.unique(
            self.mapping.simulated_map, return_counts=True
        )
        target_counts = map_counts[-1]
        entropy_masked = entropy_map.copy()
        entropy_masked[self.mapping.simulated_map == 0] = 0
        entropy = np.sum(entropy_masked) / target_counts

        # entropy = np.mean(
        #     get_w_entropy_map(None, current_global_map, self.mapping.simulated_map, "eval", self.agent_state_space)[0])
        rmse = get_wrmse(current_global_map, self.mapping.simulated_map)
        entropies.append(entropy)
        rmses.append(rmse)

        for t in range(self.budget + 1):
            action_position_list = []
            information_gain_list = []
            next_maps = []
            maps2communicate_list = []
            next_positions = []
            altitudes = []

            global_information, positions, observations = self.coma_wrapper.build_observations(
                self.mapping,
                agents,
                self.num_episode,
                t,
                self.params,
                self.batch_memory,
                None,
            )

            if t == 0:
                agent_positions.append(positions)
                current_global_map = self.mapping.fuse_map(
                    current_global_map, global_information, None, "global"
                )

            for agent_id in range(self.n_agents):
                action_mask = self.action_space.get_action_mask(
                    agents[agent_id].position
                )[0]
                action_mask = self.action_space.apply_collision_mask(
                    agents[agent_id].position,
                    action_mask,
                    next_positions,
                    self.agent_state_space,
                )
                action_positions, information_gains = self.get_individual_ig(
                    agents[agent_id].position, action_mask, agents[agent_id].local_map
                )
                action_position_list.append(action_positions)
                information_gain_list.append(information_gains)
                next_positions.append(agents[agent_id].position)

            relative_information_gains = self.get_relative_ig(information_gain_list)
            if self.communication:
                cell_utilities = self.get_cell_utilities(
                    action_position_list, relative_information_gains
                )
            else:
                cell_utilities = relative_information_gains

            for agent_id in range(self.n_agents):
                action = self.select_action(cell_utilities[agent_id])
                agents[agent_id].position = self.action_space.action_to_position(
                    agents[agent_id].position, action
                )
                agents[agent_id].local_map, agents[agent_id].map_footprint, _, agents[
                    agent_id
                ].map2communicate, agents[
                    agent_id
                ].footprint_img = self.mapping.update_grid_map(
                    agents[agent_id].position, agents[agent_id].local_map, t, None
                )
                next_maps.append(agents[agent_id].local_map)
                maps2communicate_list.append(agents[agent_id].map2communicate)
                next_positions.append(agents[agent_id].position)
                altitudes.append(agents[agent_id].position[2])

            agent_positions.append(next_positions)
            agent_altitudes.append(altitudes)

            next_global_map = self.mapping.fuse_map(
                current_global_map, maps2communicate_list, None, "global"
            )
            current_global_map = next_global_map.copy()
            done, relative_reward, absolute_reward = get_global_reward(
                current_global_map,
                next_global_map,
                None,
                None,
                self.mapping.simulated_map,
                self.agent_state_space,
                None,
                None,
                t,
                self.budget,
            )

            relative_rewards.append(relative_reward)
            absolute_rewards.append(absolute_reward)

            entropy_map = get_w_entropy_map(
                None,
                next_global_map,
                self.mapping.simulated_map,
                "eval",
                self.agent_state_space,
            )[0]
            map_unique, map_counts = np.unique(
                self.mapping.simulated_map, return_counts=True
            )
            target_counts = map_counts[-1]
            entropy_masked = entropy_map.copy()
            entropy_masked[self.mapping.simulated_map == 0] = 0
            entropy = np.sum(entropy_masked) / target_counts

            # entropy_map = get_w_entropy_map(None, next_global_map, self.mapping.simulated_map, "eval", self.agent_state_space)[0]
            # entropy = np.mean(entropy_map)
            rmse = get_wrmse(next_global_map, self.mapping.simulated_map)
            entropies.append(entropy)
            rmses.append(rmse)

        # plot_trajectories(agent_positions, self.n_agents, self.writer, self.num_episode, None, None, self.mapping.simulated_map)

        return (
            sum(relative_rewards),
            sum(absolute_rewards),
            agent_altitudes,
            entropies,
            rmses,
        )

    def get_individual_ig(self, position, action_mask, map_state):
        action_positions = []
        information_gains = []

        for action in range(len(action_mask)):
            if action_mask[action] == 0:
                action_positions.append(0)
                information_gains.append(0)
            else:
                new_position = self.action_space.action_to_position(position, action)
                footprint = self.camera.project_field_of_view(
                    new_position, self.grid_map.resolution_x, self.grid_map.resolution_y
                )[1]
                map_section = map_state[
                    footprint[2] : footprint[3], footprint[0] : footprint[1]
                ].copy()

                noise_level = self.sensor_model.get_noise_variance(new_position[2])
                class_weightings_1 = self.mapping.update_cells(
                    map_section.copy(), 1 - noise_level, None
                )
                class_weightings_2 = self.mapping.update_cells(
                    map_section.copy(), noise_level, None
                )
                class_weightings_1[class_weightings_1 > 0.501] = 1
                # class_weightings_1[class_weightings_1 < 0.501] = 0.5
                class_weightings_1[class_weightings_1 < 0.499] = 0
                class_weightings_2[class_weightings_2 > 0.501] = 1
                # class_weightings_2[class_weightings_2 < 0.501] = 0.5
                class_weightings_2[class_weightings_2 < 0.499] = 0

                ig = (
                    map_section
                    * (
                        get_shannon_entropy(map_section)
                        - get_shannon_entropy(
                            self.mapping.update_cells(
                                map_section, 1 - noise_level, None
                            )
                        )
                    )
                    * class_weightings_1
                    + (1 - map_section)
                    * (
                        get_shannon_entropy(map_section)
                        - get_shannon_entropy(
                            self.mapping.update_cells(map_section, noise_level, None)
                        )
                    )
                    * class_weightings_2
                )

                # measurement = map_section * (1 - noise_level) + (1 - map_section) * noise_level
                # cell_update = self.mapping.update_cells(map_section, measurement, None)
                # weightings = cell_update.copy()
                # weightings[weightings > 0.501] = 1
                # # weightings[weightings < 0.501] = 0.5
                # weightings[weightings < 0.499] = 0
                #
                # entropy_reduction = get_shannon_entropy(map_section) - get_shannon_entropy(cell_update)
                # ig = weightings * entropy_reduction

                total_ig = np.sum(ig) / 1000

                action_positions.append(new_position)
                information_gains.append(total_ig)

        return action_positions, information_gains

    def get_relative_ig(self, information_gain_list):
        for agent_id in range(len(information_gain_list)):
            total_ig = sum(information_gain_list[agent_id])
            for pos in range(len(information_gain_list[agent_id])):
                information_gain_list[agent_id][pos] = (
                    information_gain_list[agent_id][pos] / total_ig
                )
        return information_gain_list

    def get_cell_utilities(self, action_position_list, relative_information_gains):
        for agent_id in range(len(action_position_list)):
            for pos1 in range(len(action_position_list[agent_id])):
                position1 = action_position_list[agent_id][pos1]
                relative_information_gain1 = relative_information_gains[agent_id][pos1]
                for id2 in range(len(action_position_list)):
                    if id2 == agent_id:
                        pass
                    else:
                        for pos2 in range(len(action_position_list[id2])):
                            position2 = action_position_list[id2][pos2]
                            relative_information_gain2 = relative_information_gains[
                                id2
                            ][pos2]
                            if np.array_equal(position1, position2):
                                if not type(position1) is np.ndarray:
                                    pass
                                else:
                                    relative_information_gains[agent_id][pos1] = (
                                        relative_information_gain1
                                        * (1 - relative_information_gain2)
                                    )
        return relative_information_gains

    def select_action(self, cell_utilities):
        return np.argmax(cell_utilities)


def save_mission_numbers(entropy_list, rmse_list, trials, budget):
    print(f"entropy_list: {entropy_list}")
    entropy_metrics = dict()
    rmse_metrics = dict()
    for i in range(trials):
        entropy_metrics[i] = {}
        rmse_metrics[i] = {}
        for t in range(budget + 2):
            entropy_metrics[i][t] = float(entropy_list[i][t])
            rmse_metrics[i][t] = float(rmse_list[i][t])
    print(f"entropy_metrics: {entropy_metrics}")
    print(f"rmse_metrics: {rmse_metrics}")

    with open("/home/penguin2/Documents/PAPER_PLOTS/ig_zero_f1.json", "w") as fp:
        json.dump([entropy_metrics, rmse_metrics], fp)
        # json.dump(entropy_metrics, fp)


def main():
    constants.log_env_variables()
    params = load_params(constants.CONFIG_FILE_PATH)
    writer = SummaryWriter(constants.LOG_DIR)
    trials = params["experiment"]["baselines"]["information_gain"]["trials"]
    budget = params["experiment"]["constraints"]["budget"]
    n_agents = params["experiment"]["missions"]["n_agents"]
    returns = []
    altitude_list = []
    entropies_list = []
    rmse_list = []
    for trial in range(1, trials + 1):
        ig_baseline = IG_baseline(params, writer, trial)
        _, global_return, agent_altitudes, entropies, rmses = ig_baseline.execute()
        altitude_list.append([item for sublist in agent_altitudes for item in sublist])
        entropies_list.append(entropies)
        rmse_list.append(rmses)
        print(f"Trial {trial}: {global_return}")
        returns.append(global_return)

    save_mission_numbers(entropies_list, rmse_list, trials, budget)
    mean_return = sum(returns) / len(returns)
    max_return = max(returns)
    min_return = min(returns)
    std_return = np.std(np.array(returns))
    print(f"Mean return of {trials} trials: {mean_return}")
    print(f"Max return of {trials} trials: {max_return}")
    print(f"Min return of {trials} trials: {min_return}")
    print(f"Std dev return of {trials} trials: {std_return}")

    altitude_list = [item for sublist in altitude_list for item in sublist]
    altitude_counts = [altitude_list.count(i) for i in [5, 10, 15]]
    altitude_ratio = [(i / sum(altitude_counts)) for i in altitude_counts]
    print(f"Altitude_counts: {altitude_counts}")
    print(f"Altitude_ratio: {altitude_ratio}")


if __name__ == "__main__":
    main()
