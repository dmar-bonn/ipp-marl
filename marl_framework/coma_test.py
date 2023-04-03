import json
from typing import Dict

import logger as logger
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import constants
from actor.transformations import get_network_input
from agent.action_space import AgentActionSpace
from agent.agent import Agent
from agent.state_space import AgentStateSpace
from batch_memory import BatchMemory
from coma_wrapper import COMAWrapper
from logger import setup_logger
from mapping.grid_maps import GridMap
from mapping.mappings import Mapping
from params import load_params
from sensors import Sensor
from sensors.models import SensorModel
from utils.plotting import plot_trajectories, plot_performance
from utils.reward import get_global_reward
from utils.state import get_w_entropy_map
from utils.utils import get_wrmse


class COMATest:
    def __init__(self, params: Dict, writer: SummaryWriter, num_episode):
        self.params = params
        self.num_episode = num_episode
        self.n_agents = params["experiment"]["missions"]["n_agents"]
        self.x_dim = params["environment"]["x_dim"]
        self.y_dim = params["environment"]["y_dim"]
        self.altitude = params["experiment"]["baselines"]["lawnmower"]["altitude"]
        self.budget = params["experiment"]["constraints"]["budget"]
        self.action_space = AgentActionSpace(self.params)
        self.coma_wrapper = COMAWrapper(params, writer)
        self.grid_map = GridMap(self.params)
        self.sensor = Sensor(SensorModel(), self.grid_map)
        self.mapping = Mapping(self.grid_map, self.sensor, self.params, num_episode)
        self.agent_state_space = AgentStateSpace(self.params)
        self.map = self.mapping.init_priors()
        self.device = torch.device("cpu")  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_memory = BatchMemory(self.params, self.coma_wrapper)

    def execute(self, test_mode, num_episode):
        agents = []
        net = torch.load(
            "/home/penguin2/jonas-project/marl_framework/logs/coma/best_model.pth",
            map_location=self.device,
        )
        net.eval()

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

        current_global_map = agents[0].local_map.copy()

        rewards = []
        relative_rewards = []
        agent_positions = []
        entropies = []
        rmses = []
        agent_altitudes = []
        map_states = []
        start_positions = None

        # plt.imshow(self.mapping.simulated_map)
        # plt.title(f"ground_truth")
        # plt.clim(0, 1)
        # plt.savefig(f"/home/penguin2/Documents/plots/groundtruth.png")

        entropy_map = get_w_entropy_map(None, self.map, self.mapping.simulated_map, "eval", self.agent_state_space)[0]
        map_unique, map_counts = np.unique(self.mapping.simulated_map, return_counts=True)
        target_counts = map_counts[-1]
        entropy_masked = entropy_map.copy()
        entropy_masked[self.mapping.simulated_map == 0] = 0
        entropy = np.sum(entropy_masked) / target_counts

        rmse = get_wrmse(self.map, self.mapping.simulated_map)
        entropies.append(entropy)
        rmses.append(rmse)
        map_states.append(self.map)

        for t in range(self.budget + 1):
            global_information, positions, observations = self.coma_wrapper.build_observations(
                self.mapping, agents, num_episode, t, self.params, self.batch_memory, None
            )

            next_positions = []
            start_positions = []
            next_maps = []
            maps2communicate_list = []
            altitudes = []

            if t == 0:
                current_global_map = self.mapping.fuse_map(current_global_map, global_information, None, "global")

            for agent in range(self.n_agents):
                input_state = observations[agent].to(self.device)
                network_input = torch.unsqueeze(input_state, 0).float()
                action_probs, _ = net.forward(network_input, 0)
                action_mask, _ = agents[agent].action_space.get_action_mask(agents[agent].position)
                action_mask = self.action_space.apply_collision_mask(agents[agent].position, action_mask,
                                                                     next_positions, self.agent_state_space)
                action_mask = torch.tensor(action_mask).to(self.device)
                action = torch.argmax(action_probs * action_mask)

                if t == 0:
                    start_positions.append(agents[agent].position)

                agents[agent].position = agents[agent].action_space.action_to_position(agents[agent].position,
                                                                                       action.item())
                next_positions.append(agents[agent].position)
                agents[agent].local_map, agents[agent].map_footprint, _, agents[agent].map2communicate, agents[agent].footprint_img = self.mapping.update_grid_map(agents[agent].position, agents[agent].local_map, t, None)
                next_maps.append(agents[agent].local_map)
                maps2communicate_list.append(agents[agent].map2communicate)
                altitudes.append(agents[agent].position[2])
            agent_altitudes.append(altitudes)

            next_global_map = self.mapping.fuse_map(current_global_map, maps2communicate_list, None, "global")
            current_global_map = next_global_map.copy()

            if t == 0:
                agent_positions.append(start_positions)
            agent_positions.append(next_positions)
            done, relative_reward, reward = get_global_reward(current_global_map, next_global_map, None, None,
                                                              self.mapping.simulated_map,
                                                              self.agent_state_space, None, None, t, self.budget)

            entropy_map = get_w_entropy_map(None, next_global_map, self.mapping.simulated_map, "eval", self.agent_state_space)[0]
            map_unique, map_counts = np.unique(self.mapping.simulated_map, return_counts=True)
            target_counts = map_counts[-1]
            entropy_masked = entropy_map.copy()
            entropy_masked[self.mapping.simulated_map == 0] = 0
            entropy = np.sum(entropy_masked) / target_counts

            # entropy_map = get_w_entropy_map(None, next_global_map, self.mapping.simulated_map, "eval", self.agent_state_space)[0]
            # entropy = np.mean(entropy_map)
            rmse = get_wrmse(next_global_map, self.mapping.simulated_map)
            entropies.append(entropy)
            rmses.append(rmse)
            rewards.append(reward)
            relative_rewards.append(relative_reward)
            map_states.append(next_global_map)

        return sum(rewards), agent_positions, agent_altitudes, entropies, rmses, sum(relative_rewards), map_states

    def plot_mission(self, agent_positions, writer, entropies, trial, map_states):
        plot_trajectories(agent_positions, self.n_agents, writer, trial, 0, self.budget, self.mapping.simulated_map, map_states)
    #     plot_performance(self.budget, entropies)


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

    with open('/home/penguin2/Documents/PAPER_PLOTS/test.json', 'w') as fp:
        json.dump([entropy_metrics, rmse_metrics], fp)
        ### json.dump(entropy_metrics, fp)


def main():
    constants.log_env_variables()
    params = load_params(constants.CONFIG_FILE_PATH)
    writer = SummaryWriter(constants.LOG_DIR)
    trials = params["experiment"]["baselines"]["information_gain"]["trials"]
    budget = params["experiment"]["constraints"]["budget"]
    avg_return = None
    test_mode = "random"  # "fix" for fix starting positions, "random" for random starting positions
    entropies_list = []
    rmse_list = []
    relative_returns = []
    if test_mode == "fix":
        coma_test = COMATest(params, writer, 0)
        avg_return, agent_positions, entropies = coma_test.execute(test_mode, 0)
        # coma_test.plot_mission(agent_positions, writer, entropies)
    else:
        returns = []
        altitude_list = []
        # entropies_list = []
        # for i in range(50):
        #     coma_test = COMATest(params, writer, i)
        #     avg_return, agent_positions, entropies = coma_test.execute(test_mode, i)
        #     entropies_list.append(entropies_list)
        #     returns.append(avg_return)
        #     avg_return = sum(returns) / len(returns)
        #     if i == 0:
        #  plot_mission(agent_positions, writer, entropies_list)

        for trial in range(1, trials + 1):
            print(f"Trial: {trial}")
            coma_test = COMATest(params, writer, trial)
            global_return, agent_positions, agent_altitudes, entropies, rmses, relative_return, map_states = coma_test.execute(
                test_mode, trial)
            altitude_list.append([item for sublist in agent_altitudes for item in sublist])
            entropies_list.append(entropies)
            rmse_list.append(rmses)
            returns.append(global_return)
            relative_returns.append(relative_return)
            coma_test.plot_mission(agent_positions, writer, None, trial, map_states)
            print(f"Return: {global_return}")
            print(f"Relative return: {relative_return}")

        save_mission_numbers(entropies_list, rmse_list, trials, budget)
        mean_return = sum(returns) / len(returns)
        max_return = max(returns)
        min_return = min(returns)
        std_return = np.std(np.array(returns))
        mean_relative_return = sum(relative_returns) / len(relative_returns)
        print(f"Mean return of {trials} trials: {mean_return}")
        print(f"Max return of {trials} trials: {max_return}")
        print(f"Min return of {trials} trials: {min_return}")
        print(f"Std dev return of {trials} trials: {std_return}")
        print(f"Mean relative return of {trials} trials: {mean_relative_return}")

        altitude_list = [item for sublist in altitude_list for item in sublist]
        altitude_counts = [altitude_list.count(i) for i in [5, 10, 15]]
        altitude_ratio = [(i / sum(altitude_counts)) for i in altitude_counts]
        print(f"Altitude_counts: {altitude_counts}")
        print(f"Altitude_ratio: {altitude_ratio}")


if __name__ == "__main__":
    logger = setup_logger()
    main()
