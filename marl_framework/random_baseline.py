import json
from typing import Dict

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import constants
from agent.agent import Agent
from agent.state_space import AgentStateSpace
from coma_wrapper import COMAWrapper
from logger import setup_logger
from mapping.grid_maps import GridMap
from mapping.mappings import Mapping
from params import load_params
from sensors import Sensor
from sensors.models import SensorModel
from utils.reward import get_global_reward
from utils.state import get_w_entropy_map
from utils.utils import get_wrmse


class RandomBaseline:
    def __init__(self, params: Dict, writer: SummaryWriter, num_episode):
        self.params = params
        self.num_episode = num_episode
        self.n_agents = params["experiment"]["missions"]["n_agents"]
        self.x_dim = params["environment"]["x_dim"]
        self.y_dim = params["environment"]["y_dim"]
        self.altitude = params["experiment"]["baselines"]["lawnmower"]["altitude"]
        self.budget = params["experiment"]["constraints"]["budget"]
        self.coma_wrapper = COMAWrapper(params, writer)
        self.grid_map = GridMap(self.params)
        self.sensor = Sensor(SensorModel(), self.grid_map)
        self.mapping = Mapping(self.grid_map, self.sensor, self.params, num_episode)
        self.agent_state_space = AgentStateSpace(self.params)
        self.map = self.mapping.init_priors()

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
        rewards = []
        updated_map = self.map.copy()
        entropies = []
        rmses = []
        init_maps = []

        for agent_id in range(self.n_agents):
            init_maps.append(agents[agent_id].local_map)
        fused_init_map = init_maps[0]

        entropy_map = get_w_entropy_map(
            None,
            fused_init_map,
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

        entropies.append(entropy)
        rmse = get_wrmse(fused_init_map, self.mapping.simulated_map)
        rmses.append(rmse)

        for t in range(self.budget + 1):
            next_positions = []
            for agent in range(self.n_agents):
                if t == 0:
                    agents[
                        agent
                    ].position = self.agent_state_space.get_random_agent_state(
                        agent, self.num_episode
                    )
                else:
                    action_mask, _ = agents[agent].action_space.get_action_mask(
                        agents[agent].position
                    )
                    action = torch.multinomial(
                        torch.tensor(action_mask).float(), 1, replacement=True
                    )
                    agents[agent].position = agents[
                        agent
                    ].action_space.action_to_position(
                        agents[agent].position, action.item()
                    )
                    next_positions.append(agents[agent].position)
                updated_map, _, _, _, _ = self.mapping.update_grid_map(
                    agents[agent].position, updated_map.copy(), t, None
                )

            entropy_map = get_w_entropy_map(
                None,
                updated_map,
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

            entropies.append(entropy)
            rmse = get_wrmse(updated_map, self.mapping.simulated_map)
            rmses.append(rmse)

            rewards.append(0)
            self.map = updated_map.copy()


        return sum(rewards), entropies, rmses


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

    with open("/home/penguin2/Documents/PAPER_PLOTS/random_f1.json", "w") as fp:
        json.dump([entropy_metrics, rmse_metrics], fp)


def main():
    constants.log_env_variables()
    params = load_params(constants.CONFIG_FILE_PATH)
    writer = SummaryWriter(constants.LOG_DIR)
    n_episodes = params["experiment"]["baselines"]["random"]["n_episodes"]
    budget = params["experiment"]["constraints"]["budget"]
    entropies_list = []
    rmse_list = []
    for episode in range(1, n_episodes + 1):
        logger.info(f"episode {episode}")
        returns = []
        random_baseline = RandomBaseline(params, writer, episode)
        global_return, entropies, rmses = random_baseline.execute()
        returns.append(global_return)
        entropies_list.append(entropies)
        rmse_list.append(rmses)

    save_mission_numbers(entropies_list, rmse_list, n_episodes, budget)


if __name__ == "__main__":
    logger = setup_logger()
    main()
