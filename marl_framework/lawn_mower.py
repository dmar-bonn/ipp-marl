import json

from matplotlib import pyplot as plt

from batch_memory import BatchMemory
from coma_wrapper import COMAWrapper
import constants
from logger import setup_logger
from mapping.grid_maps import GridMap
from params import load_params
from typing import Dict
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from mapping.mappings import Mapping
from sensors import Sensor
from sensors.models import SensorModel
from utils.reward import get_global_reward
from agent.state_space import AgentStateSpace
from utils.state import get_w_entropy_map
from utils.utils import get_wrmse


class LawnMower:
    def __init__(self, params: Dict, writer: SummaryWriter, num_episode):
        self.params = params
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
        batch_memory = BatchMemory(self.params, self.coma_wrapper)
        positions1 = None
        positions2 = None
        positions3 = None
        positions4 = None
        rewards = []
        entropies = []
        rmses = []
        # if self.n_agents == 2 and self.altitude == 10 and self.budget == 14:
        positions1 = np.array([[10, 10, self.altitude],
                               [15, 10, self.altitude],
                               [20, 10, self.altitude],
                               [25, 10, self.altitude],
                               [30, 10, self.altitude],
                               [35, 10, self.altitude],
                               [40, 10, self.altitude],
                               [40, 15, self.altitude],
                               [40, 20, self.altitude],
                               [35, 20, self.altitude],
                               [30, 20, self.altitude],
                               [25, 20, self.altitude],
                               [20, 20, self.altitude],
                               [15, 20, self.altitude],
                               [10, 20, self.altitude]])
        positions2 = np.array([[10, 30, self.altitude],
                               [15, 30, self.altitude],
                               [20, 30, self.altitude],
                               [25, 30, self.altitude],
                               [30, 30, self.altitude],
                               [35, 30, self.altitude],
                               [40, 30, self.altitude],
                               [40, 35, self.altitude],
                               [40, 40, self.altitude],
                               [35, 40, self.altitude],
                               [30, 40, self.altitude],
                               [25, 40, self.altitude],
                               [20, 40, self.altitude],
                               [15, 40, self.altitude],
                               [10, 40, self.altitude]])
        # if self.n_agents == 4 and self.altitude == 10 and self.budget == 14:
        positions3 = np.array([[10, 10, self.altitude],
                               [10, 15, self.altitude],
                               [10, 20, self.altitude],
                               [10, 25, self.altitude],
                               [10, 30, self.altitude],
                               [10, 35, self.altitude],
                               [10, 40, self.altitude],
                               [15, 40, self.altitude],
                               [20, 40, self.altitude],
                               [20, 35, self.altitude],
                               [20, 30, self.altitude],
                               [20, 25, self.altitude],
                               [20, 20, self.altitude],
                               [20, 15, self.altitude],
                               [20, 10, self.altitude]])
        positions4 = np.array([[30, 10, self.altitude],
                               [30, 15, self.altitude],
                               [30, 20, self.altitude],
                               [30, 25, self.altitude],
                               [30, 30, self.altitude],
                               [30, 35, self.altitude],
                               [30, 40, self.altitude],
                               [35, 40, self.altitude],
                               [40, 40, self.altitude],
                               [40, 35, self.altitude],
                               [40, 30, self.altitude],
                               [40, 25, self.altitude],
                               [40, 20, self.altitude],
                               [40, 15, self.altitude],
                               [40, 10, self.altitude]])

        positions5 = np.array([[10, 10, self.altitude],
                               [15, 10, self.altitude],
                               [20, 10, self.altitude],
                               [25, 10, self.altitude],
                               [30, 10, self.altitude],
                               [35, 10, self.altitude],
                               [40, 10, self.altitude],
                               [40, 15, self.altitude],
                               [40, 20, self.altitude],
                               [35, 20, self.altitude],
                               [30, 20, self.altitude],
                               [25, 20, self.altitude],
                               [20, 20, self.altitude],
                               [15, 20, self.altitude],
                               [10, 20, self.altitude]])
        positions6 = np.array([[10, 30, self.altitude],
                               [15, 30, self.altitude],
                               [20, 30, self.altitude],
                               [25, 30, self.altitude],
                               [30, 30, self.altitude],
                               [35, 30, self.altitude],
                               [40, 30, self.altitude],
                               [40, 35, self.altitude],
                               [40, 40, self.altitude],
                               [35, 40, self.altitude],
                               [30, 40, self.altitude],
                               [25, 40, self.altitude],
                               [20, 40, self.altitude],
                               [15, 40, self.altitude],
                               [10, 40, self.altitude]])
        # if self.n_agents == 4 and self.altitude == 10 and self.budget == 14:
        positions7 = np.array([[10, 10, self.altitude],
                               [10, 15, self.altitude],
                               [10, 20, self.altitude],
                               [10, 25, self.altitude],
                               [10, 30, self.altitude],
                               [10, 35, self.altitude],
                               [10, 40, self.altitude],
                               [15, 40, self.altitude],
                               [20, 40, self.altitude],
                               [20, 35, self.altitude],
                               [20, 30, self.altitude],
                               [20, 25, self.altitude],
                               [20, 20, self.altitude],
                               [20, 15, self.altitude],
                               [20, 10, self.altitude]])
        positions8 = np.array([[30, 10, self.altitude],
                               [30, 15, self.altitude],
                               [30, 20, self.altitude],
                               [30, 25, self.altitude],
                               [30, 30, self.altitude],
                               [30, 35, self.altitude],
                               [30, 40, self.altitude],
                               [35, 40, self.altitude],
                               [40, 40, self.altitude],
                               [40, 35, self.altitude],
                               [40, 30, self.altitude],
                               [40, 25, self.altitude],
                               [40, 20, self.altitude],
                               [40, 15, self.altitude],
                               [40, 10, self.altitude]])

        # for pos1, pos2 in zip(positions1, positions2):
        #     batch_memory.add(0, state=pos1)
        #     batch_memory.add(1, state=pos2)

        # for pos1, pos2, pos3, pos4 in zip(positions1, positions2, positions3, positions4):
        #     batch_memory.add(0, state=pos1)
        #     batch_memory.add(1, state=pos2)
        #     batch_memory.add(2, state=pos3)
        #     batch_memory.add(3, state=pos4)

        for pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8 in zip(positions1, positions2, positions3, positions4, positions5, positions6, positions7, positions8):
            batch_memory.add(0, state=pos1)
            batch_memory.add(1, state=pos2)
            batch_memory.add(2, state=pos3)
            batch_memory.add(3, state=pos4)
            batch_memory.add(4, state=pos5)
            batch_memory.add(5, state=pos6)
            batch_memory.add(6, state=pos7)
            batch_memory.add(7, state=pos8)

        entropy_map = get_w_entropy_map(None, self.map.copy(), self.mapping.simulated_map, "eval", self.agent_state_space)[0]
        map_unique, map_counts = np.unique(self.mapping.simulated_map, return_counts=True)
        target_counts = map_counts[-1]
        entropy_masked = entropy_map.copy()
        entropy_masked[self.mapping.simulated_map == 0] = 0
        entropy = np.sum(entropy_masked) / target_counts

        # entropy = np.mean(get_w_entropy_map(None, self.map.copy(), self.mapping.simulated_map, "eval", self.agent_state_space)[0])
        entropies.append(entropy)
        rmse = get_wrmse(self.map.copy(), self.mapping.simulated_map)
        rmses.append(rmse)

        for pos_idx in range(len(positions1)):
            position1 = batch_memory.get(pos_idx, 0, "state")
            updated_map, _, _, _, _ = self.mapping.update_grid_map(position1, self.map.copy(), pos_idx, None)
            position2 = batch_memory.get(pos_idx, 1, "state")
            updated_map, _, _, _, _  = self.mapping.update_grid_map(position2, updated_map.copy(), pos_idx, None)
            position3 = batch_memory.get(pos_idx, 2, "state")
            updated_map, _, _, _, _  = self.mapping.update_grid_map(position3, updated_map.copy(), pos_idx, None)
            position4 = batch_memory.get(pos_idx, 3, "state")
            updated_map, _, _, _, _  = self.mapping.update_grid_map(position4, updated_map.copy(), pos_idx, None)

            position5 = batch_memory.get(pos_idx, 4, "state")
            updated_map, _, _, _, _ = self.mapping.update_grid_map(position5, updated_map.copy(), pos_idx, None)
            position6 = batch_memory.get(pos_idx, 5, "state")
            updated_map, _, _, _, _ = self.mapping.update_grid_map(position6, updated_map, pos_idx, None)
            position7 = batch_memory.get(pos_idx, 6, "state")
            updated_map, _, _, _, _ = self.mapping.update_grid_map(position7, updated_map, pos_idx, None)
            position8 = batch_memory.get(pos_idx, 7, "state")
            updated_map, _, _, _, _ = self.mapping.update_grid_map(position8, updated_map, pos_idx, None)

            # _, reward, _ = get_global_reward(self.map, updated_map, None, None, self.mapping.simulated_map,
            #                               self.agent_state_space, None, None)

            entropy_map = get_w_entropy_map(None, updated_map, self.mapping.simulated_map, "eval", self.agent_state_space)[0]
            map_unique, map_counts = np.unique(self.mapping.simulated_map, return_counts=True)
            target_counts = map_counts[-1]
            entropy_masked = entropy_map.copy()
            entropy_masked[self.mapping.simulated_map == 0] = 0
            entropy = np.sum(entropy_masked) / target_counts

            # entropy_map = (get_w_entropy_map(None, updated_map, self.mapping.simulated_map, "eval", self.agent_state_space)[0])
            # entropy = np.mean(entropy_map)
            entropies.append(entropy)
            rmse = get_wrmse(updated_map, self.mapping.simulated_map)
            rmses.append(rmse)

            rewards.append(0)  # reward)
            self.map = updated_map.copy()

        return sum(rewards), entropies, rmses


def save_mission_numbers(entropy_list, rmse_list, trials, budget):
    print(f"len entropy_list: {len(entropy_list)}")
    print(f"len entropy_list[0]: {len(entropy_list[0])}")
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

    with open('/home/penguin2/Documents/PAPER_PLOTS/coverage_8agents_f1.json', 'w') as fp:
        json.dump([entropy_metrics, rmse_metrics], fp)
        # json.dump(entropy_metrics, fp)


def main():
    constants.log_env_variables()
    params = load_params(constants.CONFIG_FILE_PATH)
    writer = SummaryWriter(constants.LOG_DIR)
    trials = params["experiment"]["baselines"]["lawnmower"]["trials"]
    budget = params["experiment"]["constraints"]["budget"]
    returns = []
    entropies_list = []
    rmse_list = []
    for trial in range(1, trials + 1):
        lawn_mower = LawnMower(params, writer, trial)
        global_return, entropies, rmses = lawn_mower.execute()
        logger.info(f"Trial {trial}: {global_return}")
        returns.append(global_return)
        entropies_list.append(entropies)
        rmse_list.append(rmses)

    save_mission_numbers(entropies_list, rmse_list, trials, budget)
    mean_return = sum(returns) / len(returns)
    max_return = max(returns)
    min_return = min(returns)
    std_return = np.std(np.array(returns))
    logger.info(f"Mean return of {trials} trials: {mean_return}")
    logger.info(f"Max return of {trials} trials: {max_return}")
    logger.info(f"Min return of {trials} trials: {min_return}")
    logger.info(f"Std dev return of {trials} trials: {std_return}")


if __name__ == "__main__":
    logger = setup_logger()
    main()
