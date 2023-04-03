import logging
from typing import Dict

import numpy as np

from utils.utils import compute_euclidean_distance

logger = logging.getLogger(__name__)


class CommunicationLog:
    def __init__(self, params: Dict, num_episode: int):
        self.params = params
        self.communication_range = self.params["MARL_cast"]["state_space"][
            "communication_range"
        ]
        self.fix_range = self.params["MARL_cast"]["state_space"][
            "fix_range"
        ]
        self.failure_rate = self.params["MARL_cast"]["state_space"][
            "failure_rate"
        ]
        self.n_agents = self.params["mission"]["n_agents"]
        self.global_log = dict()

        if not self.fix_range:
            np.random.seed(num_episode)
            self.communication_range = 0
            range_idx = np.random.randint(4)
            if range_idx == 1:
                self.communication_range = 15
            elif range_idx == 2:
                self.communication_range = 25
            elif range_idx == 3:
                self.communication_range = 100

    # Insert all global information into global log
    def store_agent_message(self, message: Dict, agent_id: int):
        self.global_log[agent_id] = message
        return self.global_log

    # Extract communication message from global log within communication range (incl. own information)
    def get_messages(self, agent_id: int):
        agent_position = self.global_log[agent_id]["position"]
        local_log = dict()
        for agent_id in self.global_log.keys():
            communication = False
            other_agent_position = self.global_log[agent_id]["position"]

            r = np.random.random_sample()
            if compute_euclidean_distance(agent_position, other_agent_position) < 0.001:
                communication = True
            if (0.001 <= compute_euclidean_distance(agent_position, other_agent_position) <= self.communication_range) and (r >= self.failure_rate):
                communication = True
            if communication:
                local_log[agent_id] = self.global_log[agent_id]

            # if compute_euclidean_distance(agent_position, other_agent_position) <= self.communication_range:
            #     local_log[agent_id] = self.global_log[agent_id]

        return local_log

    # get list with all agent's positions, needed as critic input
    def get_global_positions(self):
        global_positions = []
        for agent_id in self.global_log:
            global_positions.append([self.global_log[agent_id]["position"]])
        return global_positions
