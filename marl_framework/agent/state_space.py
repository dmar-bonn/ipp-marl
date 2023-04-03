import logging
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


class AgentStateSpace:
    def __init__(self, params: Dict):
        self.params = params
        self.seed = params["environment"]["seed"]
        self.spacing = params["experiment"]["constraints"]["spacing"]
        self.min_altitude = params["experiment"]["constraints"]["min_altitude"]
        self.max_altitude = params["experiment"]["constraints"]["max_altitude"]
        self.space_x_dim = params["environment"]["x_dim"] // self.spacing + 1
        self.space_y_dim = params["environment"]["y_dim"] // self.spacing + 1
        self.space_z_dim = (self.max_altitude - self.min_altitude) // self.spacing + 1
        self.space_dim = np.array(
            [self.space_x_dim, self.space_y_dim, self.space_z_dim]
        )

        self.class_weighting = params["experiment"]["missions"]["class_weighting"]
        self.planning_uncertainty = params["experiment"]["missions"][
            "planning_uncertainty"
        ]

    def get_random_agent_state(self, agent_id, episode):
        # r = np.random.RandomState(seed=self.seed * episode * agent_id)
        # state_x = self.spacing * r.randint(0, self.space_x_dim)
        # state_y = self.spacing * r.randint(0, self.space_y_dim)
        # state_z = 15  # self.spacing * r.randint(1, self.space_z_dim + 1)

        if agent_id == 0:
            state_x = 10   # self.spacing * r.randint(0, self.space_x_dim)
            state_y = 10   # self.spacing * r.randint(0, self.space_y_dim)
            state_z = 15   # self.spacing * r.randint(1, self.space_z_dim + 1)
        elif agent_id == 1:
            state_x = 40   # self.spacing * r.randint(0, self.space_x_dim)
            state_y = 10   # self.spacing * r.randint(0, self.space_y_dim)
            state_z = 15   # self.spacing * r.randint(1, self.space_z_dim + 1)
        elif agent_id == 2:
            state_x = 40   # self.spacing * r.randint(0, self.space_x_dim)
            state_y = 40   # self.spacing * r.randint(0, self.space_y_dim)
            state_z = 15   # self.spacing * r.randint(1, self.space_z_dim + 1)
        elif agent_id == 3:
            state_x = 10   # self.spacing * r.randint(0, self.space_x_dim)
            state_y = 40   # self.spacing * r.randint(0, self.space_y_dim)
            state_z = 15   # self.spacing * r.randint(1, self.space_z_dim + 1)

        # if agent_id == 4:
        #     state_x = 10   # self.spacing * r.randint(0, self.space_x_dim)
        #     state_y = 25   # self.spacing * r.randint(0, self.space_y_dim)
        #     state_z = 15   # self.spacing * r.randint(1, self.space_z_dim + 1)
        # elif agent_id == 5:
        #     state_x = 25   # self.spacing * r.randint(0, self.space_x_dim)
        #     state_y = 40  # 10   # self.spacing * r.randint(0, self.space_y_dim)
        #     state_z = 15   # self.spacing * r.randint(1, self.space_z_dim + 1)
        # elif agent_id == 6:
        #     state_x = 40   # self.spacing * r.randint(0, self.space_x_dim)
        #     state_y = 25   # self.spacing * r.randint(0, self.space_y_dim)
        #     state_z = 15   # self.spacing * r.randint(1, self.space_z_dim + 1)
        # elif agent_id == 7:
        #     state_x = 25   # self.spacing * r.randint(0, self.space_x_dim)
        #     state_y = 10   # self.spacing * r.randint(0, self.space_y_dim)
        #     state_z = 15   # self.spacing * r.randint(1, self.space_z_dim + 1)

        return np.array([state_x, state_y, state_z])

    def position_to_index(self, position):
        state_x = position[0] // self.spacing
        state_y = position[1] // self.spacing
        state_z = (position[2] // self.spacing) - 1
        return np.array([state_x, state_y, state_z])

    def index_to_position(self, state):
        position = np.array(
            [
                state[0] * self.spacing,
                state[1] * self.spacing,
                self.spacing + state[2] * self.spacing,
            ]
        )
        return position
