import logging
from typing import Dict

import numpy as np

from marl_framework.agent.state_space import AgentStateSpace
from marl_framework.mapping.grid_maps import GridMap
from marl_framework.mapping.simulations import Simulation
from marl_framework.sensors import Sensor
from marl_framework.sensors.cameras import Camera
from marl_framework.sensors.models.sensor_models import AltitudeSensorModel
from matplotlib import pyplot as plt

from utils.utils import get_fixed_footprint_coordinates

logger = logging.getLogger(__name__)


class Mapping:
    def __init__(self, grid_map: GridMap, sensor: Sensor, params: Dict, episode: int):
        self.params = params
        self.grid_map = grid_map
        self.sensor = sensor
        self.sensor_model = AltitudeSensorModel(self.params)
        self.agent_state_space = AgentStateSpace(self.params)
        self.simulation = Simulation(
            self.params, self.sensor, episode, self.sensor_model
        )
        self.simulated_map = self.simulation.simulated_map
        self.prior = self.params["mapping"]["prior"]

    def update_grid_map(self, position, map_state, t, mode):

        camera = Camera(self.params, self.sensor_model, self.grid_map)

        # Calculate camera footprint
        footprint, footprint_clipped = camera.project_field_of_view(
            position, self.grid_map.resolution_x, self.grid_map.resolution_y
        )

        footprint_img = (
            np.ones((footprint[1] - footprint[0], footprint[3] - footprint[2])) * 0.5
        )

        # Extract map section within camera footprint
        map_section = map_state[
            footprint_clipped[2] : footprint_clipped[3],
            footprint_clipped[0] : footprint_clipped[1],
        ]
        measurement = self.simulation.get_measurement(
            position[2], footprint_clipped, mode
        )

        # Update cells of extracted map section
        cell_update = self.update_cells(map_section, measurement, mode)

        # Insert updated map section back into map
        map_state[
            footprint_clipped[2] : footprint_clipped[3],
            footprint_clipped[0] : footprint_clipped[1],
        ] = cell_update

        # Create map to communicate
        map2communicate = None

        map2communicate = np.ones_like(map_state) * 0.5
        map2communicate[
            footprint_clipped[2] : footprint_clipped[3],
            footprint_clipped[0] : footprint_clipped[1],
        ] = measurement

        footprint_fixed = get_fixed_footprint_coordinates(footprint, footprint_clipped)
        footprint_img[
            footprint_fixed[2] : footprint_fixed[3],
            footprint_fixed[0] : footprint_fixed[1],
        ] = measurement

        return map_state, cell_update, footprint_clipped, map2communicate, footprint_img

    def fuse_map(self, own_map_state, other_map_states, agent_id, fusion_mode):

        if fusion_mode == "local":  # local map fusion
            fused_map = np.float32(own_map_state.copy())
            for key in other_map_states:
                if key == agent_id:  # own measurement was already used
                    pass
                else:
                    other_state = np.float32(other_map_states[key]["map2communicate"])
                    fused_map = self.update_cells(fused_map, other_state, "eval")

        else:  # global map fusion
            if isinstance(other_map_states, dict):
                fused_map = np.float32(own_map_state.copy())  # own/previous
                for other_state in other_map_states:
                    other_map = np.float32(
                        other_map_states[other_state]["map2communicate"]
                    )
                    fused_map = self.update_cells(fused_map, other_map, "eval")
            else:
                fused_map = np.float32(own_map_state.copy())
                for other_state in other_map_states:
                    fused_map = self.update_cells(fused_map, other_state, "eval")

        return fused_map

    def update_cells(self, map_section, measurement, mode):
        return self.apply_update(map_section, measurement, mode)

    def apply_update(self, x, y, mode):
        x[0.9999 < x] = 0.9999
        x[0.0001 > x] = 0.0001
        l_x = np.log(x / (1 - x))
        l_y = np.log(y / (1 - y))
        l_xy = l_x + l_y
        l_p = np.log(self.prior / (1 - self.prior))
        l_xyp = l_xy - l_p
        update = self.get_update(l_xyp)

        return update

    @staticmethod
    def get_update(l_):
        l = 1 - (1 / (1 + np.exp(l_)))
        return l

    def init_priors(self):
        grid_map_init = np.full(
            (int(self.grid_map.x_dim), int(self.grid_map.y_dim)),
            self.prior,
            dtype="float32",
        )
        return grid_map_init
