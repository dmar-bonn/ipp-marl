import logging
from typing import Dict
import math

logger = logging.getLogger(__name__)


class GridMap:
    def __init__(self, params: Dict):
        self.params = params
        self.mean = None
        self.resolution_x = self.res_x
        self.resolution_y = self.res_y
        self.occupancy_matrix = None

    @property
    def x_dim(self) -> int:
        """Returns map's x-dimension in number of cells"""
        if "environment" not in self.params.keys():
            logger.error(f"Cannot find environment specification in config file!")
            raise ValueError

        if "x_dim" not in self.params["environment"].keys():
            logger.error(
                f"Cannot find environment's x_dim specification in config file!"
            )
            raise ValueError

        return int(self.params["environment"]["x_dim"] / self.resolution_x)

    @property
    def y_dim(self) -> int:
        """Returns map's y-dimension in number of cells"""
        if "environment" not in self.params.keys():
            logger.error(f"Cannot find environment specification in config file!")
            raise ValueError

        if "y_dim" not in self.params["environment"].keys():
            logger.error(
                f"Cannot find environment's y_dim specification in config file!"
            )
            raise ValueError

        return int(self.params["environment"]["y_dim"] / self.resolution_y)

    @property
    def res_x(self) -> int:
        min_altitude = self.params["experiment"]["constraints"]["min_altitude"]
        angle_x = self.params["sensor"]["field_of_view"]["angle_x"]
        number_x = self.params["sensor"]["pixel"]["number_x"]
        return (2 * min_altitude * math.tan(math.radians(angle_x) * 0.5)) / number_x

    @property
    def res_y(self) -> int:
        min_altitude = self.params["experiment"]["constraints"]["min_altitude"]
        angle_y = self.params["sensor"]["field_of_view"]["angle_y"]
        number_y = self.params["sensor"]["pixel"]["number_y"]
        return (2 * min_altitude * math.tan(math.radians(angle_y) * 0.5)) / number_y

    @property
    def num_grid_cells(self):
        return self.x_dim * self.y_dim
