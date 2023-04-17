import logging
from typing import Dict, Tuple

import numpy as np

from marl_framework.mapping.grid_maps import GridMap
from marl_framework.sensors import Sensor
from marl_framework.sensors.models.sensor_models import AltitudeSensorModel

logger = logging.getLogger(__name__)


class Camera(Sensor):
    def __init__(
        self, params: Dict, sensor_model: AltitudeSensorModel, grid_map: GridMap
    ):

        super().__init__(sensor_model, grid_map)

        self.params = params
        self.grid_map = GridMap(self.params)

    @property
    def angle_x(self) -> float:
        return self.params["sensor"]["field_of_view"]["angle_x"]

    @property
    def angle_y(self) -> float:
        return self.params["sensor"]["field_of_view"]["angle_y"]

    def field_of_view_range(self, height: float) -> Tuple[float, float]:
        """
        Calculates x and y dimensions [m] projecting FoV to ground plane from a given height in [m].

        Args:
            height (float): height above ground plane in [m]

        Returns:
            (float): x and y dimensions of projection in [m]
        """
        x_dim = 2 * height * np.tan(0.5 * np.radians(self.angle_x))
        y_dim = 2 * height * np.tan(0.5 * np.radians(self.angle_y))

        return x_dim, y_dim

    def project_field_of_view(self, position: np.array, res_x, res_y):
        """
        Project camera's FoV from a certain environment measurement position (and height) to planar
        top left and bottom right grid cells in grid map spanning the FoV.

        Args:
            position (np.array): UAv's continuous 3D environment position

        Returns:
            xl, yu (int, int): defining the upper left grid cell of the camera's projected FoV
            xr, yd (int, int): defining the lower right grid cell of the camera's projected FoV
            :param position:
            :param res_y:
            :param res_x:
        """

        x_range_m, y_range_m = self.field_of_view_range(position[2])

        x_range_cells = np.floor(x_range_m / res_x)
        y_range_cells = np.floor(y_range_m / res_y)
        position_grid = np.floor(position[:2] / res_x)
        fov_radius = np.floor(0.5 * np.array([x_range_cells, y_range_cells]))

        xl, yu = position_grid[:2] - fov_radius
        xr, yd = position_grid[:2] + fov_radius

        footprint = [int(yu), int(yd), int(xl), int(xr)]

        xl, xr = np.clip(np.array([xl, xr]), a_min=0, a_max=self.grid_map.x_dim - 1)
        yu, yd = np.clip(np.array([yu, yd]), a_min=0, a_max=self.grid_map.y_dim - 1)

        footprint_clipped = [int(yu), int(yd), int(xl), int(xr)]

        return footprint, footprint_clipped

    def take_measurement(self, position: np.array, verbose: bool = True) -> np.array:
        pass

    def get_resolution_factor(self, position: np.array) -> float:
        pass


class RGBCamera(Camera):
    def __init__(
        self,
        field_of_view: Dict,
        sensor_model: AltitudeSensorModel,
        grid_map: GridMap,
        encoding: str = "rgb8",
    ):
        """
        RGB camera class with specialized measurement functions for this type of cameras.

        Args:
            field_of_view (Dict): holds x and y FoV angles in degrees
            sensor_model (SensorModel): sensor model defining sensor measurement characteristics
            grid_map (GridMap): grid map representation of environment
            encoding (str, optional): states order of rgb channels

        """
        super().__init__(field_of_view, sensor_model, grid_map)
        self.encoding = encoding

    def take_measurement(self, position: np.array, verbose: bool = True) -> np.array:
        """Function returns random RGB image or simulated measurement if sensor simulation object is set"""
        if verbose:
            logger.info(f"Take measurement at point: {position}")

        if self.sensor_simulation is None:
            return (
                np.random.random((self.grid_map.x_dim, self.grid_map.y_dim, 3)) * 255
            ).astype(int)

        return self.sensor_simulation.take_measurement(position)

    def get_resolution_factor(self, position: np.array) -> float:
        """Returns sensor resolution factor based on measurement altitude"""
        altitude = position[2]
        return 2 if altitude > 6.0 else 1
