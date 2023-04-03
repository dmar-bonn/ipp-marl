import numpy as np


class Sensor:
    def __init__(self, sensor_model, grid_map):
        """
        Parent class of all sensors, defining the common interface of each sensor.

        Args:
            sensor_model (SensorModel): sensor model defining sensor measurement characteristics
            grid_map (GridMap): grid map representation of environment
        """
        super(Sensor, self).__init__()

        self.sensor_model = sensor_model
        self.grid_map = grid_map
        self.sensor_simulation = None

    def set_sensor_simulation(self, sensor_simulation):
        """Simulates ground truth map and sensor measurements"""
        self.sensor_simulation = sensor_simulation

    def take_measurement(self, position: np.array, verbose: bool = True):
        raise NotImplementedError("Sensor has no measuring function implemented")

    def get_resolution_factor(self, position):
        raise NotImplementedError(
            "Sensor has no resolution factor function implemented"
        )
