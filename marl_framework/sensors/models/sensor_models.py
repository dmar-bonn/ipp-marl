import logging
import numpy as np

from typing import Dict

logger = logging.getLogger(__name__)


class AltitudeSensorModel:
    def __init__(self, params: Dict):
        self.params = params
        self.coeff_a = self.params["sensor"]["model"]["coeff_a"]
        self.coeff_b = self.params["sensor"]["model"]["coeff_b"]

    def get_noise_variance(self, altitude) -> float:
        """Returns sensor measurement noise scaling with altitude"""
        noise = 0
        if altitude == 5:
            noise = 0.01
        if altitude == 10:
            noise = 0.265
        if altitude == 15:
            noise = 0.375
        return noise   # self.coeff_a * (1 - np.exp(-self.coeff_b * altitude))   # float(noise)
