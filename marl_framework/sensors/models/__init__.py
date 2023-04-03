class SensorModel:
    def __init__(self):
        super(SensorModel, self).__init__()

    def get_noise_variance(self, altitude) -> float:
        raise NotImplementedError("Sensor has no noise variance function implemented")
