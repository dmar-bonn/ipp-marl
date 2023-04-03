import math
import os
from typing import List
import logging
import numpy as np
from matplotlib import pyplot as plt
import imageio
import cv2

logger = logging.getLogger(__name__)


def fft_indices(n) -> List:
    a = list(range(0, math.floor(n / 2) + 1))
    b = reversed(range(1, math.floor(n / 2)))
    b = [-i for i in b]
    return a + b


def gaussian_random_field(pk, x_dim: int, y_dim: int, episode: int) -> np.array:
    """Generate 2D gaussian random field: https://andrewwalker.github.io/statefultransitions/post/gaussian-fields/"""

    TEMPERATURE_FIELD_PATH = "/home/penguin2/Downloads/FLIR_ICOS_100m_15cm_test_ground_truth.png"
    MAP_X_DIM, MAP_Y_DIM = 493, 493

    def pk2(kx, ky):
        if kx == 0 and ky == 0:
            return 0.0

        return np.sqrt(pk(np.sqrt(kx ** 2 + ky ** 2)))

    def load_raw_data(filepath: str) -> np.array:
        file_path = os.path.join(filepath)
        if not os.path.exists(file_path):
            raise ValueError(f"Cannot find temperature ground truth data! File {file_path} does not exist!")

        return imageio.v2.imread(file_path)

    def rgba_to_temperature(rbga_map: np.array) -> np.array:
        """Maps rgba data map to real temperature value map"""
        return -1 * (rbga_map[:, :, 0] - rbga_map[:, :, 2])

    def normalize_temperature_map(temperature_map: np.array) -> np.array:
        """min-max normalizes temperature map values between 0 and 1"""
        min_temp = np.min(temperature_map)
        max_temp = np.max(temperature_map)

        if min_temp == max_temp:
            return temperature_map / max_temp

        return (temperature_map - min_temp) / (max_temp - min_temp)

    def create_ground_truth_map(raw_data_filepath: str) -> np.array:
        raw_data = load_raw_data(raw_data_filepath)
        temperature_map = rgba_to_temperature(raw_data)
        normalized_temperature_map = normalize_temperature_map(temperature_map)

        downsampled_temperature_map = cv2.resize(
            normalized_temperature_map,
            dsize=(MAP_Y_DIM, MAP_X_DIM),
            interpolation=cv2.INTER_AREA,
            # TODO: to enlarging image, use either INTER_LINEAR or INTER_CUBIC, to shrink image use INTER_AREA
        )
        normalized_temperature_map = normalize_temperature_map(downsampled_temperature_map)

        normalized_temperature_map[normalized_temperature_map <= 0.5] = 0
        normalized_temperature_map[normalized_temperature_map > 0.5] = 1

        return normalized_temperature_map




    np.random.seed(episode)
    noise = np.fft.fft2(np.random.normal(size=(y_dim, x_dim)))
    amplitude = np.zeros((y_dim, x_dim))

    for i, kx in enumerate(fft_indices(y_dim)):
        for j, ky in enumerate(fft_indices(x_dim)):
            amplitude[i, j] = pk2(kx, ky)

    random_field = np.fft.ifft2(noise * amplitude).real
    normalized_random_field = (random_field - np.min(random_field)) / (
            np.max(random_field) - np.min(random_field)
    )

    # Make field binary
    normalized_random_field[normalized_random_field >= 0.5] = 1
    normalized_random_field[normalized_random_field < 0.5] = 0

    field = np.zeros((y_dim, x_dim))
    np.random.seed(episode)
    environment_type_idx = 0 # np.random.randint(4)
    split_idx = np.random.randint(4)

    if environment_type_idx == 0:
        percentage_idx = np.random.randint(30, 61)
        if split_idx == 0:
            field[:int((y_dim * percentage_idx) / 100), :] = 1
        elif split_idx == 1:
            field[int((y_dim * (1 - percentage_idx)) / 100):, :] = 1
        elif split_idx == 2:
            field[:, :int((x_dim * percentage_idx) / 100)] = 1
        elif split_idx == 3:
            field[:, int((x_dim * (1 - percentage_idx)) / 100):] = 1

    elif environment_type_idx == 1:
        percentage_idx_1 = np.random.randint(55, 75)
        percentage_idx_2 = np.random.randint(55, 75)
        if split_idx == 0:
            field[:int((y_dim * percentage_idx_1) / 100), :int((x_dim * percentage_idx_2) / 100)] = 1
        elif split_idx == 1:
            field[int((y_dim * (1 - percentage_idx_1)) / 100):, :int((x_dim * percentage_idx_2) / 100)] = 1
        elif split_idx == 2:
            field[int((y_dim * (1 - percentage_idx_1)) / 100):, int((x_dim * (1 - percentage_idx_2)) / 100):] = 1
        elif split_idx == 3:
            field[:int((y_dim * percentage_idx_1) / 100), int((x_dim * (1 - percentage_idx_2)) / 100):] = 1

    elif environment_type_idx == 2:
        percentage_idx_1 = np.random.randint(20, 35)
        percentage_idx_2 = np.random.randint(65, 80)
        if split_idx == 0:
            field[int((y_dim * percentage_idx_1) / 100):int((y_dim * percentage_idx_2) / 100), :] = 1
        elif split_idx == 1:
            field = np.ones_like(field)
            field[int((y_dim * percentage_idx_1) / 100):int((y_dim * percentage_idx_2) / 100), :] = 0
        elif split_idx == 2:
            field[:, int((x_dim * percentage_idx_1) / 100):int((x_dim * percentage_idx_2) / 100)] = 1
        elif split_idx == 3:
            field = np.ones_like(field)
            field[:, int((x_dim * percentage_idx_1) / 100):int((x_dim * percentage_idx_2) / 100)] = 0

    elif environment_type_idx == 3:
        percentage_idx = np.random.randint(40, 55)
        if split_idx == 0 or split_idx == 2:
            field[:int((y_dim * percentage_idx) / 100), :int((x_dim * percentage_idx) / 100)] = 1
            field[int((y_dim * (1 - percentage_idx)) / 100):, int((x_dim * (1 - percentage_idx)) / 100):] = 1
        elif split_idx == 1 or split_idx == 3:
            field[int((y_dim * (1 - percentage_idx)) / 100):, :int((x_dim * percentage_idx) / 100)] = 1
            field[:int((y_dim * percentage_idx) / 100), int((x_dim * (1 - percentage_idx)) / 100):] = 1

    else:
        if split_idx == 0 or split_idx == 2:
            percentage_idx = np.random.randint(28, 38)
            field[:int((y_dim * percentage_idx) / 100), :int((x_dim * percentage_idx) / 100)] = 1
            field[int((y_dim * (1 - percentage_idx)) / 100):, :int((x_dim * percentage_idx) / 100)] = 1
            field[int((y_dim * (1 - percentage_idx)) / 100):, int((x_dim * (1 - percentage_idx)) / 100):] = 1
            field[:int((y_dim * percentage_idx) / 100), int((x_dim * (1 - percentage_idx)) / 100):] = 1
        if split_idx == 1 or split_idx == 3:
            field = np.ones_like(field)
            percentage_idx = np.random.randint(32, 42)
            field[:int((y_dim * percentage_idx) / 100), :int((x_dim * percentage_idx) / 100)] = 0
            field[int((y_dim * (1 - percentage_idx)) / 100):, :int((x_dim * percentage_idx) / 100)] = 0
            field[int((y_dim * (1 - percentage_idx)) / 100):, int((x_dim * (1 - percentage_idx)) / 100):] = 0
            field[:int((y_dim * percentage_idx) / 100), int((x_dim * (1 - percentage_idx)) / 100):] = 0

    # plt.imshow(field)
    # plt.title(f"type_{environment_type_idx}")
    # plt.clim(0, 1)
    # plt.savefig(f"/home/penguin2/Documents/plots/{episode}_groundtruth.png")


    # if environment_type_idx == 0:
    #     return normalized_random_field
    # else:
    #     return field

    temperature_map = create_ground_truth_map(TEMPERATURE_FIELD_PATH)

    # plt.imshow(temperature_map, cmap="plasma", vmin=0, vmax=1)
    # plt.colorbar()
    # plt.show()

    return field     # temperature_map
