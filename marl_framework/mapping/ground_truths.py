import math
from typing import List
import logging
import numpy as np

logger = logging.getLogger(__name__)


def fft_indices(n) -> List:
    a = list(range(0, math.floor(n / 2) + 1))
    b = reversed(range(1, math.floor(n / 2)))
    b = [-i for i in b]
    return a + b


def gaussian_random_field(pk, x_dim: int, y_dim: int, episode: int) -> np.array:
    """Generate 2D gaussian random field: https://andrewwalker.github.io/statefultransitions/post/gaussian-fields/"""

    def pk2(kx, ky):
        if kx == 0 and ky == 0:
            return 0.0

        return np.sqrt(pk(np.sqrt(kx ** 2 + ky ** 2)))

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
    environment_type_idx = 0
    split_idx = np.random.randint(4)

    if environment_type_idx == 0:
        percentage_idx = np.random.randint(30, 61)
        if split_idx == 0:
            field[: int((y_dim * percentage_idx) / 100), :] = 1
        elif split_idx == 1:
            field[int((y_dim * (1 - percentage_idx)) / 100) :, :] = 1
        elif split_idx == 2:
            field[:, : int((x_dim * percentage_idx) / 100)] = 1
        elif split_idx == 3:
            field[:, int((x_dim * (1 - percentage_idx)) / 100) :] = 1

    elif environment_type_idx == 1:
        percentage_idx_1 = np.random.randint(55, 75)
        percentage_idx_2 = np.random.randint(55, 75)
        if split_idx == 0:
            field[
                : int((y_dim * percentage_idx_1) / 100),
                : int((x_dim * percentage_idx_2) / 100),
            ] = 1
        elif split_idx == 1:
            field[
                int((y_dim * (1 - percentage_idx_1)) / 100) :,
                : int((x_dim * percentage_idx_2) / 100),
            ] = 1
        elif split_idx == 2:
            field[
                int((y_dim * (1 - percentage_idx_1)) / 100) :,
                int((x_dim * (1 - percentage_idx_2)) / 100) :,
            ] = 1
        elif split_idx == 3:
            field[
                : int((y_dim * percentage_idx_1) / 100),
                int((x_dim * (1 - percentage_idx_2)) / 100) :,
            ] = 1

    elif environment_type_idx == 2:
        percentage_idx_1 = np.random.randint(20, 35)
        percentage_idx_2 = np.random.randint(65, 80)
        if split_idx == 0:
            field[
                int((y_dim * percentage_idx_1) / 100) : int(
                    (y_dim * percentage_idx_2) / 100
                ),
                :,
            ] = 1
        elif split_idx == 1:
            field = np.ones_like(field)
            field[
                int((y_dim * percentage_idx_1) / 100) : int(
                    (y_dim * percentage_idx_2) / 100
                ),
                :,
            ] = 0
        elif split_idx == 2:
            field[
                :,
                int((x_dim * percentage_idx_1) / 100) : int(
                    (x_dim * percentage_idx_2) / 100
                ),
            ] = 1
        elif split_idx == 3:
            field = np.ones_like(field)
            field[
                :,
                int((x_dim * percentage_idx_1) / 100) : int(
                    (x_dim * percentage_idx_2) / 100
                ),
            ] = 0

    elif environment_type_idx == 3:
        percentage_idx = np.random.randint(40, 55)
        if split_idx == 0 or split_idx == 2:
            field[
                : int((y_dim * percentage_idx) / 100),
                : int((x_dim * percentage_idx) / 100),
            ] = 1
            field[
                int((y_dim * (1 - percentage_idx)) / 100) :,
                int((x_dim * (1 - percentage_idx)) / 100) :,
            ] = 1
        elif split_idx == 1 or split_idx == 3:
            field[
                int((y_dim * (1 - percentage_idx)) / 100) :,
                : int((x_dim * percentage_idx) / 100),
            ] = 1
            field[
                : int((y_dim * percentage_idx) / 100),
                int((x_dim * (1 - percentage_idx)) / 100) :,
            ] = 1

    else:
        if split_idx == 0 or split_idx == 2:
            percentage_idx = np.random.randint(28, 38)
            field[
                : int((y_dim * percentage_idx) / 100),
                : int((x_dim * percentage_idx) / 100),
            ] = 1
            field[
                int((y_dim * (1 - percentage_idx)) / 100) :,
                : int((x_dim * percentage_idx) / 100),
            ] = 1
            field[
                int((y_dim * (1 - percentage_idx)) / 100) :,
                int((x_dim * (1 - percentage_idx)) / 100) :,
            ] = 1
            field[
                : int((y_dim * percentage_idx) / 100),
                int((x_dim * (1 - percentage_idx)) / 100) :,
            ] = 1
        if split_idx == 1 or split_idx == 3:
            field = np.ones_like(field)
            percentage_idx = np.random.randint(32, 42)
            field[
                : int((y_dim * percentage_idx) / 100),
                : int((x_dim * percentage_idx) / 100),
            ] = 0
            field[
                int((y_dim * (1 - percentage_idx)) / 100) :,
                : int((x_dim * percentage_idx) / 100),
            ] = 0
            field[
                int((y_dim * (1 - percentage_idx)) / 100) :,
                int((x_dim * (1 - percentage_idx)) / 100) :,
            ] = 0
            field[
                : int((y_dim * percentage_idx) / 100),
                int((x_dim * (1 - percentage_idx)) / 100) :,
            ] = 0

    return field
