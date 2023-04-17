import logging
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


class AgentActionSpace:
    def __init__(self, params: Dict):
        self.params = params
        self.spacing = params["experiment"]["constraints"]["spacing"]
        self.min_altitude = params["experiment"]["constraints"]["min_altitude"]
        self.max_altitude = params["experiment"]["constraints"]["max_altitude"]
        self.space_x_dim = 3
        self.space_y_dim = 3
        self.space_z_dim = (self.max_altitude - self.min_altitude) // self.spacing + 1
        self.num_actions = params["experiment"]["constraints"]["num_actions"]
        self.environment_x_dim = params["environment"]["x_dim"]
        self.environment_y_dim = params["environment"]["y_dim"]
        self.space_dim = np.array(
            [self.space_x_dim, self.space_y_dim, self.space_z_dim]
        )

    def get_action_mask(self, position):

        if self.num_actions == 4:
            mask = np.ones(4)
            if position[0] == 0 and position[1] == 0:
                mask = np.array([0, 0, 1, 1])
            if position[0] == 0 and 0 < position[1] < self.environment_y_dim:
                mask = np.array([0, 1, 1, 1])
            if position[0] == 0 and position[1] == self.environment_y_dim:
                mask = np.array([0, 1, 0, 1])
            if (
                0 < position[0] < self.environment_x_dim
                and position[1] == self.environment_y_dim
            ):
                mask = np.array([1, 1, 0, 1])
            if (
                position[0] == self.environment_x_dim
                and position[1] == self.environment_y_dim
            ):
                mask = np.array([1, 1, 0, 0])
            if (
                position[0] == self.environment_x_dim
                and 0 < position[1] < self.environment_y_dim
            ):
                mask = np.array([1, 1, 1, 0])
            if position[0] == self.environment_x_dim and position[1] == 0:
                mask = np.array([1, 0, 1, 0])
            if (0 < position[0] < self.environment_x_dim) and (position[1]) == 0:
                mask = np.array([1, 0, 1, 1])
            mask_flatten = mask

        elif self.num_actions == 6:
            mask = np.ones(6)
            if position[2] == self.max_altitude:
                mask[0] = 0
            if position[2] == self.min_altitude:
                mask[5] = 0
            if position[1] == 0:
                mask[2] = 0
            if position[1] == self.environment_y_dim:
                mask[3] = 0
            if position[0] == 0:
                mask[1] = 0
            if position[0] == self.environment_x_dim:
                mask[4] = 0
            mask_flatten = mask

        elif self.num_actions == 9:
            mask = np.squeeze(
                np.ones((self.space_x_dim, self.space_y_dim, self.space_z_dim))
            )
            mask[1, 1] = 0
            if position[0] == 0 and position[1] == 0:
                mask = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1]])
            if position[0] == 0 and 0 < position[1] < self.environment_y_dim:
                mask = np.array([[0, 0, 0], [1, 0, 1], [1, 1, 1]])
            if position[0] == 0 and position[1] == self.environment_y_dim:
                mask = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
            if (
                0 < position[0] < self.environment_x_dim
                and position[1] == self.environment_y_dim
            ):
                mask = np.array([[1, 1, 0], [1, 0, 0], [1, 1, 0]])
            if (
                position[0] == self.environment_x_dim
                and position[1] == self.environment_y_dim
            ):
                mask = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]])
            if (
                position[0] == self.environment_x_dim
                and 0 < position[1] < self.environment_y_dim
            ):
                mask = np.array([[1, 1, 1], [1, 0, 1], [0, 0, 0]])
            if position[0] == self.environment_x_dim and position[1] == 0:
                mask = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
            if (0 < position[0] < self.environment_x_dim) and (position[1] == 0):
                mask = np.array([[0, 1, 1], [0, 0, 1], [0, 1, 1]])

            mask_flatten = mask.flatten()

        elif self.num_actions == 27:
            mask = np.squeeze(
                np.ones((self.space_x_dim, self.space_y_dim, self.space_z_dim))
            )

            if position[0] == 0 and position[1] == 0:
                mask = np.array(
                    [
                        [[0, 0, 0], [0, 1, 1], [0, 1, 1]],
                        [[0, 0, 0], [0, 1, 1], [0, 1, 1]],
                        [[0, 0, 0], [0, 1, 1], [0, 1, 1]],
                    ]
                )
            if position[0] == 0 and 0 < position[1] < self.environment_y_dim:
                mask = np.array(
                    [
                        [[0, 0, 0], [1, 1, 1], [1, 1, 1]],
                        [[0, 0, 0], [1, 1, 1], [1, 1, 1]],
                        [[0, 0, 0], [1, 1, 1], [1, 1, 1]],
                    ]
                )
            if position[0] == 0 and position[1] == self.environment_y_dim:
                mask = np.array(
                    [
                        [[0, 0, 0], [1, 1, 0], [1, 1, 0]],
                        [[0, 0, 0], [1, 1, 0], [1, 1, 0]],
                        [[0, 0, 0], [1, 1, 0], [1, 1, 0]],
                    ]
                )
            if (
                0 < position[0] < self.environment_x_dim
                and position[1] == self.environment_y_dim
            ):
                mask = np.array(
                    [
                        [[1, 1, 0], [1, 1, 0], [1, 1, 0]],
                        [[1, 1, 0], [1, 1, 0], [1, 1, 0]],
                        [[1, 1, 0], [1, 1, 0], [1, 1, 0]],
                    ]
                )
            if (
                position[0] == self.environment_x_dim
                and position[1] == self.environment_y_dim
            ):
                mask = np.array(
                    [
                        [[1, 1, 0], [1, 1, 0], [0, 0, 0]],
                        [[1, 1, 0], [1, 1, 0], [0, 0, 0]],
                        [[1, 1, 0], [1, 1, 0], [0, 0, 0]],
                    ]
                )
            if (
                position[0] == self.environment_x_dim
                and 0 < position[1] < self.environment_y_dim
            ):
                mask = np.array(
                    [
                        [[1, 1, 1], [1, 1, 1], [0, 0, 0]],
                        [[1, 1, 1], [1, 1, 1], [0, 0, 0]],
                        [[1, 1, 1], [1, 1, 1], [0, 0, 0]],
                    ]
                )
            if position[0] == self.environment_x_dim and position[1] == 0:
                mask = np.array(
                    [
                        [[0, 1, 1], [0, 1, 1], [0, 0, 0]],
                        [[0, 1, 1], [0, 1, 1], [0, 0, 0]],
                        [[0, 1, 1], [0, 1, 1], [0, 0, 0]],
                    ]
                )
            if (0 < position[0] < self.environment_x_dim) and (position[1] == 0):
                mask = np.array(
                    [
                        [[0, 1, 1], [0, 1, 1], [0, 1, 1]],
                        [[0, 1, 1], [0, 1, 1], [0, 1, 1]],
                        [[0, 1, 1], [0, 1, 1], [0, 1, 1]],
                    ]
                )

            mask = np.transpose(mask, (1, 2, 0))

            if position[2] == self.max_altitude:
                mask[:, :, 0] = 0
            if position[2] == self.min_altitude:
                mask[:, :, 2] = 0

            mask[1, 1, 1] = 0

            mask = np.transpose(mask, (2, 0, 1))
            mask_flatten = mask.flatten()

        return mask_flatten, mask

    def action_to_position(self, position: np.array, action_index: int):
        offset = [0, 0, 0]

        if self.num_actions == 4:
            if action_index == 0:
                offset = [-self.spacing, 0, 0]
            if action_index == 1:
                offset = [0, -self.spacing, 0]
            if action_index == 2:
                offset = [0, self.spacing, 0]
            if action_index == 3:
                offset = [self.spacing, 0, 0]

        elif self.num_actions == 6:
            if action_index == 0:
                offset = [0, 0, self.spacing]
            if action_index == 1:
                offset = [-self.spacing, 0, 0]
            if action_index == 2:
                offset = [0, -self.spacing, 0]
            if action_index == 3:
                offset = [0, self.spacing, 0]
            if action_index == 4:
                offset = [self.spacing, 0, 0]
            if action_index == 5:
                offset = [0, 0, -self.spacing]

        elif self.num_actions == 9:
            if action_index == 0:
                offset = [-self.spacing, -self.spacing, 0]
            if action_index == 1:
                offset = [-self.spacing, 0, 0]
            if action_index == 2:
                offset = [-self.spacing, self.spacing, 0]
            if action_index == 3:
                offset = [0, -self.spacing, 0]
            if action_index == 4:
                offset = [
                    0,
                    0,  # mask out action of standing still to enforce agent motion
                    0,
                ]
            if action_index == 5:
                offset = [0, self.spacing, 0]
            if action_index == 6:
                offset = [self.spacing, -self.spacing, 0]
            if action_index == 7:
                offset = [self.spacing, 0, 0]
            if action_index == 8:
                offset = [self.spacing, self.spacing, 0]

        elif self.num_actions == 27:
            if action_index == 0:
                offset = [-self.spacing, -self.spacing, self.spacing]
            if action_index == 1:
                offset = [-self.spacing, 0, self.spacing]
            if action_index == 2:
                offset = [-self.spacing, self.spacing, self.spacing]
            if action_index == 3:
                offset = [0, -self.spacing, self.spacing]
            if action_index == 4:
                offset = [0, 0, self.spacing]
            if action_index == 5:
                offset = [0, self.spacing, self.spacing]
            if action_index == 6:
                offset = [self.spacing, -self.spacing, self.spacing]
            if action_index == 7:
                offset = [self.spacing, 0, self.spacing]
            if action_index == 8:
                offset = [self.spacing, self.spacing, self.spacing]
            if action_index == 9:
                offset = [-self.spacing, -self.spacing, 0]
            if action_index == 10:
                offset = [-self.spacing, 0, 0]
            if action_index == 11:
                offset = [-self.spacing, self.spacing, 0]
            if action_index == 12:
                offset = [0, -self.spacing, 0]
            if action_index == 13:
                offset = [0, 0, 0]
            if action_index == 14:
                offset = [0, self.spacing, 0]
            if action_index == 15:
                offset = [self.spacing, -self.spacing, 0]
            if action_index == 16:
                offset = [self.spacing, 0, 0]
            if action_index == 17:
                offset = [self.spacing, self.spacing, 0]
            if action_index == 18:
                offset = [-self.spacing, -self.spacing, -self.spacing]
            if action_index == 19:
                offset = [-self.spacing, 0, -self.spacing]
            if action_index == 20:
                offset = [-self.spacing, self.spacing, -self.spacing]
            if action_index == 21:
                offset = [0, -self.spacing, -self.spacing]
            if action_index == 22:
                offset = [0, 0, -self.spacing]
            if action_index == 23:
                offset = [0, self.spacing, -self.spacing]
            if action_index == 24:
                offset = [self.spacing, -self.spacing, -self.spacing]
            if action_index == 25:
                offset = [self.spacing, 0, -self.spacing]
            if action_index == 26:
                offset = [self.spacing, self.spacing, -self.spacing]

        position = position + offset

        return position

    def apply_collision_mask(
        self, position, mask, next_other_positions, agent_state_space
    ):

        for other_position in next_other_positions:
            relative_idx = agent_state_space.position_to_index(
                other_position
            ) - agent_state_space.position_to_index(position)

            if self.num_actions == 4:
                if relative_idx[0] == -1 and relative_idx[1] == 0:
                    mask[0] = 0
                if relative_idx[0] == 0 and relative_idx[1] == -1:
                    mask[1] = 0
                if relative_idx[0] == 0 and relative_idx[1] == 1:
                    mask[2] = 0
                if relative_idx[0] == 1 and relative_idx[1] == 0:
                    mask[3] = 0

            if self.num_actions == 6:
                if relative_idx[0] == 0 and relative_idx[1] == 0:
                    if np.sum(mask) > 1:
                        mask[0] = 0
                        mask[5] = 0
                if relative_idx[0] == -1 and relative_idx[1] == 0:
                    if np.sum(mask) > 1:
                        mask[1] = 0
                if relative_idx[0] == 0 and relative_idx[1] == -1:
                    if np.sum(mask) > 1:
                        mask[2] = 0
                if relative_idx[0] == 0 and relative_idx[1] == 1:
                    if np.sum(mask) > 1:
                        mask[3] = 0
                if relative_idx[0] == 1 and relative_idx[1] == 0:
                    if np.sum(mask) > 1:
                        mask[4] = 0

            elif self.num_actions == 9:
                if relative_idx[0] == -1 and relative_idx[1] == -1:
                    mask[0] = 0
                    if np.count_nonzero(mask) == 0:
                        mask[0] = 1
                if relative_idx[0] == -1 and relative_idx[1] == 0:
                    mask[1] = 0
                    if np.count_nonzero(mask) == 0:
                        mask[1] = 1
                if relative_idx[0] == -1 and relative_idx[1] == 1:
                    mask[2] = 0
                    if np.count_nonzero(mask) == 0:
                        mask[2] = 1
                if relative_idx[0] == 0 and relative_idx[1] == -1:
                    mask[3] = 0
                    if np.count_nonzero(mask) == 0:
                        mask[3] = 1
                if relative_idx[0] == 0 and relative_idx[1] == 1:
                    mask[5] = 0
                    if np.count_nonzero(mask) == 0:
                        mask[5] = 1
                if relative_idx[0] == 1 and relative_idx[1] == -1:
                    mask[6] = 0
                    if np.count_nonzero(mask) == 0:
                        mask[6] = 1
                if relative_idx[0] == 1 and relative_idx[1] == 0:
                    mask[7] = 0
                    if np.count_nonzero(mask) == 0:
                        mask[7] = 1
                if relative_idx[0] == 1 and relative_idx[1] == 1:
                    mask[8] = 0
                    if np.count_nonzero(mask) == 0:
                        mask[8] = 1

            elif self.num_actions == 27:
                if (
                    relative_idx[0] == -1
                    and relative_idx[1] == -1
                    and relative_idx[2] == 1
                ):
                    mask[0] = 0
                    mask[9] = 0
                    mask[18] = 0
                if (
                    relative_idx[0] == -1
                    and relative_idx[1] == 0
                    and relative_idx[2] == 1
                ):
                    mask[1] = 0
                    mask[10] = 0
                    mask[19] = 0
                if (
                    relative_idx[0] == -1
                    and relative_idx[1] == 1
                    and relative_idx[2] == 1
                ):
                    mask[2] = 0
                    mask[11] = 0
                    mask[20] = 0
                if (
                    relative_idx[0] == 0
                    and relative_idx[1] == -1
                    and relative_idx[2] == 1
                ):
                    mask[3] = 0
                    mask[12] = 0
                    mask[21] = 0
                if (
                    relative_idx[0] == 0
                    and relative_idx[1] == 0
                    and relative_idx[2] == 1
                ):
                    mask[4] = 0
                    mask[22] = 0
                if (
                    relative_idx[0] == 0
                    and relative_idx[1] == 1
                    and relative_idx[2] == 1
                ):
                    mask[5] = 0
                    mask[14] = 0
                    mask[23] = 0
                if (
                    relative_idx[0] == 1
                    and relative_idx[1] == -1
                    and relative_idx[2] == 1
                ):
                    mask[6] = 0
                    mask[15] = 0
                    mask[24] = 0
                if (
                    relative_idx[0] == 1
                    and relative_idx[1] == 0
                    and relative_idx[2] == 1
                ):
                    mask[7] = 0
                    mask[16] = 0
                    mask[25] = 0
                if (
                    relative_idx[0] == 1
                    and relative_idx[1] == 1
                    and relative_idx[2] == 1
                ):
                    mask[8] = 0
                    mask[17] = 0
                    mask[26] = 0
                if (
                    relative_idx[0] == -1
                    and relative_idx[1] == -1
                    and relative_idx[2] == 0
                ):
                    mask[0] = 0
                    mask[9] = 0
                    mask[18] = 0
                if (
                    relative_idx[0] == -1
                    and relative_idx[1] == 0
                    and relative_idx[2] == 0
                ):
                    mask[1] = 0
                    mask[10] = 0
                    mask[19] = 0
                if (
                    relative_idx[0] == -1
                    and relative_idx[1] == 1
                    and relative_idx[2] == 0
                ):
                    mask[2] = 0
                    mask[11] = 0
                    mask[20] = 0
                if (
                    relative_idx[0] == 0
                    and relative_idx[1] == -1
                    and relative_idx[2] == 0
                ):
                    mask[3] = 0
                    mask[12] = 0
                    mask[21] = 0
                if (
                    relative_idx[0] == 0
                    and relative_idx[1] == 1
                    and relative_idx[2] == 0
                ):
                    mask[5] = 0
                    mask[14] = 0
                    mask[23] = 0
                if (
                    relative_idx[0] == 1
                    and relative_idx[1] == -1
                    and relative_idx[2] == 0
                ):
                    mask[6] = 0
                    mask[15] = 0
                    mask[24] = 0
                if (
                    relative_idx[0] == 1
                    and relative_idx[1] == 0
                    and relative_idx[2] == 0
                ):
                    mask[7] = 0
                    mask[16] = 0
                    mask[25] = 0
                if (
                    relative_idx[0] == 1
                    and relative_idx[1] == 1
                    and relative_idx[2] == 0
                ):
                    mask[8] = 0
                    mask[17] = 0
                    mask[26] = 0

                if (
                    relative_idx[0] == -1
                    and relative_idx[1] == -1
                    and relative_idx[2] == -1
                ):
                    mask[0] = 0
                    mask[9] = 0
                    mask[18] = 0
                if (
                    relative_idx[0] == -1
                    and relative_idx[1] == 0
                    and relative_idx[2] == -1
                ):
                    mask[1] = 0
                    mask[10] = 0
                    mask[19] = 0
                if (
                    relative_idx[0] == -1
                    and relative_idx[1] == 1
                    and relative_idx[2] == -1
                ):
                    mask[2] = 0
                    mask[11] = 0
                    mask[20] = 0
                if (
                    relative_idx[0] == 0
                    and relative_idx[1] == -1
                    and relative_idx[2] == -1
                ):
                    mask[3] = 0
                    mask[12] = 0
                    mask[21] = 0
                if (
                    relative_idx[0] == 0
                    and relative_idx[1] == 0
                    and relative_idx[2] == -1
                ):
                    mask[4] = 0
                    mask[22] = 0
                if (
                    relative_idx[0] == 0
                    and relative_idx[1] == 1
                    and relative_idx[2] == -1
                ):
                    mask[5] = 0
                    mask[14] = 0
                    mask[23] = 0
                if (
                    relative_idx[0] == 1
                    and relative_idx[1] == -1
                    and relative_idx[2] == -1
                ):
                    mask[6] = 0
                    mask[15] = 0
                    mask[24] = 0
                if (
                    relative_idx[0] == 1
                    and relative_idx[1] == 0
                    and relative_idx[2] == -1
                ):
                    mask[7] = 0
                    mask[16] = 0
                    mask[25] = 0
                if (
                    relative_idx[0] == 1
                    and relative_idx[1] == 1
                    and relative_idx[2] == -1
                ):
                    mask[8] = 0
                    mask[17] = 0
                    mask[26] = 0

        return mask
