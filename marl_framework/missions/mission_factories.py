import logging
from typing import Dict

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from marl_framework.constants import MISSION_TYPES, MissionType, LOG_DIR
from marl_framework.missions.coma_mission import COMAMission
from marl_framework.missions.missions import Mission

logger = logging.getLogger(__name__)


class MissionFactory:
    def __init__(self, params: Dict):
        self.params = params
        self.writer = SummaryWriter(LOG_DIR)

    @property
    def mission_type(self) -> str:
        if "missions" not in self.params["experiment"].keys():
            logger.error("Cannot find mission specification in config file!")
            raise ValueError

        if "type" not in self.params["experiment"]["missions"].keys():
            logger.error("Cannot find mission type specification in config file!")
            raise ValueError

        return self.params["experiment"]["missions"]["type"]

    def create_mission(self) -> Mission:
        if self.mission_type not in MISSION_TYPES:
            logger.error(
                f"'{self.mission_type}' not in list of known mission types: {MISSION_TYPES}"
            )
            raise ValueError

        if (
            self.mission_type == MissionType.COMA
            or self.mission_type == MissionType.DeepQ
            or self.mission_type == MissionType.CentralQV
        ):
            return COMAMission(self.params, self.writer, -np.inf)
        else:
            raise ValueError("Mission not implemented!")
