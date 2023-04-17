from typing import Dict
from torch.utils.tensorboard import SummaryWriter


class Mission:
    def __init__(
        self, params: Dict, writer: SummaryWriter, max_mean_episode_return: float = -100
    ):
        super(Mission, self).__init__()

        self.params = params
        self.writer = writer
        self.max_mean_episode_return = max_mean_episode_return

    def execute(self):
        raise NotImplementedError(
            "Planning mission does not implement 'execute' function!"
        )
