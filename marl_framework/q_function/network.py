import logging
from typing import Dict

import torch
from torch import nn

logger = logging.getLogger(__name__)

torch.autograd.set_detect_anomaly(True)


class QNetwork(nn.Module):
    def __init__(self, params: Dict):
        super(QNetwork, self).__init__()
        self.params = params
        self.n_agents = params["missions"]["n_agents"]
        self.n_actions = params["MARL_cast"]["action_space"]["num_actions"]
        self.hidden_dim = params["networks"]["critic"]["hidden_dim"]
        self.conv1 = nn.Conv2d(12, self.hidden_dim, (5, 5))
        self.conv2 = nn.Conv2d(self.hidden_dim, self.hidden_dim, (4, 4))
        self.conv3 = nn.Conv2d(self.hidden_dim, self.hidden_dim, (4, 4))
        self.flatten = nn.Flatten()
        self.activation = torch.nn.ReLU()
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.n_actions)
        self.log_softmax = nn.LogSoftmax(dim=0)

    def forward(self, input_state):
        if input_state.dim() == 3:
            input_state = torch.permute(input_state, (2, 0, 1))
        elif input_state.dim() == 4:
            input_state = torch.permute(input_state, (0, 3, 1, 2))

        output = self.activation(self.conv1(input_state))
        output = self.activation(self.conv2(output))
        output = self.activation(self.conv3(output))
        h = torch.squeeze(self.flatten(output))
        output = self.activation(self.fc1(h))
        output = self.fc2(output)

        with torch.no_grad():
            log_probs = self.log_softmax(output)

        return output, log_probs
