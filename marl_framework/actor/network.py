import logging
from typing import Dict

import numpy as np
import torch
from torch import nn


logger = logging.getLogger(__name__)


class ActorNetwork(nn.Module):
    def __init__(self, params: Dict):
        super(ActorNetwork, self).__init__()
        self.params = params
        self.n_agents = self.params["experiment"]["missions"]["n_agents"]
        self.mission_type = self.params["experiment"]["missions"]["type"]
        self.hidden_dim = self.params["networks"]["actor"]["hidden_dim"]
        self.n_actions = self.params["experiment"]["constraints"]["num_actions"]

        self.conv1 = nn.Conv2d(7, 256, (5, 5))
        self.conv2 = nn.Conv2d(256, 256, (4, 4))
        self.conv3 = nn.Conv2d(256, 256, (4, 4))
        # self.conv4 = nn.Conv2d(128, 256, (3, 3))
        self.activation = torch.nn.ReLU()
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, self.n_actions)

        self.device = torch.device("cpu")  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.hidden_states = [[]] * self.n_agents
        self.eps_max = params["experiment"]["missions"]["eps_max"]
        self.eps_min = params["experiment"]["missions"]["eps_min"]
        self.eps_anneal_phase = params["experiment"]["missions"]["eps_anneal_phase"]
        self.use_eps = params["experiment"]["missions"]["use_eps"]
        self.network_type = self.params["networks"]["type"]
        self.baseline = "no"

    def get_action_index(
        self, batch_memory, action_mask_1d, agent_id, t, num_episode: int, mode: str
    ):
        self.train()
        if mode == "eval":
            self.eval()

        input_state = torch.unsqueeze(
            batch_memory.get(-1, agent_id, "observation"), 0
        ).to(self.device)
        action_mask_1d = torch.tensor(action_mask_1d).to(self.device)

        if num_episode > self.eps_anneal_phase:
            eps = self.eps_min
        else:
            eps = self.eps_max - num_episode / self.eps_anneal_phase * (
                self.eps_max - self.eps_min
            )

        with torch.no_grad():
            action_probs, _ = self.forward(input_state.float(), eps)

        # action_probs = torch.squeeze(torch.exp(log_action_probs)) * action_mask_1d

        edge_penalty = 0
        if action_mask_1d[torch.argmax(action_probs)] == 0:
            edge_penalty = 1

        action_probs = torch.squeeze(action_probs) * action_mask_1d

        # if torch.sum(action_probs) >= 0.00001:
        #     action_probs = action_probs / action_probs.sum()

        action_index_chosen = self.do_eps_exploration(num_episode, action_probs, action_mask_1d, mode, eps)

        return action_probs, action_index_chosen, action_mask_1d, eps

    def forward(self, input_state, eps):
        # print(f"input state size: {input_state.size()}")
        # input_state[:, :, :, 6] = 0

        if input_state.dim() == 3:
            input_state = torch.permute(input_state, (2, 0, 1))
            # input_state = torch.unsqueeze(input_state, 1)
        elif input_state.dim() == 4:
            input_state = torch.permute(input_state, (0, 3, 1, 2))


        output = self.activation(self.conv1(input_state))
        output = self.activation(self.conv2(output))
        output = self.activation(self.conv3(output))
        # output = self.activation(self.conv4(output))
        h = self.flatten(output)
        output = self.activation(self.fc1(h))
        # output = self.activation(self.fc2(output))
        output = self.fc3(output)
        # log_probs = self.log_softmax(output)
        probs = self.softmax(output)

        ### PLOT HISTOGRAM

        final = (1 - eps) * probs
        final = final + eps / self.n_actions
        return final, h

    def do_eps_exploration(self, num_episode, action_probs, action_mask, mode, eps):
        # if num_episode > self.eps_anneal_phase:
        #     eps = self.eps_min
        # else:
        #     eps = self.eps_max - num_episode / self.eps_anneal_phase * (
        #         self.eps_max - self.eps_min
        #     )
        # index_found = False
        # if np.random.random_sample() < eps:
        #     index = None
        #     while not index_found:
        #         index = np.random.randint(self.n_actions)
        #         if action_mask[index] == 1:
        #             index_found = True
        #     action_index_chosen = torch.tensor([index]).to(self.device)
        # #     else:
        # #         logger.info("NO VALID PROBABILITY DISTRIBUTION!")
        #
        # else:
        #     if torch.sum(action_probs) < 0.00001:
        #         index = None
        #         while not index_found:
        #             index = np.random.randint(self.n_actions)
        #             if action_mask[index] == 1:
        #                 index_found = True
        #         action_index_chosen = torch.tensor([index]).to(self.device)
        #         # logger.info(f"action mask: {action_mask}")
        #         # logger.info(f"action_index_chosen: {action_index_chosen}")
        #     else:
        #         action_index_chosen = torch.multinomial(action_probs, 1, replacement=True)

        # if num_episode > self.eps_anneal_phase:
        #     eps = self.eps_min
        # else:
        #     eps = self.eps_max - num_episode / self.eps_anneal_phase * (
        #         self.eps_max - self.eps_min
        #     )


        # action_probs = (1 - eps) * action_probs + eps

        # print(f"train: action_probs {action_probs}")
        # action_probs = action_probs * action_mask
        # print(f"train: action_probs masked {action_probs}")
        if mode == "eval":
            # if torch.sum(action_probs) >= 0.00001:
            action_index_chosen = torch.argmax(action_probs)
            # else:
            # index = np.randint(-1, 1)
            # action_index_chosen = torch.tensor([(action_mask == 1).nonzero(as_tuple=True)[0][index]]).to(self.device)
        else:
            action_index_chosen = torch.multinomial(action_probs, 1, replacement=True)

        return action_index_chosen
