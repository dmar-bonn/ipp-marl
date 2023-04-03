import logging
from typing import Dict
import torch
from torch import nn


logger = logging.getLogger(__name__)


class ActorNetwork(nn.Module):
    def __init__(self, params: Dict):
        super(ActorNetwork, self).__init__()
        self.params = params
        self.n_agents = params["mission"]["n_agents"]
        self.mission_type = params["mission"]["type"]
        self.hidden_dim = params["networks"]["actor"]["hidden_dim"]
        self.n_actions = params["MARL_cast"]["action_space"]["num_actions"]
        self.eps_max = params["exploration"]["eps_max"]
        self.eps_min = params["exploration"]["eps_min"]
        self.eps_anneal_phase = params["exploration"]["eps_anneal_phase"]
        self.use_eps = params["exploration"]["use_eps"]

        self.conv1 = nn.Conv2d(7, self.hidden_dim, (5, 5))
        self.conv2 = nn.Conv2d(self.hidden_dim, self.hidden_dim, (4, 4))
        self.conv3 = nn.Conv2d(self.hidden_dim, self.hidden_dim, (4, 4))
        self.activation = torch.nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.n_actions)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.softmax = nn.Softmax(dim=1)

    def get_action_index(
        self, batch_memory, action_mask_1d, agent_id, num_episode: int, mode: str
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

        action_probs = torch.squeeze(action_probs) * action_mask_1d
        action_index_chosen = self.do_eps_exploration(action_probs, mode)

        return action_probs, action_index_chosen, action_mask_1d, eps

    def forward(self, input_state, eps):
        if input_state.dim() == 3:
            input_state = torch.permute(input_state, (2, 0, 1))
        elif input_state.dim() == 4:
            input_state = torch.permute(input_state, (0, 3, 1, 2))

        output = self.activation(self.conv1(input_state))
        output = self.activation(self.conv2(output))
        output = self.activation(self.conv3(output))
        h = self.flatten(output)
        output = self.activation(self.fc1(h))
        output = self.fc2(output)
        probs = self.softmax(output)

        final = (1 - eps) * probs
        final = final + eps / self.n_actions
        return final, h

    def do_eps_exploration(self, action_probs, mode):

        if mode == "eval":
            action_index_chosen = torch.argmax(action_probs)
        else:
            action_index_chosen = torch.multinomial(action_probs, 1, replacement=True)

        return action_index_chosen
