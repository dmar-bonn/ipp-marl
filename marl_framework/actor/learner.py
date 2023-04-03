import logging
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
from scipy.special import rel_entr
from scipy.stats import entropy
from utils.utils import clip_gradients
import torch.nn as nn

logger = logging.getLogger(__name__)


class ActorLearner:
    def __init__(self, params: Dict, writer, actor, agent_state_space):
        self.actor = actor
        self.agent_state_space = agent_state_space
        self.params = params
        self.n_actions = params["experiment"]["constraints"]["num_actions"]
        self.writer = writer
        self.device = torch.device("cpu")
        self.actor.to(self.device)
        self.batch_size = params["networks"]["batch_size"]
        self.n_agents = self.params["experiment"]["missions"]["n_agents"]
        self.budget = params["experiment"]["constraints"]["budget"]
        self.lr = params["networks"]["actor"]["learning_rate"]
        self.momentum = params["networks"]["actor"]["momentum"]
        self.gradient_norm = params["networks"]["actor"]["gradient_norm"]
        self.optimizer = torch.optim.Adam(self.actor.parameters(),
                                          lr=self.lr)  # optim.RMSprop(self.actor.parameters(), self.lr, self.momentum)
        self.optimizer.zero_grad()
        self.kl_loss = nn.KLDivLoss()

    def learn(self, batches, q_values, eps):
        advantages = []
        log_probs_chosen = []
        log_probs_all = []
        losses = []
        hidden_state_differences = []
        for i, batch in enumerate(batches):
            batch_observations = []
            batch_actions = []
            batch_masks = []
            batch_q_values = q_values[i]
            for batch_idx in batch:
                batch_observations.append(batch_idx.observation.float())
                batch_actions.append(batch_idx.action)
                batch_masks.append(batch_idx.mask)

            batch_probs, batch_hidden_states = self.actor.forward(               #
                torch.stack(batch_observations).squeeze().to(self.device), eps
            )
            batch_log_probs = torch.log(batch_probs)
            with torch.no_grad():
                batch_probs = batch_probs * torch.stack(batch_masks)               #
                batch_sum = batch_probs.sum(-1).unsqueeze(1).repeat(1, self.n_actions)
                batch_sum[batch_sum < 0.00001] = 0.00001
                batch_probs_norm = batch_probs/batch_sum
                batch_probs_norm[batch_probs_norm <= 0.00001] = 0.00001
                batch_log_probs_norm = torch.log(batch_probs_norm)

            hidden_state_differences.append(
                np.mean(
                    np.square(
                        batch_hidden_states[0].detach().cpu().numpy(),
                        batch_hidden_states[1].detach().cpu().numpy(),
                    )
                )
            )

            batch_q_chosen = torch.gather(
                batch_q_values, dim=1, index=torch.stack(batch_actions)
            )
            baseline = (torch.exp(batch_log_probs_norm) * batch_q_values * torch.stack(batch_masks))
            baseline_sum = baseline.sum(-1)

            batch_advantages = batch_q_chosen - baseline_sum.unsqueeze(1)        ##############S
            batch_log_probs_chosen = torch.gather(
                batch_log_probs, dim=1, index=torch.stack(batch_actions)
            )
            advantages.append(batch_advantages)
            log_probs_chosen.append(batch_log_probs_chosen)
            log_probs_all.append(batch_log_probs)

            loss = -(batch_advantages.detach() * batch_log_probs_chosen * torch.stack(batch_masks)).mean()
            losses.append(loss)

            self.optimizer.zero_grad()
            loss.backward()
            # clip_gradients(self.actor, self.gradient_norm)
            self.optimizer.step()

        kl_divergence = self.calculate_kl_divergence(
            batches, torch.stack(log_probs_all), eps
        )

        conv1_grad_norm = 0
        conv2_grad_norm = 0
        conv3_grad_norm = 0
        fc1_grad_norm = 0
        fc2_grad_norm = 0
        fc3_grad_norm = 0
        conv1_count = 0
        conv2_count = 0
        conv3_count = 0
        fc1_count = 0
        fc2_count = 0
        fc3_count = 0
        for params in self.actor.conv1.parameters():
            if params.grad is not None:
                conv1_grad_norm += params.grad.data.norm(1).item()
                conv1_count += 1
        for params in self.actor.conv2.parameters():
            if params.grad is not None:
                conv2_grad_norm += params.grad.data.norm(1).item()
                conv2_count += 1
        for params in self.actor.conv3.parameters():
            if params.grad is not None:
                conv3_grad_norm += params.grad.data.norm(1).item()
                conv3_count += 1
        for params in self.actor.fc1.parameters():
            if params.grad is not None:
                fc1_grad_norm += params.grad.data.norm(1).item()
                fc1_count += 1
        for params in self.actor.fc2.parameters():
            if params.grad is not None:
                fc2_grad_norm += params.grad.data.norm(1).item()
                fc2_count += 1
        for params in self.actor.fc3.parameters():
            if params.grad is not None:
                fc3_grad_norm += params.grad.data.norm(1).item()
                fc3_count += 1

        return (
            self.actor,
            [
                torch.mean(torch.stack(losses)),
                torch.mean(torch.stack(advantages)),
                torch.std(torch.stack(advantages)),
                torch.mean(torch.stack(log_probs_chosen)),
                np.mean(entropy(torch.exp(torch.stack(log_probs_all)).detach().cpu().numpy(), axis=2)),
                np.mean(kl_divergence),
                np.mean(hidden_state_differences),
                [conv1_grad_norm / conv1_count, conv2_grad_norm / conv2_count, conv3_grad_norm / conv3_count,
                 fc1_grad_norm / fc1_count, 0, fc3_grad_norm / fc3_count]
            ],
        )

    def calculate_kl_divergence(self, batches, log_probs_all, eps):
        log_probs_new = []
        for i, batch in enumerate(batches):
            batch_observations = []
            for batch_idx in batch:
                batch_observations.append(batch_idx.observation.float())
            with torch.no_grad():
                batch_log_probs, _ = self.actor.forward(
                    torch.stack(batch_observations).squeeze().to(self.device), eps,
                )
            log_probs_new.append(batch_log_probs)
        kl_divergence = sum(
            rel_entr(
                torch.exp(log_probs_all).detach().cpu().numpy(),
                torch.exp(torch.stack(log_probs_new)).detach().cpu().numpy(),
            )
        )
        return kl_divergence
