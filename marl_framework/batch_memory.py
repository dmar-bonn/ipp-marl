import logging
from typing import Dict, List

import numpy as np
import torch

from utils.utils import TransitionCOMA

logger = logging.getLogger(__name__)


class BatchMemory:
    def __init__(self, params: Dict, coma_network):
        self.params = params
        self.coma_network = coma_network
        self.initialized = "no"
        self.batch_size = self.params["networks"]["batch_size"]
        self.n_agents = self.params["experiment"]["missions"]["n_agents"]
        self.budget = params["experiment"]["constraints"]["budget"]
        self.gamma = params["networks"]["gamma"]
        self.lam = params["networks"]["lambda"]
        self.network_type = self.params["networks"]["type"]
        self.transitions = {agent_id: [] for agent_id in range(self.n_agents)}

    def clear(self):
        self.transitions = {agent_id: [] for agent_id in range(self.n_agents)}

    def add(
        self,
        agent_id: int,
        state=None,
        observation=None,
        action=None,
        mask=None,
        reward=None,
        done=None,
        td_target=None,
        discounted_return=None,
    ):
        self.transitions[agent_id].append(
            TransitionCOMA(
                state,
                observation,
                action,
                mask,
                reward,
                done,
                td_target,
                discounted_return,
            )
        )

    def insert(
        self,
        t: int,
        agent_id: int,
        state=None,
        observation=None,
        action=None,
        mask=None,
        reward=None,
        done=None,
        td_target=None,
        discounted_return=None,
    ):
        if state is not None:
            self.transitions[agent_id][t] = self.transitions[agent_id][t]._replace(
                state=state
            )
        if observation is not None:
            self.transitions[agent_id][t] = self.transitions[agent_id][t]._replace(
                observation=observation
            )
        if action is not None:
            self.transitions[agent_id][t] = self.transitions[agent_id][t]._replace(
                action=action
            )
        if mask is not None:
            self.transitions[agent_id][t] = self.transitions[agent_id][t]._replace(
                mask=mask
            )
        if reward is not None:
            self.transitions[agent_id][t] = self.transitions[agent_id][t]._replace(
                reward=reward
            )
        if done is not None:
            self.transitions[agent_id][t] = self.transitions[agent_id][t]._replace(
                done=done
            )
        if td_target is not None:
            self.transitions[agent_id][t] = self.transitions[agent_id][t]._replace(
                td_target=td_target
            )
        if discounted_return is not None:
            self.transitions[agent_id][t] = self.transitions[agent_id][t]._replace(
                discounted_return=discounted_return
            )

    def get(self, t: int, agent_id: int, argument: str):
        if argument == "state":
            return self.transitions[agent_id][t].state
        if argument == "observation":
            return self.transitions[agent_id][t].observation
        if argument == "action":
            return self.transitions[agent_id][t].action
        if argument == "mask":
            return self.transitions[agent_id][t].masks
        if argument == "reward":
            return self.transitions[agent_id][t].reward
        if argument == "done":
            return self.transitions[agent_id][t].done
        if argument == "td_target":
            return self.transitions[agent_id][t].td_target
        if argument == "discounted_return":
            return self.transitions[agent_id][t].discounted_return

    def size(self):
        return len(self.transitions[0]) * self.n_agents

    def build_td_targets(self, target_critic_network):
        for agent_id in range(self.n_agents):
            for t in range(len(self.transitions[agent_id])):
                sum_n_step_returns = torch.tensor([0.0])
                for n in range(1, len(self.transitions[agent_id]) - t + 1):
                    leave = False
                    n_step_return = 0
                    discounted_return = torch.tensor([0.0])
                    for l in range(0, n):
                        if (not self.get(t + l - 1, agent_id, "done")) or (t + l == 0):
                            n_step_return += np.power(self.gamma, l) * self.get(
                                t + l, agent_id, "reward"
                            )
                            discounted_return += np.power(self.gamma, l) * self.get(
                                t + l, agent_id, "reward"
                            )
                        else:
                            leave = True
                            break
                    if leave:
                        sum_n_step_returns += np.power(self.lam, n) * n_step_return
                        break
                    if t + n >= len(self.transitions[agent_id]):
                        pass
                    else:
                        if not (
                            (self.get(t + n, agent_id, "done"))
                            or (t + n + 1 >= len(self.transitions[agent_id]))
                        ):
                            with torch.no_grad():
                                target_q_values, _ = target_critic_network.forward(
                                    self.get(t + n, agent_id, "state")
                                )
                                target_q_value = target_q_values[
                                    self.get(t + n, agent_id, "action").to("cpu")
                                ]

                            n_step_return += np.power(self.gamma, n) * target_q_value

                    sum_n_step_returns += np.power(self.lam, n - 1) * n_step_return

                self.insert(t, agent_id, td_target=(1 - self.lam) * sum_n_step_returns)
                self.insert(t, agent_id, discounted_return=discounted_return)

    def build_batches(self):
        batch_start_indices = np.arange(
            0, self.size() - self.size() % self.batch_size, self.batch_size
        )
        transition_indices = np.arange(
            0, self.size() - self.size() % self.batch_size, dtype=np.int32
        )
        np.random.shuffle(transition_indices)
        batches_indices = [
            transition_indices[i : i + self.batch_size] for i in batch_start_indices
        ]
        batches = []
        for batch_indices in batches_indices:
            batches.append(
                [
                    self.concatenated_transitions[batch_index]
                    for batch_index in batch_indices
                ]
            )
        return batches

    @property
    def concatenated_transitions(self):
        return [
            self.transitions[agent_id][t]
            for t in range(len(self.transitions[0]))
            for agent_id in range(self.n_agents)
        ]
