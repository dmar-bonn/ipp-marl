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
        self.n_agents = self.params["mission"]["n_agents"]
        self.budget = params["MARL_cast"]["state_space"]["budget"]
        self.batch_size = self.params["networks"]["updates"]["batch_size"]
        self.shuffle_batches = params["networks"]["updates"]["shuffle_batches"]
        self.gamma = params["rl_hyperparams"]["gamma"]
        self.lam = params["rl_hyperparams"]["lambda"]
        self.transitions = {agent_id: [] for agent_id in range(self.n_agents)}

    def clear(self):
        self.transitions = {agent_id: [] for agent_id in range(self.n_agents)}

    def add(
        self,
        agent_id: int,
        q_state=None,
        v_state=None,
        observation=None,
        action=None,
        mask=None,
        reward=None,
        done=None,
        q_target=None,
        v_target=None,
        discounted_return=None,
    ):
        self.transitions[agent_id].append(
            TransitionCOMA(
                q_state,
                v_state,
                observation,
                action,
                mask,
                reward,
                done,
                q_target,
                v_target,
                discounted_return,
            )
        )

    def insert(
        self,
        t: int,
        agent_id: int,
        q_state=None,
        v_state=None,
        observation=None,
        action=None,
        mask=None,
        reward=None,
        done=None,
        q_target=None,
        v_target=None,
        discounted_return=None,
    ):
        if q_state is not None:
            self.transitions[agent_id][t] = self.transitions[agent_id][t]._replace(
                q_state=q_state
            )
        if v_state is not None:
            self.transitions[agent_id][t] = self.transitions[agent_id][t]._replace(
                v_state=v_state
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
        if q_target is not None:
            self.transitions[agent_id][t] = self.transitions[agent_id][t]._replace(
                q_target=q_target
            )
        if v_target is not None:
            self.transitions[agent_id][t] = self.transitions[agent_id][t]._replace(
                v_target=v_target
            )
        if discounted_return is not None:
            self.transitions[agent_id][t] = self.transitions[agent_id][t]._replace(
                discounted_return=discounted_return
            )

    def get(self, t: int, agent_id: int, argument: str):
        if argument == "q_state":
            return self.transitions[agent_id][t].q_state
        if argument == "v_state":
            return self.transitions[agent_id][t].v_state
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
        if argument == "q_target":
            return self.transitions[agent_id][t].q_target
        if argument == "v_target":
            return self.transitions[agent_id][t].v_target
        if argument == "discounted_return":
            return self.transitions[agent_id][t].discounted_return

    def size(self):
        return len(self.transitions[0]) * self.n_agents

    def build_q_targets(self, target_q_network):
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
                                target_q_values, _ = target_q_network.forward(
                                    self.get(t + n, agent_id, "q_state")
                                )
                                target_q_value = target_q_values[
                                    self.get(t + n, agent_id, "action").to("cpu")
                                ]

                            n_step_return += np.power(self.gamma, n) * target_q_value

                    sum_n_step_returns += np.power(self.lam, n - 1) * n_step_return

                self.insert(t, agent_id, q_target=(1 - self.lam) * sum_n_step_returns)
                self.insert(t, agent_id, discounted_return=discounted_return)

    def build_v_targets(self, target_v_network):
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
                                target_v_value, _ = target_v_network.forward(
                                    self.get(t + n, agent_id, "v_state")
                                )

                            n_step_return += np.power(self.gamma, n) * target_v_value

                    sum_n_step_returns += np.power(self.lam, n - 1) * n_step_return

                self.insert(t, agent_id, v_target=(1 - self.lam) * sum_n_step_returns)

    def build_batches(self):
        batch_start_indices = np.arange(
            0, self.size() - self.size() % self.batch_size, self.batch_size
        )
        transition_indices = np.arange(
            0, self.size() - self.size() % self.batch_size, dtype=np.int32
        )
        if self.shuffle_batches:
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
