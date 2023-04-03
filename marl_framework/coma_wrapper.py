import copy
import logging
from typing import Dict, List
import numpy as np
import torch
from matplotlib import pyplot as plt

from torch.utils.tensorboard import SummaryWriter

from marl_framework.actor.learner import ActorLearner
from marl_framework.actor.network import ActorNetwork
from marl_framework.agent.agent import Agent
from marl_framework.agent.communication_log import CommunicationLog
from marl_framework.agent.state_space import AgentStateSpace
from marl_framework.critic.learner import CriticLearner
from marl_framework.critic.network import CriticNetwork

from actor.transformations import get_network_input as get_actor_input
from critic.transformations import get_network_input as get_critic_input
from utils.reward import get_global_reward
from utils.state import get_w_entropy_map

logger = logging.getLogger(__name__)


class COMAWrapper:
    def __init__(self, params: Dict, writer: SummaryWriter):
        self.params = params
        self.mission_type = self.params["experiment"]["missions"]["type"]
        self.n_agents = self.params["experiment"]["missions"]["n_agents"]
        self.budget = params["experiment"]["constraints"]["budget"]
        self.agent_state_space = AgentStateSpace(self.params)
        self.actor_network = ActorNetwork(self.params)
        self.actor_learner = ActorLearner(
            self.params, writer, self.actor_network, self.agent_state_space
        )
        self.q_network = QNetwork(self.params)
        self.target_q_network = copy.deepcopy(self.q_network)
        self.q_learner = QLearner(self.params, writer, self.q_network)

    def build_observations(self, mapping, agents, num_episode, t, params, batch_memory, mode):
        communication_log = CommunicationLog(self.params, num_episode)
        local_maps = []
        positions = []
        global_information = {}
        for agent_id in range(self.n_agents):
            global_information, local_map, position = agents[agent_id].communicate(
                t, num_episode, communication_log, mode
            )
            local_maps.append(local_map)
            positions.append(position)

        observations = []
        for agent_id in range(self.n_agents):
            local_information, fused_local_map = agents[agent_id].receive_messages(communication_log, agent_id, t)

            observation = get_actor_input(
                local_information,
                fused_local_map,
                mapping.simulated_map,
                agent_id,
                t,
                params,
                batch_memory,
                self.agent_state_space,
            )

            batch_memory.add(agent_id, observation=observation)
            observations.append(observation)

        return global_information, positions, observations

    def steps(
            self,
            mapping,
            t: int,
            agents: List[Agent],
            accumulated_map_knowledge,
            num_episode,
            batch_memory,
            global_information,
            simulated_map,
            params,
            mode,
    ):
        next_maps = []
        next_positions = []
        actions = []
        footprints = []
        altitudes = []
        maps2communicate = []

        # if t == 0:
        critic_map_knowledge = mapping.fuse_map(accumulated_map_knowledge, global_information, mode, "global")

        for agent_id in range(self.n_agents):
            next_map, next_position, eps, action, footprint_idx, map2communicate = agents[agent_id].step(
                agent_id, t, num_episode, batch_memory, mode, next_positions
            )
            next_maps.append(next_map)
            next_positions.append(next_position)
            try:
                actions.append(action.tolist()[0])
            except:
                actions.append(action)
            footprints.append(footprint_idx)
            altitudes.append(next_position[2])
            maps2communicate.append(map2communicate)

        if self.mission_type == "DeepQ":
            for agent_id in range(self.n_agents):
                update_simulation = mapping.fuse_map(critic_map_knowledge.copy(), [maps2communicate[agent_id]], agent_id, "global")
                done, relative_reward, absolute_reward = get_global_reward(critic_map_knowledge, update_simulation, self.mission_type, footprints, simulated_map, self.agent_state_space, actions[agent_id], agent_id, t, self.budget)
                batch_memory.insert(-1, agent_id, reward=relative_reward)

        for agent_id in range(self.n_agents):
            states = get_critic_input(
                t,
                global_information,
                critic_map_knowledge,
                batch_memory,
                agent_id,
                simulated_map,
                params,
            )
        next_global_map = mapping.fuse_map(accumulated_map_knowledge, global_information, mode, "global")

        if self.mission_type == "COMA":
            done, relative_reward, absolute_reward = get_global_reward(
                accumulated_map_knowledge,  #
                next_global_map,
                self.mission_type,
                None,
                simulated_map,
                self.agent_state_space,
                actions,
                None,
                t,
                self.budget
            )

        if t == self.budget:
            done = True
            # reward += 0.5

        if self.mission_type == "COMA":
            for agent_id in range(self.n_agents):
                batch_memory.insert(-1, agent_id, reward=relative_reward, done=done)
        if self.mission_type == "DeepQ":
            for agent_id in range(self.n_agents):
                batch_memory.insert(-1, agent_id, done=done)

        return (
            batch_memory,
            relative_reward,
            absolute_reward,
            done,
            next_positions,
            eps,
            actions,
            altitudes,
            next_global_map,
        )



