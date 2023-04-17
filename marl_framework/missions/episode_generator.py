import logging
from typing import Dict, List

from torch.utils.tensorboard import SummaryWriter

from marl_framework.agent.agent import Agent
from marl_framework.coma_wrapper import COMAWrapper
from marl_framework.mapping.grid_maps import GridMap
from marl_framework.mapping.mappings import Mapping
from marl_framework.sensors import Sensor
from marl_framework.sensors.models import SensorModel
from marl_framework.utils.state import get_shannon_entropy

logger = logging.getLogger(__name__)


class EpisodeGenerator:
    def __init__(
        self, params: Dict, writer: SummaryWriter, grid_map: GridMap, sensor: Sensor
    ):
        self.params = params
        self.writer = writer
        self.grid_map = grid_map
        self.sensor = sensor
        self.mission_mode = params["experiment"]["missions"]["mission_mode"]
        self.budget = params["experiment"]["constraints"]["budget"]
        self.n_agents = self.params["experiment"]["missions"]["n_agents"]
        self.batch_size = self.params["networks"]["batch_size"]
        self.prior = self.params["mapping"]["prior"]
        self.class_weighting = params["experiment"]["missions"]["class_weighting"]
        self.mission_time: int = 0
        self.state = None
        self.data_episodes = []
        self.episode_returns = []
        self.collision_returns = []
        self.utility_returns = []

    def execute(self, num_episode: int, batch_memory, coma_wrapper, mode):
        mapping = Mapping(self.grid_map, self.sensor, self.params, num_episode)
        agents = self.init_agents(mapping, coma_wrapper)
        episode_return = 0
        absolute_return = 0
        episode_rewards = []
        agent_positions = []
        agent_actions = []
        agent_altitudes = []
        current_global_map = agents[0].local_map.copy()

        for t in range(self.budget + 1):
            global_information, positions, observations = coma_wrapper.build_observations(
                mapping, agents, num_episode, t, self.params, batch_memory, mode
            )
            batch_memory, relative_reward, absolute_reward, done, new_positions, eps, actions, altitudes, current_global_map = coma_wrapper.steps(
                mapping,
                t,
                agents,
                current_global_map,
                num_episode,
                batch_memory,
                global_information,
                mapping.simulated_map,
                self.params,
                mode,
            )

            agent_actions.append(actions)

            episode_return += relative_reward
            episode_rewards.append(relative_reward)
            absolute_return += absolute_reward

            if t == 0:
                agent_positions.append(positions)
            agent_positions.append(new_positions)
            agent_altitudes.append(altitudes)

        return (
            episode_return,
            episode_rewards,
            absolute_return,
            mapping.simulated_map,
            batch_memory,
            agent_positions,
            t,
            eps,
            agent_actions,
            agent_altitudes,
        )

    def init_agents(self, mapping: Mapping, coma_wrapper: COMAWrapper) -> List[Agent]:
        agents = []
        for agent_id in range(self.n_agents):
            agents.append(
                Agent(
                    coma_wrapper.actor_network,
                    self.params,
                    mapping,
                    agent_id,
                    coma_wrapper.agent_state_space,
                )
            )
        return agents
