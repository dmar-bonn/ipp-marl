import logging
import os
import time
from typing import Dict

import numpy as np
import optuna
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import marl_framework.constants as constants
from marl_framework.missions.episode_generator import EpisodeGenerator
from marl_framework.missions.missions import Mission

from batch_memory import BatchMemory
from coma_wrapper import COMAWrapper
from mapping.grid_maps import GridMap
from sensors import Sensor
from sensors.models import SensorModel
from utils.plotting import plot_trajectories

logger = logging.getLogger(__name__)


class COMAMission(Mission):
    def __init__(
            self, params: Dict, writer: SummaryWriter, max_mean_episode_return: float
    ):
        super().__init__(params, writer, max_mean_episode_return)

        self.grid_map = GridMap(self.params)
        self.sensor = Sensor(SensorModel(), self.grid_map)
        self.num_episodes = self.params["experiment"]["missions"]["n_episodes"]
        self.batch_size = self.params["networks"]["batch_size"]
        self.batch_number = self.params["networks"]["batch_number"]
        self.n_agents = self.params["experiment"]["missions"]["n_agents"]
        self.n_actions = self.params["experiment"]["constraints"]["num_actions"]
        self.budget = params["experiment"]["constraints"]["budget"]
        self.data_passes = self.params["networks"]["data_passes"]
        self.coma_wrapper = COMAWrapper(self.params, self.writer)
        self.training_step_idx = 0
        self.environment_step_idx = 0
        self.episode_returns = []
        self.collision_returns = []
        self.utility_returns = []
        self.mode = "train"

    def execute(self):

        batch_memory = BatchMemory(self.params, self.coma_wrapper)
        episode_returns = []
        episode_reward_list = []
        absolute_returns = []
        chosen_actions = []
        chosen_altitudes = []

        for episode_idx in range(1, int(self.num_episodes * (self.batch_size * self.batch_number) / ((self.budget + 1) * self.n_agents)) + 1):

            episode = EpisodeGenerator(
                self.params, self.writer, self.grid_map, self.sensor
            )
            (
                episode_return,
                episode_rewards,
                absolute_return,
                simulated_map,
                batch_memory,
                _,
                _,
                eps,
                agent_actions,
                agent_altitudes,
            ) = episode.execute(episode_idx, batch_memory, self.coma_wrapper, self.mode)

            episode_returns.append(episode_return)
            episode_reward_list.append(episode_rewards)
            absolute_returns.append(absolute_return)
            chosen_actions.append(agent_actions)
            chosen_altitudes.append(agent_altitudes)

            if batch_memory.size() >= self.batch_size * self.batch_number:
                batch_memory.build_td_targets(self.coma_wrapper.target_critic_network)
                for data_pass in range(self.data_passes):
                    batches = batch_memory.build_batches()
                    q_values, q_metrics = self.coma_wrapper.q_learner.learn(
                        self.training_step_idx, batches, data_pass
                    )
                    v_values, v_metrics = self.coma_wrapper.v_learner.learn(
                        self.training_step_idx, batches, data_pass
                    )
                    actor_network, actor_metrics = self.coma_wrapper.actor_learner.learn(
                        batches, q_values, eps
                    )
                    if data_pass == 0:
                        self.training_step_idx += 1
                        self.environment_step_idx += batch_memory.size()
                        logger.info(f"Training step: {self.training_step_idx}")
                        logger.info(f"Environment step: {self.environment_step_idx}")
                        self.add_to_tensorboard(
                            chosen_actions, chosen_altitudes, episode_returns, absolute_returns, episode_reward_list,
                            q_metrics, actor_metrics
                        )

                batch_memory.clear()
                self.episode_returns.append(episode_return)
                self.save_best_model(actor_network)
                episode_returns = []
                episode_reward_list = []
                absolute_returns = []
                chosen_actions = []
                chosen_altitudes = []

                if self.training_step_idx % 50 == 0:
                    self.mode = "eval"
                    for i in range(50):
                        (
                            episode_return,
                            episode_rewards,
                            absolute_return,
                            simulated_map,
                            batch_memory,
                            agent_positions,
                            t_collision,
                            _,
                            agent_actions,
                            agent_altitudes,
                        ) = episode.execute(
                            episode_idx + i, batch_memory, self.coma_wrapper, self.mode
                        )
                        if i == 0:
                            plot_trajectories(
                                agent_positions,
                                self.n_agents,
                                self.writer,
                                self.training_step_idx,
                                t_collision,
                                self.budget,
                                simulated_map,
                            )

                        episode_returns.append(episode_return)
                        episode_reward_list.append(episode_rewards)
                        absolute_returns.append(absolute_return)
                        chosen_actions.append(agent_actions)
                        chosen_altitudes.append(agent_altitudes)

                    batch_memory.clear()
                    self.add_to_tensorboard(chosen_actions, chosen_altitudes, episode_returns, absolute_returns,
                                            episode_reward_list)
                    self.mode = "train"
                    episode_returns = []
                    episode_reward_list = []
                    collision_returns = []
                    chosen_actions = []
                    chosen_altitudes = []

        return self.max_mean_episode_return

    def add_to_tensorboard(
            self, chosen_actions, chosen_altitudes, episode_returns, absolute_returns, episode_rewards,
            critic_metrics=None,
            actor_metrics=None
    ):

        episode_rewards = [item for sublist in episode_rewards for item in sublist]
        chosen_actions = [item for sublist in chosen_actions for item in sublist]
        chosen_actions = [item for sublist in chosen_actions for item in sublist]
        chosen_altitudes = [item for sublist in chosen_altitudes for item in sublist]
        chosen_altitudes = [item for sublist in chosen_altitudes for item in sublist]

        action_counts = [chosen_actions.count(i) for i in range(self.n_actions)]
        altitude_counts = [chosen_altitudes.count(i) for i in [5, 10, 15]]

        plt.figure()
        fig_ = sns.barplot(
            x=list(range(self.n_actions)),
            y=action_counts,
            color="blue",
        ).get_figure()
        self.writer.add_figure(
            f"Sampled_actions_{self.mode}", fig_, self.training_step_idx, close=True
        )

        plt.figure()
        fig_ = sns.barplot(
            x=[5, 10, 15],
            y=altitude_counts,
            color="blue",
        ).get_figure()
        self.writer.add_figure(
            f"Altitudes_{self.mode}", fig_, self.training_step_idx, close=True
        )

        self.writer.add_scalar(
            f"{self.mode}Return/Episode/mean",
            np.mean(absolute_returns),
            self.training_step_idx,
        )
        self.writer.add_scalar(
            f"{self.mode}Return/Episode/std",
            np.std(absolute_returns),
            self.training_step_idx,
        )
        self.writer.add_scalar(
            f"{self.mode}Return/Episode/max",
            np.max(absolute_returns),
            self.training_step_idx,
        )
        self.writer.add_scalar(
            f"{self.mode}Return/Episode/min",
            np.min(absolute_returns),
            self.training_step_idx,
        )

        self.writer.add_scalar(
            f"{self.mode}Rewards/Episode/mean",
            np.mean(episode_rewards),
            self.training_step_idx,
        )
        self.writer.add_scalar(
            f"{self.mode}Rewards/Episode/std",
            np.std(episode_rewards),
            self.training_step_idx,
        )
        self.writer.add_scalar(
            f"{self.mode}Rewards/Episode/max",
            np.max(episode_rewards),
            self.training_step_idx,
        )
        self.writer.add_scalar(
            f"{self.mode}Rewards/Episode/min",
            np.min(episode_rewards),
            self.training_step_idx,
        )
        self.writer.add_scalar(
            f"{self.mode}Return/Relative(used)/Episode/mean",
            np.mean(episode_returns),
            self.training_step_idx,
        )
        self.writer.add_scalar(
            f"{self.mode}Return/Relative(used)/Episode/std",
            np.std(episode_returns),
            self.training_step_idx,
        )
        self.writer.add_scalar(
            f"{self.mode}Return/Relative(used)/Episode/max",
            np.max(episode_returns),
            self.training_step_idx,
        )
        self.writer.add_scalar(
            f"{self.mode}Return/Relative(used)/Episode/min",
            np.min(episode_returns),
            self.training_step_idx,
        )

        if self.mode == "train":
            self.writer.add_scalar(
                "Critic/Loss", critic_metrics[0], self.training_step_idx
            )
            self.writer.add_scalar(
                "Critic/TD-Targets mean", critic_metrics[1], self.training_step_idx
            )
            self.writer.add_scalar(
                "Critic/TD-Targets std", critic_metrics[2], self.training_step_idx
            )
            self.writer.add_scalar(
                "Critic/Q chosen mean", critic_metrics[3], self.training_step_idx
            )
            self.writer.add_scalar(
                "Critic/Q values mean", critic_metrics[4], self.training_step_idx
            )
            self.writer.add_scalar(
                "Critic/Q values min", critic_metrics[5], self.training_step_idx
            )
            self.writer.add_scalar(
                "Critic/Q values std", critic_metrics[6], self.training_step_idx
            )
            self.writer.add_scalar(
                "Critic/Explained variance", critic_metrics[7], self.training_step_idx
            )
            self.writer.add_scalar(
                "Critic/Discounted returns mean",
                np.array(critic_metrics[8]),
                self.training_step_idx,
            )
            self.writer.add_scalar(
                "Critic/Discounted_returns std",
                np.array(critic_metrics[9]),
                self.training_step_idx,
            )
            self.writer.add_scalar(
                "Critic/Abs deviation Q-value <-> Return mean",
                np.array(critic_metrics[10]),
                self.training_step_idx,
            )
            self.writer.add_scalar(
                "Critic/Abs deviation Q-value <-> Return std",
                np.array(critic_metrics[11]),
                self.training_step_idx,
            )
            self.writer.add_scalar(
                "Critic/Log probs according to critic",
                np.array(critic_metrics[12]),
                self.training_step_idx,
            )
            self.writer.add_scalar(
                "Actor/Loss", actor_metrics[0], self.training_step_idx
            )
            self.writer.add_scalar(
                "Actor/Advantages mean", actor_metrics[1], self.training_step_idx
            )
            self.writer.add_scalar(
                "Actor/Advantages std", actor_metrics[2], self.training_step_idx
            )
            self.writer.add_scalar(
                "Actor/Log probs chosen mean", actor_metrics[3], self.training_step_idx
            )
            self.writer.add_scalar(
                "Actor/Policy entropy",
                np.array(actor_metrics[4]),
                self.training_step_idx,
            )
            self.writer.add_scalar(
                "Actor/KL divergence policy",
                np.array(actor_metrics[5]),
                self.training_step_idx,
            )
            self.writer.add_scalar(
                "Actor/Hidden state entropy",
                np.array(actor_metrics[6]),
                self.training_step_idx,
            )

            if self.training_step_idx % 100 == 0:
                for tag, params in self.coma_wrapper.critic_network.named_parameters():
                    if params.grad is not None:
                        self.writer.add_histogram(
                            f"Critic/Parameters/{tag}",
                            params.data.cpu().numpy(),
                            self.training_step_idx,
                        )
                for tag, params in self.coma_wrapper.actor_network.named_parameters():
                    if params.grad is not None:
                        self.writer.add_histogram(
                            f"Actor/Parameters/{tag}",
                            params.data.cpu().numpy(),
                            self.training_step_idx,
                        )

            self.writer.add_scalar(
                "Parameters/Actor/Conv1 gradients",
                np.array(actor_metrics[-1][0]),
                self.training_step_idx,
            )
            self.writer.add_scalar(
                "Parameters/Actor/Conv2 gradients",
                np.array(actor_metrics[-1][1]),
                self.training_step_idx,
            )
            self.writer.add_scalar(
                "Parameters/Actor/Conv3 gradients",
                np.array(actor_metrics[-1][2]),
                self.training_step_idx,
            )
            self.writer.add_scalar(
                "Parameters/Actor/FC1 gradients",
                np.array(actor_metrics[-1][3]),
                self.training_step_idx,
            )
            self.writer.add_scalar(
                "Parameters/Actor/FC2 gradients",
                np.array(actor_metrics[-1][4]),
                self.training_step_idx,
            )
            self.writer.add_scalar(
                "Parameters/Actor/FC3 gradients",
                np.array(actor_metrics[-1][5]),
                self.training_step_idx,
            )
            self.writer.add_scalar(
                "Parameters/Critic/Conv1 gradients",
                np.array(critic_metrics[-1][0]),
                self.training_step_idx,
            )
            self.writer.add_scalar(
                "Parameters/Critic/Conv2 gradients",
                np.array(critic_metrics[-1][1]),
                self.training_step_idx,
            )
            self.writer.add_scalar(
                "Parameters/Critic/Conv3 gradients",
                np.array(critic_metrics[-1][2]),
                self.training_step_idx,
            )
            self.writer.add_scalar(
                "Parameters/Critic/FC1 gradients",
                np.array(critic_metrics[-1][3]),
                self.training_step_idx,
            )
            self.writer.add_scalar(
                "Parameters/Critic/FC2 gradients",
                np.array(critic_metrics[-1][4]),
                self.training_step_idx,
            )
            self.writer.add_scalar(
                "Parameters/Critic/FC3 gradients",
                np.array(critic_metrics[-1][5]),
                self.training_step_idx,
            )

    def save_best_model(self, actor_network):
        running_mean_return = sum(self.episode_returns) / len(self.episode_returns)
        patience = self.params["experiment"]["missions"]["patience"]
        best_model_file_path = os.path.join(constants.LOG_DIR, "best_model.pth")

        if (
                len(self.episode_returns) >= patience
                and running_mean_return > self.max_mean_episode_return
        ):
            self.max_mean_episode_return = running_mean_return
            torch.save(actor_network, best_model_file_path)
        if self.training_step_idx == 300:
            torch.save(actor_network, os.path.join(constants.LOG_DIR, "best_model_300.pth"))
        if self.training_step_idx == 400:
            torch.save(actor_network, os.path.join(constants.LOG_DIR, "best_model_400.pth"))
        if self.training_step_idx == 500:
            torch.save(actor_network, os.path.join(constants.LOG_DIR, "best_model_500.pth"))
        if self.training_step_idx == 600:
            torch.save(actor_network, os.path.join(constants.LOG_DIR, "best_model_600.pth"))
