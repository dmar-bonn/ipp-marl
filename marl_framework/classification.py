import copy
import logging
import os
from typing import Dict
import constants
import time

import json

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from torch import nn
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TKAgg')

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

from marl_framework.missions.episode_generator import EpisodeGenerator
from marl_framework.utils.utils import clip_gradients
from marl_framework.agent.state_space import AgentStateSpace
from marl_framework.mapping.mappings import Mapping

from agent.action_space import AgentActionSpace
from batch_memory import BatchMemory
from coma_wrapper import COMAWrapper
from logger import setup_logger
from mapping.grid_maps import GridMap
from params import load_params
from sensors import Sensor
from sensors.models import SensorModel
from utils.reward import get_utility_reward

logger = logging.getLogger(__name__)


class Classification:
    def __init__(self, params: Dict, writer: SummaryWriter):
        self.params = params
        self.writer = writer
        self.coma_wrapper = COMAWrapper(params, writer)
        self.grid_map = GridMap(self.params)
        self.sensor = Sensor(SensorModel(), self.grid_map)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_episodes = self.params["classification"]["n_episodes"]
        self.batch_size = self.params["classification"]["batch_size"]
        self.number_epochs = self.params["classification"]["number_epochs"]
        self.data_split = self.params["classification"]["data_split"]
        self.lr = params["networks"]["critic"]["learning_rate"]
        self.momentum = params["networks"]["critic"]["momentum"]
        self.gradient_norm = params["networks"]["actor"]["gradient_norm"]
        self.mapping = Mapping(self.grid_map, self.sensor, self.params, None)
        self.agent_state_space = AgentStateSpace(params)
        self.n_actions = params["experiment"]["constraints"]["num_actions"]

        ####### Set experiment run name phase ("collect", "train", or "evaluate"), and path #######

        self.path = " "  # "/data_time/ipp_regression/cnn4"
        self.phase = "collect"
        self.experiment = "Experiment1"
        self.input_type = self.path.split("/")[-1]

        ###########################################################################################

        if self.phase == "train" or self.phase == "evaluate":
            if self.input_type == "mlp1":
                self.model = ModelMLP1(params)
            elif self.input_type == "mlp2":
                self.model = ModelMLP2(params)
            elif self.input_type == "cnn1":
                self.model = ModelCNN1(params)
            elif self.input_type == "cnn2" or self.input_type == "advantage1":
                self.model = ModelCNN2(params)
            elif self.input_type == "cnn3":
                self.model = ModelCNN3(params)
            elif self.input_type == "cnn4":
                self.model = ModelCNN4(params)
            elif self.input_type == "mixed":
                self.model = ModelMixed(params)

            if self.input_type == "ipp_classification":
                self.criterion = torch.nn.CrossEntropyLoss()  # torch.nn.MSELoss()
            elif self.input_type == "collision_classification":
                self.criterion = torch.nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor([7]))  # SET WEIGHT AND ADAPT SIGMOID!!!
            else:
                self.criterion = torch.nn.MSELoss()

                # self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr)   # , momentum=0.2)
                self.optimizer = torch.optim.RMSprop(
                    self.model.parameters(), self.lr)  # , weight_decay=0.0001
                # )  # , self.momentum)
                # self.optimizer = optim.Adam(self.model.parameters(), self.lr, weight_decay=0.0001)

    def execute(self):
        if self.phase == "collect":
            batch_memory = BatchMemory(self.params, self.coma_wrapper)
            for episode_idx in range(1, self.num_episodes + 1):
                print(f"episode {episode_idx}")
                episode = EpisodeGenerator(
                    self.params, self.writer, self.grid_map, self.sensor
                )
                (
                    episode_return,
                    collision_return,
                    utility_return,
                    simulated_map,
                    batch_memory,
                    _,
                    _,
                ) = episode.execute(
                    episode_idx, batch_memory, self.coma_wrapper, "train"
                )
            self.split_dataset(batch_memory)
        elif self.phase == "regression":
            self.regression()
        elif self.phase == "train":
            self.train()
        elif self.phase == "evaluate":
            self.evaluate()

    def split_dataset(self, batch_memory):
        sample_indices = np.arange(0, batch_memory.size(), dtype=np.int32)
        np.random.shuffle(sample_indices)

        training_set_indices = sample_indices[
                               : int(batch_memory.size() * self.data_split[0])
                               ]
        training_input = []
        training_labels = []

        for c_i, i in enumerate(training_set_indices):
            print(f"c_i: {c_i}")
            training_input, training_labels = add_data(self.params, training_input, training_labels, batch_memory, i,
                                                       self.mapping)
        validation_set_indices = sample_indices[
                                 int(batch_memory.size() * self.data_split[0]): int(
                                     batch_memory.size() * self.data_split[0]
                                 )
                                                                                + int(
                                     batch_memory.size() * self.data_split[1])
                                 ]

        validation_input = []
        validation_labels = []
        for c_i, i in enumerate(validation_set_indices):
            print(f"j: {c_i}")
            validation_input, validation_labels = add_data(self.params, validation_input, validation_labels,
                                                           batch_memory, i, self.mapping)

        testing_set_indices = sample_indices[
                              int(batch_memory.size() * self.data_split[1]): int(
                                  batch_memory.size() * self.data_split[1]
                              )
                                                                             + int(
                                  batch_memory.size() * self.data_split[2])
                              ]
        testing_input = []
        testing_labels = []

        for c_i, i in enumerate(testing_set_indices):
            print(f"k: {c_i}")
            testing_input, testing_labels = add_data(self.params, testing_input, testing_labels, batch_memory, i,
                                                     self.mapping)

        store_data(
            self.experiment,
            training_input,
            training_labels,
            validation_input,
            validation_labels,
            testing_input,
            testing_labels,
            self.batch_size,
        )

    def regression(self):
        training_input, training_labels, validation_input, validation_labels = read_training_data(
            self.experiment, self.path
        )
        clf = LogisticRegression().fit(training_input, training_labels)
        training_accuracy = clf.score(training_input, training_labels)
        validation_accuracy = clf.score(validation_input, validation_labels)
        parameters = LogisticRegression().get_params()
        print(f"training_accuracy: {training_accuracy}")
        print(f"validation_accuracy: {validation_accuracy}")
        print(f"parameters: {parameters}")

    def train(self):
        training_input, training_labels, validation_input, validation_labels = read_training_data(
            self.experiment, self.path
        )

        print(f"len training_input: {len(training_input)}")
        print(f"len training_labels: {len(training_labels)}")
        print(f"len validation_input: {len(validation_input)}")
        print(f"len validation_labels: {len(validation_labels)}")

        best_epoch = 0
        best_performance = 100        # 0
        best_model_parameters = copy.deepcopy(self.model.state_dict())

        training_loss_series = []
        training_gradient_series = []
        training_precision_series = []
        training_recall_series = []
        training_f1_score_series = []
        training_accuracy_series = []
        training_classified_series = []
        training_mse_series = []
        validation_loss_series = []
        validation_gradient_series = []
        validation_precision_series = []
        validation_recall_series = []
        validation_f1_score_series = []
        validation_accuracy_series = []
        validation_classified_series = []
        validation_mse_series = []

        self.model = self.model.to(self.device)

        for epoch in range(self.number_epochs):
            print(f"Epoch {epoch + 1}")
            training_phases = ["Training", "Validation"]

            for tp in training_phases:
                if tp == "Training":
                    self.model.train()
                    batches_inputs, batches_labels = self.extract_batches(
                        training_input, training_labels
                    )
                else:
                    self.model.eval()
                    batches_inputs, batches_labels = self.extract_batches(
                        validation_input, validation_labels
                    )
                labels = []
                classifications = []
                losses = []

                for batch_input, batch_label in zip(batches_inputs, batches_labels):
                    batch_input = torch.tensor(batch_input).to(self.device)
                    batch_label = torch.tensor(batch_label).to(self.device)

                    # if self.input_type[:-1] == "cnn" or self.input_type == "mixed":
                    #     batch_input = torch.unsqueeze(batch_input, 1)
                    if self.input_type == "mixed":
                        batch_input1 = batch_input[:, :, :self.agent_state_space.space_x_dim, :]
                        batch_input2 = batch_input[:, :, self.agent_state_space.space_x_dim:, :]
                        classification = self.model.forward(batch_input1.float(), batch_input2[:, 0, :, 0].float(),
                                                            self.path.split("/")[-2])
                    else:
                        classification = self.model.forward(batch_input.float(),
                                                            self.path.split("/")[-2])

                    classification = torch.squeeze(classification)

                    loss = self.criterion(classification, batch_label)

                    if tp == "Training":
                        self.optimizer.zero_grad()
                        loss.backward()
                        clip_gradients(self.model, self.gradient_norm)
                        self.optimizer.step()

                    labels.append(
                        torch.clone(batch_label).cpu().detach().numpy().tolist()
                    )
                    classifications.append(
                        torch.clone(classification)  # (torch.sigmoid(classification))
                        .cpu()
                        .detach()
                        .numpy()
                        .tolist()
                    )
                    losses.append(torch.clone(loss).cpu().detach().numpy().tolist())

                precision, recall, f1_score, accuracy, distribution, classified, report, mse = get_metrics(
                    classifications, labels, self.path.split("/")[-2].split("_")[-1], self.path.split("/")[-2].split("_")[0], self.n_actions
                )

                if tp == "Validation" and sum(losses) / len(losses) < best_performance:        # f1_score >= best_performance
                    best_epoch = epoch
                    best_performance = sum(losses) / len(losses)              # f1 score
                    best_model_parameters = copy.deepcopy(self.model.state_dict())

                if epoch % 20 == 0:
                    visualize_regression_performance(classifications, labels, tp, epoch, self.params)

                avg_gradient_norm = get_avg_gradient_norm(self.model)

                if tp == "Training":
                    training_loss_series.append(np.mean(np.array(losses)))
                    training_gradient_series.append(avg_gradient_norm)
                    training_precision_series.append(precision)
                    training_recall_series.append(recall)
                    training_f1_score_series.append(f1_score)
                    training_accuracy_series.append(accuracy)
                    training_classified_series.append(classified)
                    training_mse_series.append(mse)
                elif tp == "Validation":
                    validation_loss_series.append(np.mean(np.array(losses)))
                    validation_gradient_series.append(avg_gradient_norm)
                    validation_precision_series.append(precision)
                    validation_recall_series.append(recall)
                    validation_f1_score_series.append(f1_score)
                    validation_accuracy_series.append(accuracy)
                    validation_classified_series.append(classified)
                    validation_mse_series.append(mse)

                print(f"{tp} - LOSS: {np.mean(np.array(losses))}")
                print(f"{tp} - avg gradient norm: {avg_gradient_norm}")
                print(" ")
                print(f"{tp} - Precision: {precision}")
                print(f"{tp} - Recall: {recall}")
                print(f"{tp} - F1 Score: {f1_score}")
                print(f"{tp} - Accuracy: {accuracy}")
                print(f"{tp} - Distribution: {distribution}")
                print(f"{tp} - Classified / mean absolute label: {classified}")
                print(f"{tp} - Classification report: {report}")
                print(" ")
                print(f"{tp} - Mean Squared Error: {mse}")

                if tp == "Training":
                    print("-----------")
                else:
                    print("====================")

        results_all = {}
        results_best = {}
        for training_phase in training_phases:
            if training_phase == "Training":
                results_all, results_best = self.store_results(
                    results_all,
                    results_best,
                    training_phase,
                    best_epoch,
                    training_loss_series,
                    training_gradient_series,
                    training_precision_series,
                    training_recall_series,
                    training_f1_score_series,
                    training_accuracy_series,
                    distribution,
                    training_classified_series,
                    training_mse_series,
                )
            if training_phase == "Validation":
                results_all, _ = self.store_results(
                    results_all,
                    results_best,
                    training_phase,
                    best_epoch,
                    validation_loss_series,
                    validation_gradient_series,
                    validation_precision_series,
                    validation_recall_series,
                    validation_f1_score_series,
                    validation_accuracy_series,
                    distribution,
                    validation_classified_series,
                    validation_mse_series,
                )

        plot_training_results(
            self.experiment,
            self.path,
            training_loss_series,
            training_gradient_series,
            training_precision_series,
            training_recall_series,
            training_f1_score_series,
            training_accuracy_series,
            training_classified_series,
            training_mse_series,
            validation_loss_series,
            validation_gradient_series,
            validation_precision_series,
            validation_recall_series,
            validation_f1_score_series,
            validation_accuracy_series,
            validation_classified_series,
            validation_mse_series,
        )
        print(
            f"Best model after {best_epoch + 1} epochs with an f1 score of {best_performance}"
        )
        self.model.load_state_dict(best_model_parameters)
        torch.save(
            self.model.state_dict(),
            f"{self.experiment}{self.path}/results/best_model.pth",
        )

    def evaluate(self):
        self.model.load_state_dict(torch.load(
            f"{self.experiment}{self.path}/results/best_model.pth", map_location=self.device
        ))
        self.model.eval()
        testing_input, testing_labels = read_testing_data(self.experiment, self.path)
        batches_inputs, batches_labels = self.extract_batches(
            testing_input, testing_labels
        )
        labels = []
        classifications = []
        stamps = []

        for batch_input, batch_label in zip(batches_inputs, batches_labels):
            batch_input = torch.tensor(batch_input).to(self.device)
            batch_label = torch.tensor(batch_label).to(self.device)
            self.model.to(self.device)

            # if self.input_type[:-1] == "cnn" or self.input_type == "mixed":
            #     batch_input = torch.unsqueeze(batch_input, 1)
            if self.input_type == "mixed":
                batch_input1 = batch_input[:, :, :self.agent_state_space.space_x_dim, :]
                batch_input2 = batch_input[:, :, self.agent_state_space.space_x_dim:, :]
                classification = self.model.forward(batch_input1.float(), batch_input2[:, 0, :, 0].float(),
                                                    self.path.split("/")[-2])
            else:
                classification = self.model.forward(batch_input.float(),
                                                    self.path.split("/")[-2])

            # classification = self.model.forward(batch_input.float())
            classification = torch.squeeze(classification)
            labels.append(torch.clone(batch_label).cpu().detach().numpy().tolist())
            classifications.append(
                torch.clone(classification).cpu().detach().numpy().tolist()
            )
            # stamps.append(batch_stamp)

        print(f"classification: {classifications}")
        print(f"labels: {labels}")

        precision, recall, f1_score, accuracy, distribution, classified, report, mse = get_metrics(
            classifications, labels, self.path.split("/")[-2].split("_")[-1], self.path.split("/")[-2].split("_")[0], self.n_actions
        )

        visualize_regression_performance(classifications, labels, "Testing", None, self.params)

        print(f"Testing - Precision: {precision}")
        print(f"Testing - Recall: {recall}")
        print(f"Testing - F1 Score: {f1_score}")
        print(f"Testing - Accuracy: {accuracy}")
        print(f"Testing - Distribution: {distribution}")
        print(f"Testing - Classified / mean_absolute_label: {classified}")
        print(f"Testing - Classification report: {report}")
        print(f"Testing - Mean squared error: {mse}")
        self.store_results(
            None,
            None,
            "Testing",
            None,
            None,
            None,
            precision,
            recall,
            f1_score,
            accuracy,
            distribution,
            classified,
            mse,
        )
        print("====================")

    def extract_batches(self, inputs, labels):
        batch_start_indices = np.arange(0, len(inputs), self.batch_size)
        sample_indices = np.arange(0, len(inputs))
        np.random.shuffle(sample_indices)
        batches_indices = [
            sample_indices[i: i + self.batch_size] for i in batch_start_indices
        ]
        batches_input = []
        batches_labels = []
        batches_stamps = []
        for batch_indices in batches_indices:
            batches_input.append([inputs[batch_index] for batch_index in batch_indices])
        for batch_indices in batches_indices:
            batches_labels.append(
                [labels[batch_index][0] for batch_index in batch_indices]
            )
        # for batch_indices in batches_indices:
        #     batches_stamps.append(
         #        [labels[batch_index][1] for batch_index in batch_indices]
          #   )

       #  print(f"batches_labels: {batches_labels}")
       #  print(f"batches_stamps: {batches_stamps}")

        return batches_input, batches_labels   # , batches_stamps

    def store_results(
            self,
            results_all,
            results_best,
            tp,
            best_epoch,
            loss,
            avg_gradient_norm,
            precision,
            recall,
            f1_score,
            accuracy,
            distribution,
            classified,
            mse,
    ):
        run_path = self.experiment + f"{self.path}/results"
        if not os.path.exists(run_path):
            os.mkdir(run_path)

        if not tp == "Testing":
            for epoch in range(self.number_epochs):
                if tp == "Training":
                    results_all[str(epoch)] = {}
                results_all[str(epoch)][tp] = {
                    "loss": loss[epoch],
                    "avg_gradient_norm": avg_gradient_norm[epoch],
                    "precision": precision[epoch],
                    "recall": recall[epoch],
                    "f1_score": f1_score[epoch],
                    "accuracy": accuracy[epoch],
                    "classified": classified[epoch],
                    "mse": mse[epoch],
                }
            json_object = json.dumps(results_all, indent=4)
            with open(f"{run_path}/training_all.json", "w") as outfile:
                outfile.write(json_object)

            if tp == "Training":
                results_best[str(best_epoch)] = {}
            results_best[str(best_epoch)][tp] = {
                "loss": loss[best_epoch],
                "avg_gradient_norm": avg_gradient_norm[best_epoch],
                "precision": precision[best_epoch],
                "recall": recall[best_epoch],
                "f1_score": f1_score[best_epoch],
                "accuracy": accuracy[best_epoch],
                "distribution": distribution,
                "classified": classified[best_epoch],
                "mse": mse[best_epoch],
            }
            json_object = json.dumps(results_best, indent=4)
            with open(f"{run_path}/training_best.json", "w") as outfile:
                outfile.write(json_object)

            return results_all, results_best

        elif tp == "Testing":
            results = {
                "loss": loss,
                "avg_gradient_norm": avg_gradient_norm,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "accuracy": accuracy,
                "distribution": distribution,
                "classified": classified,
                "mse": mse,
            }
            json_object = json.dumps(results, indent=4)
            with open(f"{run_path}/evaluation.json", "w") as outfile:
                outfile.write(json_object)

            return None, None


class ModelMLP1(nn.Module):
    def __init__(self, params: Dict):
        super(ModelMLP1, self).__init__()
        self.params = params
        self.batch_size = self.params["classification"]["batch_size"]
        self.device = torch.device("cpu")  # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = params["experiment"]["constraints"]["num_actions"]

        self.activation = torch.nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.batch_norm3 = nn.BatchNorm1d(64)
        self.dropout = torch.nn.Dropout()

        self.fc1 = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 1)
        self.fc_head = nn.Linear(64, self.n_actions)

    def forward(self, data, task_type):
        output = self.activation(self.fc1(data))
        output = self.batch_norm1(output)
        output = self.activation(self.fc2(output))
        output = self.batch_norm2(output)
        output = self.activation(self.fc3(output))
        output = self.batch_norm3(output)
        # output = self.dropout(output)
        if task_type == "ipp_classification":
            output = self.fc_head(output)
        else:
            output = self.fc4(output)
        if task_type == "ipp_classification":
            output = torch.sigmoid(output)
        return output


class ModelMLP2(nn.Module):
    def __init__(self, params: Dict):
        super(ModelMLP2, self).__init__()
        self.params = params
        self.batch_size = self.params["classification"]["batch_size"]
        self.device = torch.device("cpu")  # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = params["experiment"]["constraints"]["num_actions"]

        self.activation = torch.nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.batch_norm3 = nn.BatchNorm1d(256)
        self.dropout = torch.nn.Dropout()

        self.fc1 = nn.Linear(12, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 1)
        self.fc_head = nn.Linear(256, self.n_actions)

    def forward(self, data, task_type):
        output = self.activation(self.fc1(data))
        output = self.batch_norm1(output)
        output = self.activation(self.fc2(output))
        output = self.batch_norm2(output)
        output = self.activation(self.fc3(output))
        output = self.batch_norm3(output)
        # output = self.dropout(output)
        if task_type == "ipp_classification":
            output = self.fc_head(output)
        else:
            output = self.fc4(output)
        if task_type.split("_")[-1] == "classification":
            output = torch.sigmoid(output)
        return output


class ModelCNN1(nn.Module):
    def __init__(self, params: Dict):
        super(ModelCNN1, self).__init__()
        self.params = params
        self.batch_size = self.params["classification"]["batch_size"]
        self.device = torch.device("cpu")  # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = params["experiment"]["constraints"]["num_actions"]

        self.activation = torch.nn.ReLU()
        self.batch_norm1 = torch.nn.BatchNorm2d(256)
        self.batch_norm2 = torch.nn.BatchNorm2d(256)
        self.dropout = torch.nn.Dropout()

        self.conv1 = torch.nn.Conv2d(4, 256, 4)
        self.conv2 = torch.nn.Conv2d(256, 256, 3)
        self.conv3 = torch.nn.Conv2d(256, 256, 3)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(256, 1)
        self.linear_head = nn.Linear(256, self.n_actions)

    def forward(self, data, task_type):
        data = torch.stack([data[:, :(data.size()[1] // 4), :],
                            data[:, (data.size()[1] // 4):2 * (data.size()[1] // 4), :],
                            data[:, 2 * (data.size()[1] // 4):3 * (data.size()[1] // 4), :],
                            data[:, 3 * (data.size()[1] // 4):, :]], dim=1)

        output = self.activation(self.conv1(data))
        # output = self.dropout(output)
        output = self.batch_norm1(output)
        output = self.activation(self.conv2(output))
        # output = self.dropout(output)
        output = self.batch_norm2(output)
        output = self.activation(self.conv3(output))
        # output = self.dropout(output)
        output = self.flatten(output)
        if task_type == "ipp_classification":
            output = self.linear_head(output)
        else:
            output = self.linear(output)
        if task_type.split("_")[-1] == "classification":
            output = torch.sigmoid(output)
        return output


class ModelCNN2(nn.Module):
    def __init__(self, params: Dict):
        super(ModelCNN2, self).__init__()
        self.params = params
        self.batch_size = self.params["classification"]["batch_size"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = params["experiment"]["constraints"]["num_actions"]

        self.activation = torch.nn.ReLU()
        self.batch_norm1 = torch.nn.BatchNorm2d(512)
        self.batch_norm2 = torch.nn.BatchNorm2d(512)
        self.dropout = torch.nn.Dropout()

        self.conv1 = torch.nn.Conv2d(7, 512, 4)
        self.conv2 = torch.nn.Conv2d(512, 512, 3)
        self.conv3 = torch.nn.Conv2d(512, 512, 3)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)
        self.linear_head = nn.Linear(512, self.n_actions)

        # self.activation = torch.nn.ReLU()
        # self.batch_norm1 = torch.nn.BatchNorm2d(64)
        # self.batch_norm2 = torch.nn.BatchNorm2d(64)
        # self.batch_norm3 = torch.nn.BatchNorm2d(256)
        # self.batch_norm4 = torch.nn.BatchNorm2d(256)
        # self.dropout = torch.nn.Dropout()
        #
        # self.conv1 = torch.nn.Conv2d(3, 64, 1)
        # self.conv2 = torch.nn.Conv2d(64, 1, 1)
        # self.conv3 = torch.nn.Conv2d(3, 64, 1)
        # self.conv4 = torch.nn.Conv2d(64, 1, 1)
        # # self.conv5 = torch.nn.Conv2d(128, 1, 1)
        # self.conv6 = torch.nn.Conv2d(3, 256, 4)
        # self.conv7 = torch.nn.Conv2d(256, 256, 3)
        # self.conv8 = torch.nn.Conv2d(256, 256, 3)
        #
        # self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(256, 256)
        # self.fc2 = nn.Linear(256, 1)
        # # self.fc3 = nn.Linear(128, 1)
        # self.linear_head = nn.Linear(256, self.n_actions)

    def forward(self, data, task_type):

        option = 1

        if option == 1:
            data = torch.stack([data[:, :(data.size()[1] // 7), :],
                                data[:, (data.size()[1] // 7):2 * (data.size()[1] // 7), :],
                                data[:, 2 * (data.size()[1] // 7):3 * (data.size()[1] // 7), :],
                                data[:, 3 * (data.size()[1] // 7):4 * (data.size()[1] // 7), :],
                                data[:, 4 * (data.size()[1] // 7):5 * (data.size()[1] // 7), :],
                                data[:, 5 * (data.size()[1] // 7):6 * (data.size()[1] // 7), :],
                                data[:, 6 * (data.size()[1] // 7):, :]], dim=1)

            output = self.activation(self.conv1(data))
            # output = self.dropout(output)
            output = self.batch_norm1(output)
            output = self.activation(self.conv2(output))
            # output = self.dropout(output)
            output = self.batch_norm2(output)
            output = self.activation(self.conv3(output))
            # output = self.dropout(output)
            output = self.flatten(output)
            output = self.activation(self.fc1(output))
            if task_type == "ipp_classification":
                output = self.linear_head(output)
            else:
                # output = self.fc2(output)
                output = self.fc3(output)
            if task_type.split("_")[-1] == "classification":
                output = torch.sigmoid(output)

        else:
            position_data = torch.stack([data[:, :(data.size()[1] // 7), :],
                                data[:, (data.size()[1] // 7):2 * (data.size()[1] // 7), :],
                                data[:, 2 * (data.size()[1] // 7):3 * (data.size()[1] // 7), :]])
            action_data = torch.stack([data[:, 3 * (data.size()[1] // 7):4 * (data.size()[1] // 7), :],
                                data[:, 4 * (data.size()[1] // 7):5 * (data.size()[1] // 7), :],
                                data[:, 5 * (data.size()[1] // 7):6 * (data.size()[1] // 7), :]])
            map_data = torch.unsqueeze(data[:, 6 * (data.size()[1] // 7):, :], 0)

            position_data = torch.permute(position_data, (1, 0, 2, 3))
            action_data = torch.permute(action_data, (1, 0, 2, 3))
            map_data = torch.permute(map_data, (1, 0, 2, 3))

            # position_data = torch.unsqueeze(position_data, 2)
            # action_data = torch.unsqueeze(action_data, 2)

            position_output = self.activation(self.conv1(position_data))
            position_output = torch.squeeze(position_output)
            position_output = self.batch_norm1(position_output)
            position_output = self.activation(self.conv2(position_output))
            action_output = self.activation(self.conv3(action_data))
            action_output = torch.squeeze(action_output)
            action_output = self.batch_norm2(action_output)
            action_output = self.activation(self.conv4(action_output))

            output = torch.cat((position_output, action_output, map_data), dim=1)
            output = self.activation(self.conv6(output))
            # output = self.dropout(output)
            output = self.batch_norm3(output)
            output = self.activation(self.conv7(output))
            # output = self.dropout(output)
            output = self.batch_norm4(output)
            output = self.activation(self.conv8(output))
            # output = self.dropout(output)
            output = self.flatten(output)
            output = self.activation(self.fc1(output))
            if task_type == "ipp_classification":
                output = self.linear_head(output)
            else:
                output = self.fc2(output)
            if task_type.split("_")[-1] == "classification":
                output = torch.sigmoid(output)

        return output


class ModelMixed(nn.Module):
    def __init__(self, params: Dict):
        super(ModelMixed, self).__init__()
        self.params = params
        self.batch_size = self.params["classification"]["batch_size"]
        self.device = torch.device("cpu")    # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = params["experiment"]["constraints"]["num_actions"]
        self.n_agents = params["experiment"]["missions"]["n_agents"]

        self.activation = torch.nn.ReLU()
        self.batch_norm1 = torch.nn.BatchNorm2d(256)
        self.batch_norm2 = torch.nn.BatchNorm2d(256)
        self.batch_norm3 = torch.nn.BatchNorm1d(256)
        self.dropout = torch.nn.Dropout()

        self.conv1 = torch.nn.Conv2d(1, 256, 4)
        self.conv2 = torch.nn.Conv2d(256, 256, 3)
        self.conv3 = torch.nn.Conv2d(256, 256, 3)
        self.linear1 = nn.Linear(256, 8)
        self.linear2 = nn.Linear(self.n_agents, 4)
        self.fc1 = nn.Linear(12, 256)
        self.fc2 = nn.Linear(256, 1)
        self.fc2_head = nn.Linear(256, self.n_actions)
        self.flatten = nn.Flatten()

    def forward(self, data1, data2, task_type):
        output = self.activation(self.conv1(data1))
        output = self.batch_norm1(output)
        output = self.activation(self.conv2(output))
        output = self.batch_norm2(output)
        output = self.activation(self.conv3(output))
        # output = self.dropout(output)
        output = self.flatten(output)
        hidden_state = self.linear1(output)
        # hidden2 = self.linear2(data2)
        state = torch.cat((hidden_state, data2), dim=1)
        output = self.activation(self.fc1(state))
        output = self.batch_norm3(output)
        # output = self.dropout(output)
        if task_type == "ipp_classification":
            output = self.fc2_head(output)
        else:
            output = self.fc2(output)
        if task_type.split("_")[-1] == "classification":
            output = torch.sigmoid(output)
        return output


class ModelCNN3(nn.Module):
    def __init__(self, params: Dict):
        super(ModelCNN3, self).__init__()
        self.params = params
        self.batch_size = self.params["classification"]["batch_size"]
        self.device = torch.device(
            "cpu")  # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = params["experiment"]["constraints"]["num_actions"]
        self.n_agents = params["experiment"]["missions"]["n_agents"]

        self.activation = torch.nn.ReLU()
        self.batch_norm1 = torch.nn.BatchNorm2d(512)
        self.batch_norm2 = torch.nn.BatchNorm2d(512)
        self.dropout = torch.nn.Dropout()

        self.conv1 = torch.nn.Conv2d(4, 512, 4)
        self.conv2 = torch.nn.Conv2d(512, 512, 3)
        self.conv3 = torch.nn.Conv2d(512, 512, 3)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(512, 512)
        self.linear2 = nn.Linear(512, 1)
        # self.linear_head = nn.Linear(32, self.n_actions)

    def forward(self, data, task_type):
        data = torch.stack([data[:, :(data.size()[1] // 4), :],
                            data[:, (data.size()[1] // 4):2 * (data.size()[1] // 4), :],
                            data[:, 2 * (data.size()[1] // 4):3 * (data.size()[1] // 4), :],
                            data[:, 3 * (data.size()[1] // 4):, :]], dim=1)

        output = self.activation(self.conv1(data))
        # output = self.dropout(output)
        output = self.batch_norm1(output)
        output = self.activation(self.conv2(output))
        # output = self.dropout(output)
        output = self.batch_norm2(output)
        output = self.activation(self.conv3(output))
        # output = self.dropout(output)
        output = self.flatten(output)
        output = self.activation(self.linear1(output))
        if task_type == "ipp_classification":
            output = self.linear_head(output)
        else:
            output = self.linear2(output)
        if task_type.split("_")[-1] == "classification":
            output = torch.sigmoid(output)
        return output


class ModelCNN4(nn.Module):
    def __init__(self, params: Dict):
        super(ModelCNN4, self).__init__()
        self.params = params
        self.batch_size = self.params["classification"]["batch_size"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = params["experiment"]["constraints"]["num_actions"]
        self.n_agents = params["experiment"]["missions"]["n_agents"]

        self.activation = torch.nn.ReLU()
        self.batch_norm1 = torch.nn.BatchNorm2d(256)
        self.batch_norm2 = torch.nn.BatchNorm2d(256)
        self.dropout = torch.nn.Dropout()

        self.conv1 = torch.nn.Conv2d(3, 256, 4)
        self.conv2 = torch.nn.Conv2d(256, 256, 3)
        self.conv3 = torch.nn.Conv2d(256, 256, 3)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(256, 256)
        self.linear2 = nn.Linear(256, 1)
        self.linear_head = nn.Linear(256, self.n_actions)

    def forward(self, data, task_type):
        stack_data = []
        depth = data.size()[1] // data.size()[2]

        for d in range(depth):
            stack_data.append(
                torch.squeeze(data[:, d * (data.size()[1] // depth): (d + 1) * (data.size()[1] // depth), :]))
        data = torch.stack(stack_data, dim=1)

        output = self.activation(self.conv1(data))
        # output = self.dropout(output)
        output = self.batch_norm1(output)
        output = self.activation(self.conv2(output))
        # output = self.dropout(output)
        output = self.batch_norm2(output)
        output = self.activation(self.conv3(output))
        # output = self.dropout(output)
        output = self.flatten(output)
        output = self.activation(self.linear1(output))
        if task_type == "ipp_classification":
            output = self.linear_head(output)
        else:
            output = self.linear2(output)
        if task_type.split("_")[-1] == "classification":
            output = torch.sigmoid(output)
        return output


def get_metrics(classifications, labels, loss_type, task_type, n_actions):
    classifications = [item for sublist in classifications for item in sublist]
    labels = [item for sublist in labels for item in sublist]

    classifications = np.array(classifications)
    labels = np.array(labels)

    if loss_type == "classification":
        if task_type == "collision":
            classifications[classifications < 0.5] = 0
            classifications[classifications >= 0.5] = 1
            precision = precision_score(labels, classifications)
            recall = recall_score(labels, classifications)
            f1 = f1_score(labels, classifications)
            accuracy = accuracy_score(labels, classifications)
            distribution = [int(len(labels) - np.sum(labels)), int(np.sum(labels))]
            classified = int(np.sum(classifications))
            report = confusion_matrix(labels, classifications)
        else:
            classifications = classify(classifications)
            precision = precision_score(labels, classifications, average="weighted")
            recall = recall_score(labels, classifications, average="weighted")
            f1 = f1_score(labels, classifications, average="weighted")
            accuracy = accuracy_score(labels, classifications)
            distribution = get_distribution(labels, n_actions)
            classified = 0
            report = 0
        return precision, recall, f1, accuracy, distribution, classified, report, 0
    elif loss_type == "regression":
        mse = mean_squared_error(labels, classifications)
        mean_absolute_label = sum([abs(ele) for ele in labels]) / len(labels)
        return 0, 0, 0, 0, 0, mean_absolute_label, 0, mse


def get_distribution(labels, n_actions):
    label_counts = np.zeros(n_actions)
    for label in labels:
        index = np.argmax(label)
        label_counts[index] = label_counts[index] + 1
    return label_counts


def classify(outputs):
    for output_index in range(len(outputs)):
        max_output = np.argmax(outputs[output_index])
        outputs[output_index][:] = 0
        outputs[output_index][max_output] = 1
    return outputs


def get_avg_gradient_norm(model):
    total_grad_norm = 0
    number_params = 0
    for params in model.parameters():
        if params.grad is not None:
            total_grad_norm += params.grad.data.norm(1).item()
            number_params += 1
    return total_grad_norm / number_params


def get_network_input(transition):
    return np.squeeze(transition.state.cpu().detach().numpy())


def add_data(params, inputs, labels, batch_memory, i, mapping):
    agent_state_space = AgentStateSpace(params)
    agent_action_space = AgentActionSpace(params)
    n_agents = params["experiment"]["missions"]["n_agents"]
    n_actions = params["experiment"]["constraints"]["num_actions"]

    global_input = batch_memory.concatenated_transitions[i].state

    mlp1 = get_input_mlp1(global_input, n_agents, n_actions, agent_state_space, agent_action_space)
    mlp2 = get_input_mlp2(global_input)
    # cnn1 = get_input_cnn1(mlp1, agent_state_space, n_agents, global_input[1])
    # cnn2 = get_input_cnn2(mlp2, n_agents, agent_state_space, global_input[1])
    # # mixed = get_input_mixed(cnn2, mlp2, n_agents, agent_state_space)
    # cnn3 = get_input_cnn3(mlp2, n_agents, agent_state_space, global_input[1])
    # advantage1 = get_input_cnn2(mlp2, n_agents, agent_state_space)
    cnn4 = get_input_cnn4(mlp2, n_agents, agent_state_space, global_input[1])

    #
    # collision_classification = get_label_collision_classification(mlp1)
    # collision_regression = get_label_collision_regression(mlp1, agent_state_space)
    # collision_advantages = get_label_collision_advantages(mlp2, n_agents, n_actions, agent_state_space,
    #                                                       agent_action_space, params)
    ipp_regression = get_label_ipp_regression(global_input, mlp1, agent_state_space, mapping)
    # ipp_classification = get_label_ipp_classification(global_input, mapping, n_agents, n_actions, agent_action_space,
    #                                                   agent_state_space, params)

    # both_regression = get_label_both_regression(collision_classification, ipp_regression)
    # both_classification = get_label_ipp_classification(global_input, mapping, n_agents, n_actions, agent_action_space, agent_state_space, params)

    # inputs.append([mlp1, mlp2, cnn1, cnn2, cnn3])
    inputs.append([cnn4])   # cnn1, cnn2, cnn3, cnn4])
    # labels.append([collision_classification, collision_regression, ipp_regression, ipp_classification])
    labels.append([ipp_regression])

    return inputs, labels


def get_input_mlp1(global_input, n_agents, n_actions, agent_state_space, agent_action_space):
    position_input = np.asarray(global_input[0][:n_agents * 2])       #
    action_input = np.asarray(global_input[0][n_agents * 2:])         #

    positions = np.zeros_like(position_input)
    positions_3d = []
    for pos_index, position in enumerate(position_input):
        if pos_index % 2 == 0:  #
            positions[pos_index] = (agent_state_space.space_x_dim - 1) * position
        elif pos_index % 2 == 1:    #
            positions[pos_index] = (agent_state_space.space_y_dim - 1) * position
        # elif pos_index % 3 == 2:
        #     positions[pos_index] = (agent_state_space.space_z_dim - 1) * position

    for coordinate in range(len(position_input) // 2):      #
        positions_3d.append(
            agent_state_space.index_to_position(
                [positions[coordinate * 2], positions[coordinate * 2 + 1]]))   # , positions[coordinate * 3 + 2]]))

    new_positions = np.zeros_like(positions)
    for pos_index in range(len(positions_3d)):
        action = n_actions * action_input[pos_index] - 1
        new_position = agent_action_space.action_to_position(np.squeeze(positions_3d[pos_index]), action)
        new_positions[pos_index * 2] = agent_state_space.position_to_index(new_position)[0] / (    #
                agent_state_space.space_x_dim - 1)
        new_positions[pos_index * 2 + 1] = agent_state_space.position_to_index(new_position)[1] / (  #
                agent_state_space.space_y_dim - 1)
        # new_positions[pos_index * 3 + 2] = agent_state_space.position_to_index(new_position)[2] / (
        #         agent_state_space.space_z_dim - 1)
    return new_positions


def get_input_mlp2(global_input):
    return global_input[0]


def get_input_cnn1(new_position_input, agent_state_space, n_agents, w_entropy_map):

    position_map = np.zeros(
        (agent_state_space.space_x_dim, agent_state_space.space_y_dim, agent_state_space.space_z_dim))
    positions = np.zeros_like(new_position_input)
    for pos_index, position in enumerate(new_position_input):
        if pos_index % 3 == 0:
            positions[pos_index] = (agent_state_space.space_x_dim - 1) * position
        elif pos_index % 3 == 1:
            positions[pos_index] = (agent_state_space.space_y_dim - 1) * position
        elif pos_index % 3 == 2:
            positions[pos_index] = (agent_state_space.space_z_dim - 1) * position

    for position in range(len(positions) // 3):
        position_map[int(positions[position * 3]), int(
            positions[position * 3 + 1]), int(positions[position * 3 + 2])] = 1

    return np.vstack([position_map[:, :, 2], position_map[:, :, 1], position_map[:, :, 0], w_entropy_map])


def get_input_cnn2(mlp_input, n_agents, agent_state_space, w_entropy_map):
    position_input = mlp_input[:n_agents * 3]
    action_input = mlp_input[n_agents * 3:]
    positions = np.zeros_like(position_input)
    position_map = np.zeros(
        (agent_state_space.space_x_dim, agent_state_space.space_y_dim, agent_state_space.space_z_dim))
    action_map = np.zeros((agent_state_space.space_x_dim, agent_state_space.space_y_dim, agent_state_space.space_z_dim))
    for pos_index, position in enumerate(position_input):
        if pos_index % 3 == 0:
            positions[pos_index] = (agent_state_space.space_x_dim - 1) * position
        elif pos_index % 3 == 1:
            positions[pos_index] = (agent_state_space.space_y_dim - 1) * position
        elif pos_index % 3 == 2:
            positions[pos_index] = (agent_state_space.space_z_dim - 1) * position

    for position in range(len(positions) // 3):
        position_map[
            int(positions[position * 3]), int(positions[position * 3 + 1]), int(positions[position * 3 + 2])] = 1
        # if position == 0:  ##### CHANGED FOR ADVANTAGE CALCULATION!!!
        #     pass
        # else:
        action_map[int(positions[position * 3]), int(positions[position * 3 + 1]), int(positions[position * 3 + 2])] = \
        action_input[position]

    return np.vstack(
        [position_map[:, :, 2], position_map[:, :, 1], position_map[:, :, 0], action_map[:, :, 2], action_map[:, :, 1],
         action_map[:, :, 0], w_entropy_map])


def get_input_cnn3(mlp_input, n_agents, agent_state_space, w_entropy_map):
    position_input = mlp_input[:n_agents * 3]
    action_input = mlp_input[n_agents * 3:]

    positions = np.zeros_like(position_input)
    position_map = np.zeros((agent_state_space.space_x_dim, agent_state_space.space_y_dim, agent_state_space.space_z_dim))
    for pos_index, position in enumerate(position_input):
        if pos_index % 3 == 0:
            positions[pos_index] = (agent_state_space.space_x_dim - 1) * position
        elif pos_index % 3 == 1:
            positions[pos_index] = (agent_state_space.space_y_dim - 1) * position
        elif pos_index % 3 == 2:
            positions[pos_index] = (agent_state_space.space_z_dim - 1) * position
    for position in range(len(positions) // 3):
        position_map[int(positions[position * 3]), int(positions[position * 3 + 1]), int(positions[position * 3 + 2])] = action_input[position]
    return np.vstack([position_map[:, :, 2], position_map[:, :, 1], position_map[:, :, 0], w_entropy_map])


def get_input_cnn4(mlp_input, n_agents, agent_state_space, w_entropy_map):
    position_input = mlp_input[:n_agents * 2]  #
    action_input = mlp_input[n_agents * 2:]    #
    positions = np.zeros_like(position_input)
    position_map = np.zeros(
        (agent_state_space.space_x_dim, agent_state_space.space_y_dim))
    action_map = np.zeros((agent_state_space.space_x_dim, agent_state_space.space_y_dim))
    for pos_index, position in enumerate(position_input):
        if pos_index % 2 == 0:  #
            positions[pos_index] = (agent_state_space.space_x_dim - 1) * position
        elif pos_index % 2 == 1:  #
            positions[pos_index] = (agent_state_space.space_y_dim - 1) * position
        # elif pos_index % 3 == 2:
        #     positions[pos_index] = (agent_state_space.space_z_dim - 1) * position + 1

    for position in range(len(positions) // 2):  #
        position_map[
            int(positions[position * 2]), int(positions[position * 2 + 1])] = 1   # positions[position * 3 + 2] / agent_state_space.space_z_dim
        # if position == 0:  ##### CHANGED FOR ADVANTAGE CALCULATION!!!
        #     pass
        # else:
        action_map[int(positions[position * 2]), int(positions[position * 2 + 1])] = action_input[position]  #

    return np.vstack([position_map, action_map, w_entropy_map])


def get_input_mixed(position_map, vector_input, n_agents, agent_state_space):
    position_map = position_map[:len(position_map) // 2, :]
    actions = vector_input[n_agents * 2:].tolist() * np.ones((1, agent_state_space.space_y_dim))
    return np.vstack([position_map, actions])


def get_label_collision_classification(new_position_input):
    positions_3d = []
    done = False
    for coordinate in range(len(new_position_input) // 3):
        positions_3d.append([new_position_input[coordinate * 3], new_position_input[coordinate * 3 + 1],
                             new_position_input[coordinate * 3 + 2]])

    for pos1 in range(len(positions_3d)):
        for pos2 in range(pos1):
            if np.array_equal(positions_3d[pos1], positions_3d[pos2]):
                done = True
    return 1 if done else 0


def get_label_collision_regression(new_position_input, agent_state_space):
    positions = np.zeros_like(new_position_input)
    positions_2d = []
    t = []
    for pos_index, position in enumerate(new_position_input):
        if pos_index % 2 == 0:
            positions[pos_index] = (agent_state_space.space_x_dim - 1) * position
        elif pos_index % 2 == 1:
            positions[pos_index] = (agent_state_space.space_y_dim - 1) * position
    for coordinate in range(len(positions) // 2):
        positions_2d.append([positions[coordinate * 2], positions[coordinate * 2 + 1]])
    for pos1 in range(len(positions_2d)):
        for pos2 in range(pos1):
            t.append(np.ceil(manhattan(positions_2d[pos1], positions_2d[pos2]) / 2))
    return min(t).item()


def get_label_collision_advantages(mlp2, n_agents, n_actions, agent_state_space, agent_action_space, params):
    new_other_positions = []
    rewards = []
    for agent in range(1, n_agents):
        other_position = np.array(agent_state_space.index_to_position(
            [(mlp2[2 * agent] * (agent_state_space.space_x_dim - 1)).item(),
             (mlp2[2 * agent + 1] * (agent_state_space.space_y_dim - 1)).item()]))
        new_other_positions.append(
            agent_action_space.action_to_position(other_position, mlp2[n_agents * 2 + agent] * (n_actions - 1)))
    own_position = np.array(agent_state_space.index_to_position([(mlp2[0] * (agent_state_space.space_x_dim - 1)).item(),
                                                                 (mlp2[1] * (
                                                                         agent_state_space.space_y_dim - 1)).item()]))
    n_outliers = 0
    for action in range(n_actions):
        new_position = agent_action_space.action_to_position(own_position, action)
        reward = 1.0
        if not is_in_map(new_position, params):
            n_outliers += 1
            reward = 0.0
        for new_other_position in new_other_positions:
            if np.array_equal(new_position, new_other_position):
                reward = 0.0
                break
        rewards.append(reward)

    taken_action = int((mlp2[n_agents * 2] * n_actions - 1).item())
    received_reward = rewards[taken_action]
    rewards = [r * (1 / (len(rewards) - n_outliers)) for r in rewards]
    advantage = received_reward - sum(rewards)
    return advantage


def get_label_ipp_regression(global_input, new_position_input, agent_state_space, mapping):
    map_state = global_input[2].cpu().detach().numpy()
    simulated_map = global_input[3].cpu().detach().numpy()
    t = global_input[4]

    positions = np.zeros_like(new_position_input)
    positions_3d = []

    for pos_index, position in enumerate(new_position_input):
        if pos_index % 2 == 0:       #
            positions[pos_index] = (agent_state_space.space_x_dim - 1) * position
        elif pos_index % 2 == 1:     #
            positions[pos_index] = (agent_state_space.space_y_dim - 1) * position
        # elif pos_index % 3 == 2:
        #     positions[pos_index] = (agent_state_space.space_z_dim - 1) * position

    for coordinate in range(len(positions) // 2):  #
        positions_3d.append(agent_state_space.index_to_position(
            [positions[coordinate * 2].item(), positions[coordinate * 2 + 1].item()]))   # , positions[coordinate * 3 + 2].item()]))

    updated_maps = []
    for pos_3d in range(len(positions_3d)):
        updated_maps.append(mapping.update_grid_map(np.array(positions_3d[pos_3d]), map_state.copy(), None))
    fused_map = mapping.fuse_map(updated_maps[0], updated_maps[1:])
    mapping_reward = get_utility_reward(map_state, fused_map, simulated_map, agent_state_space)
    return [mapping_reward * 100, t]


def get_label_ipp_classification(global_input, mapping, n_agents, n_actions, agent_action_space, agent_state_space,
                                 params):
    positions_3d = []
    position_input = np.asarray(global_input[0][:n_agents * 3])
    map_state = global_input[2].cpu().detach().numpy()
    own_position = agent_state_space.index_to_position(
        [int(np.round((agent_state_space.space_x_dim - 1) * position_input[0].item(), 2)),
         int(np.round((agent_state_space.space_y_dim - 1) * position_input[1].item(), 2)),
         int(np.round((agent_state_space.space_z_dim - 1) * position_input[2].item(), 2))])
    other_position_input = position_input[3:]

    for coordinate in range(len(other_position_input) // 3):
        positions_3d.append(agent_state_space.index_to_position(
            [int(np.round((agent_state_space.space_x_dim - 1) * other_position_input[coordinate * 3].item(), 2)),
             int(np.round((agent_state_space.space_y_dim - 1) * other_position_input[coordinate * 3 + 1].item(), 2)),
             int(np.round((agent_state_space.space_z_dim - 1) * other_position_input[coordinate * 3 + 2].item(), 2))]))

    mapping_rewards = []
    for action_index in range(n_actions):
        new_position = agent_action_space.action_to_position(own_position, action_index)

        positions_3d.append(new_position)
        updated_map = map_state.copy()

        if is_in_map(new_position, params):
            for pos_3d in range(len(positions_3d)):
                new_map = mapping.update_grid_map(np.array(positions_3d[pos_3d]), updated_map, None)
                updated_map = new_map.copy()
            mapping_rewards.append(
                get_utility_reward(map_state, updated_map, global_input[2], agent_state_space))
        else:
            mapping_rewards.append(0)
        del positions_3d[-1]

    label_vector = np.zeros(n_actions, dtype=np.int8)
    label_vector[np.argmax(np.asarray(mapping_rewards))] = 1
    label_vector = label_vector.tolist()

    return label_vector


def get_label_both_regression(collision_classification, ipp_regression):
    return ipp_regression - (collision_classification - 1)


def is_in_map(position, params):
    x_dim = params["environment"]["x_dim"]
    y_dim = params["environment"]["y_dim"]
    if position[0] <= x_dim and position[0] >= 0 and position[1] <= y_dim and position[1] >= 0:
        return True
    else:
        return False


def manhattan(a, b):
    return sum(abs(val1 - val2) for val1, val2 in zip(a, b))


def store_data(
        path,
        training_input,
        training_labels,
        validation_input,
        validation_labels,
        testing_input,
        testing_labels,
        batch_size,
):
    # tasks = ["collision_classification", "collision_regression", "ipp_regression", "ipp_classification"]
    # input_types = ["mlp1", "mlp2", "cnn1", "cnn2", "cnn3"]
    tasks = ["ipp_regression"]
    input_types = ["cnn4"]
    split_folders = ["training_input", "training_labels", "validation_input", "validation_labels", "testing_input",
                     "testing_labels"]

    for task in tasks:
        os.mkdir(f"{path}/data/{task}")
        for input_type in input_types:
            os.mkdir(f"{path}/data/{task}/{input_type}")
            for split_folder in split_folders:
                os.mkdir(f"{path}/data/{task}/{input_type}/{split_folder}")

    for tr in range(len(training_input)):
        for task_index, task in enumerate(tasks):
            for input_type_index, input_type in enumerate(input_types):
                np.savetxt(path + f"/data/{task}/{input_type}/{split_folders[0]}/training_input_{tr}.txt",
                           training_input[tr][input_type_index], fmt='%s')
                np.savetxt(path + f"/data/{task}/{input_type}/{split_folders[1]}/training_labels_{tr}.txt",
                           np.asarray(training_labels[tr][task_index])) # * np.ones(1))

    for va in range(len(validation_input)):
        for task_index, task in enumerate(tasks):
            for input_type_index, input_type in enumerate(input_types):
                np.savetxt(path + f"/data/{task}/{input_type}/{split_folders[2]}/validation_input_{va}.txt",
                           validation_input[va][input_type_index], fmt='%s')
                np.savetxt(path + f"/data/{task}/{input_type}/{split_folders[3]}/validation_labels_{va}.txt",
                           np.asarray(validation_labels[va][task_index])) #  * np.ones(1))

    for te in range(len(testing_input)):
        for task_index, task in enumerate(tasks):
            for input_type_index, input_type in enumerate(input_types):
                np.savetxt(path + f"/data/{task}/{input_type}/{split_folders[4]}/testing_input_{te}.txt",
                           testing_input[te][input_type_index], fmt='%s')
                np.savetxt(path + f"/data/{task}/{input_type}/{split_folders[5]}/testing_labels_{te}.txt",
                           np.asarray(testing_labels[te][task_index])) #  * np.ones(1))


def read_training_data(path, experiment_run):
    training_inputs = []
    training_labels = []
    validation_inputs = []
    validation_labels = []

    for training_input, training_label in zip(
            sorted(os.listdir(path + experiment_run + "/training_input")),
            sorted(os.listdir(path + experiment_run + "/training_labels")),
    ):
        training_inputs.append(
            np.genfromtxt(
                path + experiment_run + f"/training_input/{training_input}"
            )
        )
        try:
            training_labels.append(
                np.genfromtxt(
                    path + experiment_run + f"/training_labels/{training_label}"
                ).item()
            )
        except:
            with open(path + experiment_run + f'/training_labels/{training_label}') as f:
                label_data = f.readlines()
                label_data = [float(label[:-2]) for label in label_data]
                training_labels.append(label_data)

    for validation_input, validation_label in zip(
            sorted(os.listdir(path + experiment_run + "/validation_input")),
            sorted(os.listdir(path + experiment_run + "/validation_labels")),
    ):
        validation_inputs.append(
            np.genfromtxt(
                path + experiment_run + f"/validation_input/{validation_input}"
            )
        )
        try:
            validation_labels.append(
                np.genfromtxt(
                    path + experiment_run + f"/validation_labels/{validation_label}"
                ).item()
            )
        except:
            with open(path + experiment_run + f'/validation_labels/{validation_label}') as f:
                label_data = f.readlines()
                label_data = [float(label[:-2]) for label in label_data]
                validation_labels.append(label_data)

    return training_inputs, training_labels, validation_inputs, validation_labels


def read_testing_data(path, experiment_run):
    testing_inputs = []
    testing_labels = []

    for testing_input, testing_label in zip(
            sorted(os.listdir(path + experiment_run + "/testing_input")),
            sorted(os.listdir(path + experiment_run + "/testing_labels")),
    ):
        testing_inputs.append(
            np.genfromtxt(
                path + experiment_run + f"/testing_input/{testing_input}"
            )
        )
        try:
            testing_labels.append(
                np.genfromtxt(
                    path + experiment_run + f"/testing_labels/{testing_label}"
                ).item()
            )
        except:
            with open(path + experiment_run + f'/testing_labels/{testing_label}') as f:
                label_data = f.readlines()
                label_data = [float(label[:-2]) for label in label_data]
                testing_labels.append(label_data)
    return testing_inputs, testing_labels


def plot_training_results(
        experiment,
        path,
        training_loss_series,
        training_gradient_series,
        training_precision_series,
        training_recall_series,
        training_f1_score_series,
        training_accuracy_series,
        training_classified_series,
        training_mse_series,
        validation_loss_series,
        validation_gradient_series,
        validation_precision_series,
        validation_recall_series,
        validation_f1_score_series,
        validation_accuracy_series,
        validation_classified_series,
        validation_mse_series,
):
    if not os.path.exists(f"{experiment}{path}/results/figures"):
        os.mkdir(f"{experiment}{path}/results/figures")

    epochs = np.arange(0, len(training_loss_series))

    plt.plot(epochs, training_loss_series, label="training")
    plt.plot(epochs, validation_loss_series, label="validation")
    plt.title(f"{path} - Loss")
    plt.xlim(0, len(training_loss_series))
    # plt.ylim(0, 0.2)
    plt.legend()
    plt.savefig(f"{experiment}{path}/results/figures/loss.png")

    plt.figure()
    plt.plot(epochs, training_gradient_series, label="training")
    plt.plot(epochs, validation_gradient_series, label="validation")
    plt.title(f"{path} - Avg gradient norm")
    plt.xlim(0, len(training_loss_series))
    # plt.ylim(0, 0.5)
    plt.legend()
    plt.savefig(f"{experiment}{path}/results/figures/gradient.png")

    plt.figure()
    plt.plot(epochs, training_precision_series, label="training")
    plt.plot(epochs, validation_precision_series, label="validation")
    plt.title(f"{path} - Precision")
    plt.xlim(0, len(training_loss_series))
    # plt.ylim(0, 0.5)
    plt.legend()
    plt.savefig(f"{experiment}{path}/results/figures/precision.png")

    plt.figure()
    plt.plot(epochs, training_recall_series, label="training")
    plt.plot(epochs, validation_recall_series, label="validation")
    plt.title(f"{path} - Recall")
    plt.xlim(0, len(training_loss_series))
    # plt.ylim(0, 0.5)
    plt.legend()
    plt.savefig(f"{experiment}{path}/results/figures/recall.png")

    plt.figure()
    plt.plot(epochs, training_f1_score_series, label="training")
    plt.plot(epochs, validation_f1_score_series, label="validation")
    plt.title(f"{path} - F1 score")
    plt.xlim(0, len(training_loss_series))
    # plt.ylim(0, 0.5)
    plt.legend()
    plt.savefig(f"{experiment}{path}/results/figures/f1_score.png")

    plt.figure()
    plt.plot(epochs, training_accuracy_series, label="training")
    plt.plot(epochs, validation_accuracy_series, label="validation")
    plt.title(f"{path} - Accuracy")
    plt.xlim(0, len(training_loss_series))
    # plt.ylim(0, 0.5)
    plt.legend()
    plt.savefig(f"{experiment}{path}/results/figures/accuracy.png")

    plt.figure()
    plt.plot(epochs, training_classified_series, label="training")
    plt.plot(epochs, validation_classified_series, label="validation")
    plt.title(f"{path} - Classified")
    plt.xlim(0, len(training_loss_series))
    # plt.ylim(0, 0.5)
    plt.legend()
    plt.savefig(f"{experiment}{path}/results/figures/classified.png")

    plt.figure()
    plt.plot(epochs, training_mse_series, label="training")
    plt.plot(epochs, validation_mse_series, label="validation")
    plt.title(f"{path} - MSE")
    plt.xlim(0, len(training_loss_series))
    # plt.ylim(0, 0.5)
    plt.legend()
    plt.savefig(f"{experiment}{path}/results/figures/mse.png")


def visualize_regression_performance(classifications, labels, phase, epoch, params):
    n_actions = params["experiment"]["constraints"]["num_actions"]

    flat_labels = [label for batch in labels for label in batch]
    flat_classifications = [classification for batch in classifications for classification in batch]

    plt.figure()
    if phase == "Training":
        plt.scatter(flat_labels, flat_classifications, color='g')  # , linestyle="-", linewidth=2, label="training")
    elif phase == "Validation":
        plt.scatter(flat_labels, flat_classifications, color='r')  # , linestyle="-", linewidth=2, label="validation")
    else:
        plt.scatter(flat_labels, flat_classifications, color='orange')  # , linestyle="-", linewidth=2, label="testing")

    plt.plot(flat_labels, flat_labels, color='b', linestyle="--", linewidth=1, label="labels")
    plt.xlabel("true information gain")
    plt.ylabel("estimated information gain")
    # plt.show()
    # plt.legend()
    plt.savefig(
        f"/home/penguin2/jonas-project/Experiment1/data1237_all/ipp_regression/cnn1/plots/means_{phase}_{epoch}.png")


def main():
    constants.log_env_variables()
    params = load_params(constants.CONFIG_FILE_PATH)
    writer = SummaryWriter(constants.LOG_DIR)
    classification = Classification(params, writer)
    classification.execute()


if __name__ == "__main__":
    logger = setup_logger()
    main()
