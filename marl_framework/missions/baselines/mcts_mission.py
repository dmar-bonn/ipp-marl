import random_baseline
import time

import numpy as np


class MCTSMission:
    def __init__(self):
        self.budget = 60
        self.init_action = np.array([0, 0, 0])
        # self.n_actions = 294
        self.action_goal = np.array([5, 10, 15])
        self.episode_horizon = 5
        self.num_simulations = 1
        self.min_altitude = 5
        self.max_altitude = 75
        self.altitude_spacing = 5
        self.x_dim = 16
        self.y_dim = 16
        self.planning_res = 5
        self.actions = self.enumerate_actions(
            self.min_altitude, self.max_altitude, self.altitude_spacing
        )
        self.actions_np = self.action_dict_to_np_array(self.actions)
        self.epsilon = 0.05
        self.gamma = 0.99
        self.c = 2.0
        self.alpha = 0.75
        self.k = 4.0
        self.epsilon_expand = 0.2
        self.epsilon_rollout = 0.5

    def execute(self, mapping):
        self.mapping = mapping
        remaining_budget = self.budget
        previous_waypoint = np.array(
            [
                5 * np.random.randint(0, self.x_dim),
                5 * np.random.randint(0, self.y_dim),
                5 * np.random.randint(1, self.max_altitude / self.altitude_spacing),
            ]
        )
        root = Node(
            state=self.mapping.init_priors(), parent=None, action=previous_waypoint
        )
        start_run_time = time.time()
        waypoint = self.replan(root, remaining_budget)
        finish_run_time = time.time()
        print(f"MCTS RUNTIME: {finish_run_time-start_run_time}")

    def replan(self, root, budget):
        root = self.run_simulations_proxy(root, budget)
        best_child = self.select_best_child(root)
        next_waypoint = [0, 0, 0]
        return next_waypoint

    def run_simulations_proxy(self, root, budget):
        for i in range(self.num_simulations):
            self.simulate(root, budget, depth=self.episode_horizon)
        return root

    def simulate(self, node, remaining_budget, depth):
        if node.visits == 0:
            value = self.rollout(node, remaining_budget, node.action, depth)
            node.visits += 1
            node.value_sum += value
            return value
        next_node, expanded = self.progressive_widening(node, remaining_budget)
        if expanded:
            node.children.append(next_node)
        remaining_budget -= self.compute_flight_time(next_node.action, node.action)
        reward = np.mean(
            self.mapping.calculate_w_entropy(
                next_node.state, self.mapping.init_priors(), "global"
            )
        ) - np.mean(
            self.mapping.calculate_w_entropy(
                node.state, self.mapping.init_priors(), "global"
            )
        )
        value = reward + self.simulate(next_node, remaining_budget, depth - 1)
        node.visits += 1
        next_node.visits += 1
        node.value_sum += value
        return value

    def rollout(self, node, remaining_budget, previous_action, depth):
        if depth == 0:
            return 0
        sampled_action = self.eps_greedy_policy(
            node, remaining_budget, self.epsilon_rollout
        )
        next_state = self.prediction_step(node.state, sampled_action)
        remaining_budget -= self.compute_flight_time(sampled_action, previous_action)
        next_node = Node(next_state, node, sampled_action)
        reward = np.mean(
            self.mapping.calculate_w_entropy(
                next_node.state, self.mapping.init_priors(), "global"
            )
        ) - np.mean(
            self.mapping.calculate_w_entropy(
                node.state, self.mapping.init_priors(), "global"
            )
        )
        return reward + self.gamma * self.rollout(
            next_node, remaining_budget, node.action, depth - 1
        )

    def eps_greedy_policy(self, node, remaining_budget, epsilon):
        available_actions = self.actions_np
        if np.random.uniform(0, 1) > epsilon:
            return self.greedy_action(node, available_actions)
        else:
            action_idx = np.random.choice(len(available_actions))
            return available_actions[action_idx]

    def greedy_action(self, node, actions):
        greedy_action = None
        max_reward = -np.inf
        # r = np.random.randint(100, 200)
        # actions = actions[0:r, :]
        for action in actions:
            next_state = self.prediction_step(node.state, action)
            reward = np.mean(
                self.mapping.calculate_w_entropy(
                    next_state, self.mapping.init_priors(), "global"
                )
            ) - np.mean(
                self.mapping.calculate_w_entropy(
                    node.state, self.mapping.init_priors(), "global"
                )
            )
            if reward > max_reward:
                greedy_action = action
                max_reward = reward
        return greedy_action

    def prediction_step(self, state, action):
        next_state = self.mapping.update_grid_map(action, state, 5)
        return next_state

    def compute_flight_time(self, start, goal):
        return compute_distance(start, goal) * 2 + np.square(5) / (2 * 5)

    def enumerate_actions(
        self, min_altitude: float, max_altitude: float, altitude_spacing: float
    ) -> np.array:
        altitude_levels = np.linspace(
            min_altitude,
            max_altitude,
            int((max_altitude - min_altitude) / altitude_spacing) + 1,
        )

        x_meshed, y_meshed = np.meshgrid(
            np.arange(self.x_dim) * self.planning_res,
            np.arange(self.y_dim) * self.planning_res,
        )
        positions = np.array([x_meshed.ravel(), y_meshed.ravel()], dtype=np.float64).T
        actions = {}
        for h, altitude in enumerate(altitude_levels):
            for pos in positions:
                pos_idx2d = pos / self.planning_res
                i = self.flatten_grid_index(pos_idx2d)
                actions[h * (self.x_dim * self.y_dim) + i] = np.array(
                    [pos[0], pos[1], altitude]
                )
        return actions

    def expand(self, node, remaining_budget):
        sampled_action = self.eps_greedy_policy(
            node, remaining_budget, self.epsilon_expand
        )
        next_state = self.prediction_step(node.state, sampled_action)
        return Node(state=next_state, parent=node, action=sampled_action)

    def flatten_grid_index(self, index_2d):
        return int(self.x_dim * index_2d[0] + index_2d[1])

    def action_dict_to_np_array(self, actions):
        actions_np = np.zeros((len(actions), 3))
        for idx, action in actions.items():
            actions_np[idx, :] = action
        return actions_np

    def progressive_widening(self, node, remaining_budget):
        if len(node.children) == 0 or (
            len(node.children) <= self.k * node.visits ** self.alpha
        ):
            return self.expand(node, remaining_budget), True
        return node.select_child(remaining_budget, c=self.c), False

    @staticmethod
    def select_best_child(root):
        best_child = None
        for child in root.children:
            if best_child is None:
                best_child = child
                continue
            if child.value > best_child.value:
                best_child = child

        return best_child


def compute_distance(start, goal):
    return np.sqrt(
        np.square(goal[0] - start[0])
        + np.square(goal[1] - start[1])
        + np.square(goal[2] - start[2])
    )


class Node:
    def __init__(self, state: np.array, parent=None, action=None):
        self.parent = parent
        self.state = state
        self.action = action
        self.value_sum = 0
        self.visits = 0
        self.children = []

    @property
    def value(self):
        return self.value_sum / self.visits

    @property
    def is_leaf(self):
        return len(self.children) == 0

    @staticmethod
    def uct(node, min_val: float, max_val: float, c: float = 2.0):
        if node.visits == 0:
            return np.inf

        exploration = c * np.sqrt(np.log(node.parent.visits) / node.visits)
        if max_val == 0:
            return node.value + exploration

        if max_val == min_val:
            normalized_value = node.value / max_val
        else:
            normalized_value = node.value - min_val / (max_val - min_val)

        return normalized_value + exploration

    def select_child(self, budget: float, c: float = 2.0):
        max_children = []
        max_uct = -np.inf

        children_vals = [child.value for child in self.children]
        min_child_val, max_child_val = min(children_vals), max(children_vals)
        for i in range(len(self.children)):
            uct = self.uct(self.children[i], min_child_val, max_child_val, c=c)
            next_action_costs = MCTSMission.compute_flight_time(
                self.children[i].action, self.action
            )
            if next_action_costs == 0 or next_action_costs >= budget:
                uct = -np.inf

            if max_uct == uct:
                max_children.append(self.children[i])
            elif uct > max_uct:
                max_uct = uct
                max_children = [self.children[i]]

        return random.choice(max_children)
