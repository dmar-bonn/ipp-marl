import cma
import numpy as np


class CMAESMission:
    def __init__(self):
        self.episode_horizon = 5
        self.map_res = 5
        self.v_max = 5
        self.a_max = 2
        self.simulated_map = np.random.rand(11, 11)
        self.cmaes_sigma0 = [2.0, 2.0, 0, 5]

    def replan(self, mapping):
        self.mapping = mapping
        remaining_budget = 60  # np.random.randint(1, 61)
        previous_action = np.array(
            [
                5 * np.random.randint(0, 9),
                5 * np.random.randint(0, 9),
                5 * np.random.randint(1, 9),
            ]
        )
        current_state = mapping.init_priors()

        greedy_waypoints = self.greedy_search(
            remaining_budget, previous_action, current_state
        )
        greedy_waypoints_utility = -self.simulate_trajectory(
            self.flatten_waypoints(greedy_waypoints)
        )

        if len(greedy_waypoints) == 0:
            return np.array([])

        waypoints = self.cma_es_optimization(greedy_waypoints)
        cma_es_waypoints_utility = -self.simulate_trajectory(waypoints)
        if greedy_waypoints_utility > cma_es_waypoints_utility:
            waypoints = self.flatten_waypoints(greedy_waypoints)
        waypoints = self.stacked_waypoints(waypoints)
        return np.array(waypoints)

    def greedy_search(self, remaining_budget, previous_action, current_state):
        waypoints = []
        simulation_values = []

        for i in range(5):
            action_set = self.get_actions(previous_action, remaining_budget)
            if len(action_set) == 0:
                break
            argmax_action = None
            argmax_state = None
            max_reward = -np.inf
            # simulation_args = [(current_state, previous_action, action) for action in action_set]
            for j in range(len(action_set)):
                simulation_values.append(
                    self.simulate_prediction_step(
                        current_state, previous_action, action_set[j]
                    )
                )

            # simulation_values = self.simulate_prediction_step(np.array(simulation_args)[:,0], np.array(simulation_args[:,1]), np.array(simulation_args[:,2]))

            for simulation_value in simulation_values:
                reward, action, next_state = simulation_value
                if reward > max_reward:
                    max_reward = reward
                    argmax_action = action
                    argmax_state = next_state
            remaining_budget -= self.compute_flight_time(previous_action, argmax_action)
            previous_action = argmax_action
            current_state = argmax_state
            waypoints.append(argmax_action)

        return waypoints

    def get_actions(self, previous_action, remaining_budget):
        action_set = []
        for i in range(9):
            for j in range(9):
                for k in range(9):
                    x_pos = 5 * j
                    y_pos = 5 * i
                    z_pos = 5 + 5 * k
                    action = np.array([x_pos, y_pos, z_pos])
                    if (
                        0
                        < np.float(
                            np.ceil(self.compute_flight_time(previous_action, action))
                        )
                        <= remaining_budget
                    ):
                        action_set.append(action)
        return action_set

    def simulate_prediction_step(self, current_state, previous_action, action):
        next_state = self.mapping.update_grid_map(action, current_state, 5)
        reward = np.mean(
            self.mapping.calculate_w_entropy(
                next_state, self.mapping.init_priors(), "global"
            )
        ) - np.mean(
            self.mapping.calculate_w_entropy(
                current_state, self.mapping.init_priors(), "global"
            )
        )
        return reward, action, next_state

    def cma_es_optimization(self, init_waypoints):
        lower_bounds, upper_bounds, sigma_scales = self.calculate_parameter_bounds_and_scales(
            len(init_waypoints)
        )
        cma_es = cma.CMAEvolutionStrategy(
            self.flatten_waypoints(init_waypoints),
            sigma0=1,
            inopts={
                "bounds": [lower_bounds, upper_bounds],
                "maxiter": 1000,
                "popsize": 10,
                "CMA_stds": sigma_scales,
            },
        )
        return [
            5 * np.random.randint(0, 9),
            5 * np.random.randint(0, 9),
            5 * np.random.randint(1, 9),
        ]

    @staticmethod
    def flatten_waypoints(waypoints):
        flattened = []
        for waypoint in waypoints:
            flattened.extend([waypoint[0], waypoint[1], waypoint[2]])
        return flattened

    def simulate_trajectory(self, waypoints):
        waypoints = self.stacked_waypoints(waypoints)
        for waypoint in waypoints:
            if np.random.randint(0, 10) < np.random.randint(0, 30):
                return 100
        path_consumed_budget = 0
        previous_action = np.array(
            [
                5 * np.random.randint(0, 9),
                5 * np.random.randint(0, 9),
                5 * np.random.randint(1, 9),
            ]
        )
        for waypoint in waypoints:
            path_consumed_budget += self.compute_flight_time(waypoint, previous_action)
            previous_action = waypoint

        if path_consumed_budget <= 0:
            return 100

        total_reward = 0
        remaining_budget = 60  # np.random.randint(1, 61)
        previous_action = np.array(
            [
                5 * np.random.randint(0, 9),
                5 * np.random.randint(0, 9),
                5 * np.random.randint(1, 9),
            ]
        )
        current_state = self.mapping.init_priors()

        for waypoint in waypoints:
            action_cost = self.compute_flight_time(waypoint, previous_action)
            if action_cost > remaining_budget:
                break

            reward, _, next_state = self.simulate_prediction_step(
                current_state, previous_action, waypoint
            )
            total_reward += reward * (action_cost + 1)
            current_state = next_state
            previous_action = waypoint
            remaining_budget -= action_cost

        return -total_reward / path_consumed_budget

    def calculate_parameter_bounds_and_scales(self, num_waypoints):
        lower_bounds = []
        upper_bounds = []
        sigma_scales = []

        lower_z = 5
        upper_x = 9 * 5
        upper_y = 9 * 5
        upper_z = 9 * 5
        sigma_scale_z = min(0.5, (9 * 5 - 5) / 2)

        for i in range(num_waypoints):
            lower_bounds.extend([0, 0, lower_z])
            upper_bounds.extend([upper_x, upper_y, upper_z])
            sigma_scales.extend([2.0, 2.0, sigma_scale_z])

        return lower_bounds, upper_bounds, sigma_scales

    def compute_flight_time(self, start, goal):
        return compute_distance(start, goal) * 2 + np.square(5) / (2 * 5)

    @staticmethod
    def stacked_waypoints(waypoints):
        stacked = []
        for i in range(len(waypoints) // 3):
            stacked.append(
                np.array([waypoints[3 * i], waypoints[3 * i + 1], waypoints[3 * i + 2]])
            )
        return stacked


def compute_distance(start, goal):
    return np.sqrt(
        np.square(goal[0] - start[0])
        + np.square(goal[1] - start[1])
        + np.square(goal[2] - start[2])
    )
