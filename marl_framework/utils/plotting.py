import matplotlib
from matplotlib import cm
import cv2

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import seaborn as sns


def plot_trajectories(
        agent_positions, n_agents, writer, training_step_index, t_collision, budget, simulated_map, map_states
):
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    #
    # # ax = plt.gca(projection='3d')
    #
    # colors = ["c", "g", "m", "orange", "k", "w", "m", "y"]
    #
    # resolution = 0.1014
    #
    # # simulated_map = cv2.resize(
    # #     simulated_map,
    # #     (51, 51),
    # #     interpolation=cv2.INTER_AREA,
    # # )
    #
    # Y, X = np.meshgrid(range(0, 493), range(0, 493))    # 51
    # ax.plot_surface(Y, X, np.zeros_like(simulated_map), facecolors=cm.coolwarm(simulated_map), zorder=1)
    # # plt.colorbar()
    #
    # for agent_id in range(n_agents):
    #     x = []
    #     y = []
    #     z = []
    #     for positions in agent_positions:
    #         x.append(positions[agent_id][0] / resolution)
    #         y.append(positions[agent_id][1] / resolution)
    #         z.append(positions[agent_id][2])
    #
    #     ax.plot(y, x, z, color=colors[agent_id], linestyle="-", linewidth=6, zorder=100)
    #     # ax.plot(y[0], x[0], color=colors[agent_id], marker="o", markersize=14)
    # ax.view_init(40, 50)
    #
    # ax.set_xlim(0, 493)
    # ax.set_ylim(0, 493)
    # ax.set_zlim(0, 15)
    # ax.set_xticks([0, 98.6, 197.2, 295.8, 394.4, 493])
    # ax.set_xticklabels([0, 10, 20, 30, 40, 50])
    # ax.set_yticks([0, 98.6, 197.2, 295.8, 394.4, 493])
    # ax.set_yticklabels([0, 10, 20, 30, 40, 50])
    # ax.set_zticks([5, 10, 15])
    #
    # fig.savefig(f"/home/penguin2/jonas-project/plots/coma_pathes_3d_{training_step_index}.png")
    # # writer.add_figure(f"Agent trajectories", plt.gcf(), training_step_index, close=True)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    colors = ["c", "g", "m", "orange", "k", "w", "brown", "y"]

    resolution = 0.1014

    # plt.colorbar()

    # x = [[], [], [], []]
    # y = [[], [], [], []]
    # z = [[], [], [], []]

    x = [[], [], [], [], [], [], [], []]
    y = [[], [], [], [], [], [], [], []]
    z = [[], [], [], [], [], [], [], []]

    for idx, positions in enumerate(agent_positions):

        Y, X = np.meshgrid(range(0, 493), range(0, 493))  # 51
        ax.plot_surface(Y, X, np.zeros_like(map_states[idx]), facecolors=cm.coolwarm(map_states[idx]), zorder=1)

        for agent_id in range(n_agents):
            x[agent_id].append(positions[agent_id][0] / resolution)
            y[agent_id].append(positions[agent_id][1] / resolution)
            z[agent_id].append(positions[agent_id][2])


        # ax.plot(y[0][0][-1], x[0][0][-1], z[0][0][-1], color="c", linestyle="-", linewidth=6, zorder=100)
        # ax.plot(y[0][1][-1], x[0][1][-1], z[0][1][-1], color="g", linestyle="-", linewidth=6, zorder=100)
        # ax.plot(y[0][2][-1], x[0][2][-1], z[0][2][-1], color="m", linestyle="-", linewidth=6, zorder=100)
        # ax.plot(y[0][3][-1], x[0][3][-1], z[0][3][-1], color="orange", linestyle="-", linewidth=6, zorder=100)

        ax.plot(y[0], x[0], z[0], color="c", marker="o", markersize=6, zorder=100)
        ax.plot(y[1], x[1], z[1], color="g", marker="o", markersize=6, zorder=100)
        ax.plot(y[2], x[2], z[2], color="m", marker="o", markersize=6, zorder=100)
        ax.plot(y[3], x[3], z[3], color="orange", marker="o", markersize=6, zorder=100)

        # ax.plot(y[4], x[4], z[4], color="k", marker="o", markersize=6, zorder=100)
        # ax.plot(y[5], x[5], z[5], color="gray", marker="o", markersize=6, zorder=100)
        # ax.plot(y[6], x[6], z[6], color="brown", marker="o", markersize=6, zorder=100)
        # ax.plot(y[7], x[7], z[7], color="y", marker="o", markersize=6, zorder=100)

        ax.view_init(40, 50)

        ax.set_xlim(0, 493)
        ax.set_ylim(0, 493)
        ax.set_zlim(0, 15)
        ax.set_xticks([0, 98.6, 197.2, 295.8, 394.4, 493])
        ax.set_xticklabels([0, 10, 20, 30, 40, 50])
        ax.set_yticks([0, 98.6, 197.2, 295.8, 394.4, 493])
        ax.set_yticklabels([0, 10, 20, 30, 40, 50])
        ax.set_zticks([5, 10, 15])

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")

        fig.savefig(f"/home/penguin2/jonas-project/plots/coma_pathes_3d_{idx}.png")
        # writer.add_figure(f"Agent trajectories", plt.gcf(), training_step_index, close=True)


def plot_performance(budget, entropies):
    x = list(range(0, budget + 2))
    y = entropies

    plt.plot(x, y)
    np.savetxt("/home/penguin2/jonas-project/plots/learned_new.txt", y, delimiter=",")
    plt.savefig(
        f"/home/penguin2/jonas-project/plots/lawnmower_comparison_uncertainty_reduction.png")
