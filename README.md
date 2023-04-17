# Multi-UAV Adaptive Path Planning Using Deep Reinforcement Learning

Jonas Westheider, Julius Rückin, Marija Popovic
University of Bonn

This repository contains the implementation of our paper [Multi-UAV Adaptive Path Planning Using Deep Reinforcement Learning](https://arxiv.org/pdf/2303.01150.pdf) submitted to IROS2023 (under review). The implementation is currently being cleaned up.


<img src="https://user-images.githubusercontent.com/97049858/231695909-e0ee56fa-2c92-4b32-8176-ba95cf15d8fb.png" width="700" height="400">


## Abstract

Efficient aerial data collection is important in many remote sensing applications. In large-scale monitoring scenarios, deploying a team of unmanned aerial vehicles (UAVs) offers improved spatial coverage and robustness against individual failures. However, a key challenge is cooperative path planning for the UAVs to efficiently achieve a joint mission goal. We propose a novel multi-agent informative path planning approach based on deep reinforcement learning for adaptive terrain monitoring scenarios using UAV teams. We introduce new network feature representations to effectively learn path planning in a 3D workspace. By leveraging a counterfactual baseline, our approach explicitly addresses credit assignment to learn cooperative behaviour. Our experimental evaluation shows improved planning performance, i.e. maps regions of interest more quickly, with respect to non-counterfactual vari- ants. Results on synthetic and real-world data show that our approach has superior performance compared to state-of-the-art non-learning-based methods, while being transferable to varying team sizes and communication constraints.


## Overview


<img src="https://user-images.githubusercontent.com/97049858/231695763-bcb053d2-edf4-4c91-9963-e8e531d0c00d.png" width="700" height="485">


At each time step during a mission, each UAV takes a measurement and updates its local map state. The local map is input to an actor network, which outputs a policy from which an action is sampled. During training, a centralised critic network is additionally trained using global map information and outputs Q-values for each action from the current state, i.e. the expected future return.


## Maintainer

Jonas Westheider, jwestheider@uni-bonn.de, Ph.D. student at PhenoRob - University of Bonn.


## Funding

This work was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany’s Excellence Strategy - EXC 2070 – 390732324. Authors are with the Cluster of Excellence PhenoRob, Institute of Geodesy and Geoinformation, University of Bonn.
