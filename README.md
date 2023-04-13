# Multi-UAV Adaptive Path Planning Using Deep Reinforcement Learning

Jonas Westheider, Julius Rückin, Marija Popovic
University of Bonn

This repository contains the implementation of our paper "Multi-UAV Adaptive Path Planning Using Deep Reinforcement Learning" submitted to IROS2023 (under review).


[teaser7.pdf](https://github.com/dmar-bonn/ipp-marl/files/11219440/teaser7.pdf)

![teaser_edit](https://user-images.githubusercontent.com/97049858/231687843-991eb10f-ad77-45a3-86a2-3b378333d064.PNG)


Efficient aerial data collection is important in many remote sensing applications. In large-scale monitoring scenarios, deploying a team of unmanned aerial vehicles (UAVs) offers improved spatial coverage and robustness against individual failures. However, a key challenge is cooperative path planning for the UAVs to efficiently achieve a joint mission goal. We propose a novel multi-agent informative path planning approach based on deep reinforcement learning for adaptive terrain monitoring scenarios using UAV teams. We introduce new network feature representations to effectively learn path planning in a 3D workspace. By leveraging a counterfactual baseline, our approach explicitly addresses credit assignment to learn cooperative behaviour. Our experimental evaluation shows improved planning performance, i.e. maps regions of interest more quickly, with respect to non-counterfactual vari- ants. Results on synthetic and real-world data show that our approach has superior performance compared to state-of-the-art non-learning-based methods, while being transferable to varying team sizes and communication constraints.

Overview of our approach. At each time step during a mission, each UAV takes a measurement and updates its local map state. The local map is input to an actor network, which outputs a policy from which an action is sampled. During training, a centralised critic network is additionally trained using global map information and outputs Q-values for each action from the current state, i.e. the expected future return.

