import torch


def polyak_averaging(target_net: torch.nn.Module, net: torch.nn.Module, tau: float):
    for target_param, param in zip(target_net.parameters(), net.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
