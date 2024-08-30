import torch

def repeat_layers(module: torch.nn.Module, num_repeats: int) -> torch.nn.Module:
    return torch.nn.ModuleList([module for _ in range(num_repeats)])
