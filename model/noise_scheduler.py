import torch
from typing import Tuple


class LinearNoiseScheduler:

    def __init__(self, n_time_steps: int, beta_start: float, beta_end: float):
        self.betas = torch.linspace(beta_start, beta_end, n_time_steps)
        self.alphas = 1 - self.betas
        self.alpha_cum_prod = self.alphas.cumprod(dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)

    def forward(self, image: torch.Tensor, noise: torch.Tensor, t: int) -> torch.Tensor:
        batch_size = image.shape[0]

        batched_sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod[t].reshape(batch_size)
        batched_sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod[t].reshape(batch_size)

        for _ in range(len(image.shape) - 1):
            batched_sqrt_alpha_cum_prod = batched_sqrt_alpha_cum_prod.unsqueeze(-1)
            batched_sqrt_one_minus_alpha_cum_prod = batched_sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)

        return image * batched_sqrt_alpha_cum_prod + noise * batched_sqrt_one_minus_alpha_cum_prod

    def backward_sample(self, noise_image: torch.Tensor, noise_pred: torch.Tensor, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = (noise_image - (self.sqrt_one_minus_alpha_cum_prod[t] * noise_pred)) / self.sqrt_alpha_cum_prod[t]
        image = image.clamp(-1, 1)

        mean = noise_image - ((self.betas[t] * noise_pred) / (self.sqrt_one_minus_alpha_cum_prod[t]))
        mean = mean / torch.sqrt(self.alphas[t])

        if t == 0:
            return image, mean
        else:
            variance = ((1 - self.alpha_cum_prod[t-1]) / (1 - self.alpha_cum_prod[t])) * self.betas[t]
            sigma = torch.sqrt(variance)
            z = torch.randn(image.shape).to(image.device)
            return image + sigma * z, mean
