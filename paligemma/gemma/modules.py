from paligemma.gemma.config import GemmaConfig
import torch
import torch.nn as nn


class GemmaRMSNorm(nn.Module):
    def __init__(self, config : GemmaConfig):
        super().__init__()
        self.config = config

        self.weight = nn.Parameter(torch.empty(self.config.hidden_size))
        self.rms = nn.RMSNorm(self.config.hidden_size, config.rms_norm_eps)

    def forward(self, x : torch.Tensor):
        rms = self.rms(x)
        return x / rms * (1 + self.weight)


if __name__ == '__main__':
    default_config = GemmaConfig()
    x = torch.randn((100, default_config.hidden_size))
    rms_norm = GemmaRMSNorm(default_config)

    out = rms_norm(x)

    print(f"{x:} \n {out:}")
    print(f"x shape : {x.shape}, out shape : {out.shape}")
