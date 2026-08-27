from paligemma.gemma.config import GemmaConfig
import torch
import torch.nn as nn
import torch.nn.functional as F


class GemmaRMSNorm(nn.Module):
    def __init__(self, config : GemmaConfig):
        super().__init__()
        self.eps = config.rms_norm_eps

        self.weight = nn.Parameter(torch.zeros(config.hidden_size))

    def forward(self, x : torch.Tensor):
        x_float = x.float()
        normed = x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = normed * (1.0 + self.weight.float())

        return out.to(dtype=x.dtype)

class GemmaMLP(nn.Module):
    def __init__(self, config : GemmaConfig):
        super().__init__()
        self.config = config

        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        output = (self.up_proj(x) * F.gelu(self.gate_proj(x), approximate="tanh"))

        return self.down_proj(output)


if __name__ == '__main__':
    default_config = GemmaConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn((100, default_config.hidden_size)).to(device)
    rms_norm = GemmaRMSNorm(default_config).to(device)
    mlp = GemmaMLP(default_config).to(device)

    out = rms_norm(x)
    out = mlp(out)

    print(f"{x:} \n {out:}")
    print(f"x shape : {x.shape}, out shape : {out.shape}")
