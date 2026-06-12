from paligemma.gemma.config import GemmaConfig
from paligemma.gemma.attention import GemmaAttention
from paligemma.gemma.modules import GemmaRMSNorm, GemmaMLP
import torch
import torch.nn as nn


class GemmaDecoderLayer(nn.Module):
    def __init__(self, config : GemmaConfig):
        super().__init__()
        self.config = config

        self.rms_norm1 = GemmaRMSNorm(config)
        self.attention = GemmaAttention(config)
        self.rms_norm2 = GemmaRMSNorm(config)
        self.mlp = GemmaMLP(config)

    def forward(self, x : torch.Tensor, kv_cache : tuple | None = None, attention_mask : torch.Tensor | None = None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        hidden = self.rms_norm1(x)
        hidden, new_kv_cache = self.attention(hidden, kv_cache, attention_mask)
        hidden = x + hidden
        out = self.rms_norm2(hidden)

        return hidden + self.mlp(out), new_kv_cache


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    default_config = GemmaConfig()

    gemma_decoder = GemmaDecoderLayer(default_config).to(device)
    x = torch.randn((100, 200, default_config.hidden_size)).to(device)
    out, _ = gemma_decoder(x)
    print(f"{out.shape:}")
