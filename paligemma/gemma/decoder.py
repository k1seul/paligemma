from paligemma.gemma.config import GemmaConfig
from paligemma.gemma.attention import GemmaAttention
from paligemma.gemma.modules import GemmaRMSNorm, GemmaMLP
from paligemma.gemma.kvcache import KVCache
import torch
import torch.nn as nn


class GemmaDecoderLayer(nn.Module):
    def __init__(self, config : GemmaConfig, layer_index : int):
        super().__init__()
        self.config = config

        self.rms_norm1 = GemmaRMSNorm(config)
        self.attention = GemmaAttention(config, layer_index)
        self.rms_norm2 = GemmaRMSNorm(config)
        self.mlp = GemmaMLP(config)

    def forward(self, x : torch.Tensor, kv_cache : KVCache | None = None, attention_mask : torch.Tensor | None = None, cache_len : int = 0) -> torch.Tensor:
        hidden = self.rms_norm1(x)
        hidden = self.attention(hidden, kv_cache, attention_mask, cache_len=cache_len)
        hidden = x + hidden
        out = self.rms_norm2(hidden)

        return hidden + self.mlp(out)


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    default_config = GemmaConfig()

    gemma_decoder = GemmaDecoderLayer(default_config, 0).to(device)
    x = torch.randn((100, 200, default_config.hidden_size)).to(device)
    out = gemma_decoder(x)
    print(f"{out.shape:}")
