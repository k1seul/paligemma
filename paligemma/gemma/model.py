from paligemma.gemma.config import GemmaConfig
from paligemma.gemma.modules import GemmaRMSNorm
from paligemma.gemma.decoder import GemmaDecoderLayer
import torch
import torch.nn as nn
import math


class GemmaModel(nn.Module):
    def __init__(self, config : GemmaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.layers = nn.ModuleList([GemmaDecoderLayer(config) for _ in range(self.config.num_hidden_layers)])
        self.rms_norm = GemmaRMSNorm(config)

    def forward(self, input_ids : torch.Tensor, kv_caches = None, attention_mask : torch.Tensor | None = None):
        x = self.embed_tokens(input_ids)
        x = x * math.sqrt(self.config.hidden_size)

        _, seq_len = input_ids.shape

        if attention_mask is None:
            cache_len = kv_caches[0][0].shape[2] if kv_caches is not None else 0
            causal_mask = torch.full((seq_len, seq_len), float('-inf'), device=x.device, dtype=x.dtype).triu(1)
            cache_mask = torch.zeros(seq_len, cache_len, device=x.device, dtype=x.dtype)
            attention_mask = torch.cat([cache_mask, causal_mask], dim=1).unsqueeze(0).unsqueeze(0)

        new_caches = []

        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = layer(x, kv_cache=kv_cache, attention_mask=attention_mask)
            new_caches.append(new_cache)

        return self.rms_norm(x), new_caches


class GemmaForCausalLM(nn.Module):
    def __init__(self, config : GemmaConfig):
        super().__init__()
        self.config = config
        self.model = GemmaModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids : torch.Tensor, kv_cache = None, attention_mask : torch.Tensor | None = None):
        out, new_caches = self.model(input_ids, kv_cache, attention_mask)

        return self.lm_head(out), new_caches


if __name__ == '__main__':
    default_config = GemmaConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_ids = torch.randint(
        low=0,
        high=default_config.vocab_size,
        size=(10, 5)
    ).to(device)

    model = GemmaForCausalLM(default_config).to(device)
    out, caches = model(input_ids)
    print(f"{out.shape}")
    print(out)
