from paligemma.gemma.config import GemmaConfig
from paligemma.gemma.modules import GemmaRMSNorm
from paligemma.gemma.decoder import GemmaDecoderLayer
from paligemma.gemma.kvcache import KVCache
import torch
import torch.nn as nn
import math


class GemmaModel(nn.Module):
    def __init__(self, config : GemmaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.layers = nn.ModuleList([GemmaDecoderLayer(config, i) for i in range(self.config.num_hidden_layers)])
        self.rms_norm = GemmaRMSNorm(config)

    def forward(self, input_ids : torch.Tensor | None, input_embedding : torch.Tensor | None, kv_cache : KVCache | None = None, attention_mask : torch.Tensor | None = None):
        """
        Gemma takes two inputs
        input_ids : torch.Tensor (int/long) types;
            These are the tokenized integer indices.
            self.embedding is used to get the embedded tensors
        input_embedding : torch.Tensor;
            These are already embedded tensors.
            Directly pass though the transformer layers in this case.
        """

        if input_ids is not None and input_embedding is not None:
            raise ValueError("Only one of the input_ids and input_embedding should be passed here.")

        if input_embedding is None:
            x = self.embed_tokens(input_ids)
        else:
            x = input_embedding
        x = x * math.sqrt(self.config.hidden_size)

        _, seq_len, _ = x.shape
        cache_len = kv_cache.num_items if kv_cache else 0

        if attention_mask is None:
            causal_mask = torch.full((seq_len, seq_len), float('-inf'), device=x.device, dtype=x.dtype).triu(1)
            cache_mask = torch.zeros(seq_len, cache_len, device=x.device, dtype=x.dtype)
            attention_mask = torch.cat([cache_mask, causal_mask], dim=1).unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            x = layer(x, kv_cache=kv_cache, attention_mask=attention_mask, cache_len=cache_len)

        return self.rms_norm(x)


class GemmaForCausalLM(nn.Module):
    def __init__(self, config : GemmaConfig):
        super().__init__()
        self.config = config
        self.model = GemmaModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids : torch.Tensor | None = None, input_embedding : torch.Tensor | None = None, kv_cache : KVCache | None = None, attention_mask : torch.Tensor | None = None):
        out = self.model(input_ids, input_embedding, kv_cache, attention_mask)

        return self.lm_head(out)


if __name__ == '__main__':
    default_config = GemmaConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_ids = torch.randint(
        low=0,
        high=default_config.vocab_size,
        size=(10, 5)
    ).to(device)

    model = GemmaForCausalLM(default_config).to(device)
    out = model(input_ids)
    print(f"{out.shape}")
    print(out)
