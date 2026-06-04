from paligemma.siglip.config import SiglipVisionConfig
from paligemma.siglip.embeddings import SiglipVisionEmbeddings
from paligemma.siglip.attention import SiglipAttention, SiglipMLP
import torch
import torch.nn as nn


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layernorm1 = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
        self.layernorm2 = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
        self.attention = SiglipAttention(config)
        self.mlp = SiglipMLP(config)

    def forward(self, x: torch.Tensor):
        residual = x
        hidden = self.layernorm1(x)
        atten_outputs, _ = self.attention(hidden)
        x = residual + atten_outputs
        hidden = self.layernorm2(x)

        return x + self.mlp(hidden)


class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layer)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)

        return x


if __name__ == "__main__":
    config = SiglipVisionConfig(num_hidden_layer=12)
    encoder = SiglipEncoder(config)
    embed = SiglipVisionEmbeddings(config)

    img = torch.randn((10, 3, 224, 224))
    img_emb = embed(img)
    print(f"input shape: {img.shape}, output shape: {encoder(img_emb).shape}")
