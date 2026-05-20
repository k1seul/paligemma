from paligemma.siglip.config import SiglipVisionConifg
from paligemma.siglip.embeddings import SiglipAttention, SiglipVisionEmbeddings, SiglipMLP
import torch
import torch.nn as nn


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConifg, device: str = "cuda"):
        super().__init__()
        self.config = config
        self.device = device
        self.layernorm1 = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps).to(self.device)
        self.layernorm2 = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps).to(self.device)
        self.attention = SiglipAttention(config)
        self.mlp = SiglipMLP(config)

    def forward(self, x: torch.Tensor):
        x = self.layernorm1(x)
        atten_outputs, atten_weights = self.attention(x)
        x = x + atten_outputs
        x = self.layernorm2(x)

        return x + self.mlp(x)


class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConifg, device: str = "cuda"):
        super().__init__()
        self.config = config
        self.device = device
        self.layers = nn.ModuleList([SiglipEncoderLayer(config, device) for _ in range(config.num_hidden_layer)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)

        return x


if __name__ == "__main__":
    config = SiglipVisionConifg(num_hidden_layer=12)
    encoder = SiglipEncoder(config)
    embed = SiglipVisionEmbeddings(config)

    img = torch.randn((10, 3, 224, 224)).to("cuda")
    img_emb = embed(img)
    print(f"input shape: {img.shape}, output shape: {encoder(img_emb).shape}")
