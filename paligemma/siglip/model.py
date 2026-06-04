from paligemma.siglip.config import SiglipVisionConfig
from paligemma.siglip.encoder import SiglipEncoder
from paligemma.siglip.embeddings import SiglipVisionEmbeddings
import torch
import torch.nn as nn


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config : SiglipVisionConfig):
        super().__init__()
        self.config = config

        self.embedding = SiglipVisionEmbeddings(self.config)
        self.encoder = SiglipEncoder(self.config)
        self.layernorm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)

    def forward(self, x : torch.Tensor):
        x = self.embedding(x)
        x = self.encoder(x)
        
        return self.layernorm(x)


class SiglipVisionModel(SiglipVisionTransformer):
    def __init__(self, config : SiglipVisionConfig | None):
        if config is None:
            config = SiglipVisionConfig()

        super().__init__(config)


if __name__ == '__main__':
    config = SiglipVisionConfig()
    model = SiglipVisionModel(config)

    img = torch.randn((10, 3, 224, 224)).to("cuda")
    print(model(img).shape)
