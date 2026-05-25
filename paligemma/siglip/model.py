from paligemma.siglip.config import SiglipVisionConifg
from paligemma.siglip.encoder import SiglipEncoder
from paligemma.siglip.embeddings import SiglipVisionEmbeddings
import torch
import torch.nn as nn


class SiglipVisionTransformer(nn.Module):
    def __init__(self, device : str = "cuda"):
        super().__init__()
        self.device = device
        self.config = SiglipVisionConifg()

        self.embedding = SiglipVisionEmbeddings(self.config, device)
        self.encoder = SiglipEncoder(self.config, device)
        self.layernorm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps).to(self.device)

    def forward(self, x : torch.Tensor):
        x.to(self.device)
        x = self.embedding(x)
        x = self.encoder(x)
        
        return self.layernorm(x)


class SiglipVisionModel(SiglipVisionTransformer):
    def __init__(self, config : SiglipVisionConifg, device : str = "cuda"):
        nn.Module.__init__(self)
        self.device = device

        self.config = config
        self.embedding = SiglipVisionEmbeddings(self.config, device)
        self.encoder = SiglipEncoder(self.config, device)
        self.layernorm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps).to(self.device)


if __name__ == '__main__':
    config = SiglipVisionConifg()
    model = SiglipVisionModel(config)

    img = torch.randn((10, 3, 224, 224)).to("cuda")
    print(model(img).shape)
