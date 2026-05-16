from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class SiglipVisionConifg:
    hidden_size : int = 768 # embedding dimension
    intermediate_size : int = 3072 # FFN hidden dimension
    num_hidden_layer : int = 12 # number of transformer layers
    num_attention_heads : int = 12 # number of attention heads
    num_channels : int = 3 # number of input image channels
    image_size : int = 224 # image size of one edge (total image size is (image_size, image_size))
    patch_size : int = 16 # size of each patch (patch_size, patch_size)
    layer_norm_eps : float = 1e-6
    attention_dropout : float = 0.0 # attention dropout rate


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config : SiglipVisionConifg, device : str = "cuda"):
        super().__init__()
        self.config = config
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            padding="valid"
        ).to(device)

        self.num_patchs = (config.image_size // config.patch_size) ** 2
        self.pos_embedding = nn.Embedding(self.num_patchs, config.hidden_size).to(device)

        self.register_buffer(
            "pos_index",
            torch.arange(self.num_patchs).expand((1, -1)).to(device),
            persistent=False
        )

    def forward(self, pixel_values):
        patches = self.patch_embedding(pixel_values).flatten(1).transpose(0, 1)
        pos_emb = self.pos_embedding(self.pos_index)

        return patches + pos_emb


