import torch
import torch.nn as nn
from paligemma.siglip.config import SiglipVisionConfig


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config : SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            padding="valid"
        )

        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.pos_embedding = nn.Embedding(self.num_patches, config.hidden_size)

        self.register_buffer(
            "position_ids",
            torch.arange(self.num_patches).expand((1, -1)),
            persistent=False
        )

    def forward(self, pixel_values):
        patches = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        pos_emb = self.pos_embedding(self.position_ids)

        return patches + pos_emb


if __name__ == "__main__":
    default_config = SiglipVisionConfig()
    embedding = SiglipVisionEmbeddings(default_config)

    img = torch.randn(100, 3, 224, 224)
    embed = embedding(img)
