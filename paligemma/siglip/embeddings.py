import torch
import torch.nn as nn
from paligemma.siglip.config import SiglipVisionConifg


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config : SiglipVisionConifg, device : str = "cuda"):
        super().__init__()
        self.config = config
        self.device = device
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
            "position_ids",
            torch.arange(self.num_patchs).expand((1, -1)).to(device),
            persistent=False
        )

    def forward(self, pixel_values):
        patches = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        pos_emb = self.pos_embedding(self.position_ids)

        return patches + pos_emb


class SiglipAttention(nn.Module):
    def __init__(self, config:SiglipVisionConifg, device : str = "cuda"):
        super().__init__()
        self.config = config
        self.device = device

        self.emb_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dims = self.emb_dim // self.num_heads
        self.scale = self.head_dims ** -0.5

        assert self.emb_dim % self.num_heads == 0

        self.k_proj = nn.Linear(self.emb_dim, self.emb_dim).to(device)
        self.q_proj = nn.Linear(self.emb_dim, self.emb_dim).to(device)
        self.v_proj = nn.Linear(self.emb_dim, self.emb_dim).to(device)
        self.out_proj = nn.Linear(self.emb_dim, self.emb_dim).to(device)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, _ = hidden_states.shape

        k_states = self.k_proj(hidden_states)
        q_states = self.q_proj(hidden_states)
        v_states = self.v_proj(hidden_states)

        k_states = k_states.reshape((batch_size, num_tokens, self.num_heads, self.head_dims)).tranpose(1,2)
        q_states = q_states.reshape((batch_size, num_tokens, self.num_heads, self.head_dims))
        v_states = v_states.reshape((batch_size, num_tokens, self.num_heads, self.head_dims))

