from dataclasses import dataclass

@dataclass
class SiglipVisionConfig:
    hidden_size : int = 768 # embedding dimension
    intermediate_size : int = 3072 # FFN hidden dimension
    num_hidden_layer : int = 12 # number of transformer layers
    num_attention_heads : int = 12 # number of attention heads
    num_channels : int = 3 # number of input image channels
    image_size : int = 224 # image size of one edge (total image size is (image_size, image_size))
    patch_size : int = 16 # size of each patch (patch_size, patch_size)
    layer_norm_eps : float = 1e-6
    attention_dropout : float = 0.0 # attention dropout rate


