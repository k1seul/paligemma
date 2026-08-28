from dataclasses import dataclass

@dataclass
class GemmaConfig:
    vocab_size : int = 257216
    hidden_size : int = 2048
    intermediate_size : int = 16384
    num_hidden_layers : int = 18
    num_attention_heads : int = 8
    num_key_value_heads : int = 1
    max_position_embeddings : int = 8192
    rms_norm_eps : float = 1e-6
    rope_theta : float = 10000.0
