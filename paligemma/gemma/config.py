from dataclasses import dataclass

@dataclass
class GemmaConfig:
    vocab_size : int = 300000
    hidden_size : int = 1024
    intermediate_size : int = 4096
    num_hidden_layers : int = 8
    num_attention_heads : int = 4
    num_key_value_heads : int = 1
    max_position_embeddings : int = 2048
    rms_norm_eps : float = 1e-6
    rope_theta : float = 10000.0
