from paligemma.gemma.config import GemmaConfig
from paligemma.gemma.kvcache import KVCache
import torch
import torch.nn as nn

def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2] / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)

    freqs_cis = torch.polar(torch.ones_like(freqs) , freqs)

    return freqs_cis

def apply_rotary_emb(x : torch.Tensor, freqs_cis : torch.Tensor):
    """
    x : (batch_size, num_head, seq_len, head_dim)
    freqs_cis : (max_len, hidden_dim // 2) 
    """

    x_complex = torch.view_as_complex(
        x.reshape(
            *x.shape[:-1],
            -1,
            2
        )
    ) # (batch_size, seq_len, head_dim, hidden_dim // 2, 2) --> complex (batch_size, seq_len, head_dim, hidden_dim // 2 )

    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(1)
    x_rot = x_complex * freqs_cis
    x_out = torch.view_as_real(x_rot).flatten(-2)

    return x_out


class GemmaAttention(nn.Module):
    def __init__(self, config : GemmaConfig, layer_index : int):
        super().__init__()
        self.config = config
        self.layer_index = layer_index

        self.emb_dim = config.hidden_size
        self.num_query_heads = config.num_attention_heads
        self.num_key_value_head = config.num_key_value_heads
        self.head_dim = self.emb_dim // self.num_query_heads
        self.kv_emb_dim = self.head_dim * self.num_key_value_head
        self.scale = self.head_dim ** -0.5
        self.num_groups = self.num_query_heads // self.num_key_value_head

        assert self.emb_dim % self.num_query_heads == 0

        self.k_proj = nn.Linear(self.emb_dim, self.kv_emb_dim, bias=False)
        self.q_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.v_proj = nn.Linear(self.emb_dim, self.kv_emb_dim, bias=False)
        self.out_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=False)

        self.register_buffer('freqs_cis', precompute_freqs_cis(self.head_dim, self.config.max_position_embeddings, self.config.rope_theta), persistent=False)

    def forward(self, hidden_states : torch.Tensor, kv_cache : KVCache | None = None, attention_mask : torch.Tensor | None = None, cache_len : int = 0) -> torch.Tensor:
        batch_size, num_tokens, _ = hidden_states.shape

        k_states = self.k_proj(hidden_states)
        q_states = self.q_proj(hidden_states)
        v_states = self.v_proj(hidden_states)
        freq_cis = self.freqs_cis[cache_len : cache_len + num_tokens]
        
        k_states = k_states.reshape(batch_size, num_tokens, self.num_key_value_head, self.head_dim)
        k_states = k_states.transpose(1, 2)
        k_states = apply_rotary_emb(k_states, freq_cis)
        q_states = q_states.reshape((batch_size, num_tokens, self.num_query_heads, self.head_dim)).transpose(1, 2)
        q_states = apply_rotary_emb(q_states, freq_cis)
        v_states = v_states.reshape(batch_size, num_tokens, self.num_key_value_head, self.head_dim)
        v_states = v_states.transpose(1, 2)


        if kv_cache is not None:
            k_states, v_states = kv_cache.update(k_states, v_states, self.layer_index)

        k_expanded = k_states.repeat_interleave(self.num_groups, dim=1)
        v_expanded = v_states.repeat_interleave(self.num_groups, dim=1)

        key_states_T = k_expanded.transpose(2, 3)
        attention_weights = q_states @ key_states_T * self.scale

        if attention_mask is not None:
            attention_weights += attention_mask

        attention_weights = nn.functional.softmax(attention_weights, dim=-1)
        attention_outputs = attention_weights @ v_expanded
        attention_outputs = attention_outputs.transpose(1, 2).contiguous()
        attention_outputs = attention_outputs.reshape(batch_size, num_tokens, self.emb_dim)

        attention_outputs = self.out_proj(attention_outputs)

        return attention_outputs


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = GemmaConfig()
    x = torch.randn((100, 200, 1024)).to(device)
    attention = GemmaAttention(config, 1).to(device)
    out = attention(x)
    print(f"{out[0].shape:}")

