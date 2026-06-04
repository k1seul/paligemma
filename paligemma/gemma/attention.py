from paligemma.gemma.config import GemmaConfig
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
    x : (batch_size, seq_len, head_dim, hidden_dim)
    freqs_cis : (max_len, hidden_dim // 2) 
    """

    x_complex = torch.view_as_complex(
        x.reshape(
            *x.shape[:-1],
            -1,
            2
        )
    ) # (batch_size, seq_len, head_dim, hidden_dim // 2, 2) --> complex (batch_size, seq_len, head_dim, hidden_dim // 2 )

    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    x_rot = x_complex * freqs_cis
    x_out = torch.view_as_real(x_rot).flatten(-2)

    return x_out


class GemmaAttention(nn.Module):
    def __init__(self, config : GemmaConfig):
        super().__init__()
        self.config = config

        self.emb_dim = config.hidden_size
        self.num_query_heads = config.num_attention_heads
        self.num_key_value_head = config.num_key_value_heads
        self.head_dim = self.emb_dim // self.num_query_heads
        self.kv_emb_dim = self.head_dim * self.num_key_value_head

        assert self.emb_dim % self.num_query_heads == 0

        self.k_proj = nn.Linear(self.emb_dim, self.kv_emb_dim)
        self.q_proj = nn.Linear(self.emb_dim, self.emb_dim)
        self.v_proj = nn.Linear(self.emb_dim, self.kv_emb_dim)
        self.out_proj = nn.Linear(self.emb_dim, self.emb_dim)

    def forward(self, hidden_states : torch.Tensor) -> torch.Tensor:
        pass


if __name__ == '__main__':
    device = "cuda"
    x = torch.randn((100, 200, 8, 256)).to(device)
    freqs_cis = precompute_freqs_cis(256, 200).to(device)
    out = apply_rotary_emb(x, freqs_cis)

    print(out.shape)
