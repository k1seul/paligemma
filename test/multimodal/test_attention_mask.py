from paligemma.multimodal.model import build_prefix_lm_mask
import torch

DEVICE =  'cuda'

def _allowed(mask):
        return (mask[0, 0] == 0)

def test_no_prefix():
    mask = build_prefix_lm_mask(10, 0, 10, torch.float32, DEVICE)

    assert mask.sum().cpu().float() == 0 

def test_no_cache():
    mask = build_prefix_lm_mask(10, 10, 10, torch.float32, DEVICE)
    
    assert (torch.zeros(10, device=DEVICE)[None, None] == mask[0, 0, :10, :10]).all()

def test_prefix_suffix_pattern():
    mask = build_prefix_lm_mask(5, 0, 3, dtype=torch.float32, device=DEVICE)

    expected = torch.tensor([
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
    ], dtype=torch.bool, 
    device=DEVICE
    )

    assert torch.equal(_allowed(mask), expected)

def test_prefix_len_zero_caual():
    seq_len, cache_len = 4, 3
    
    mask = build_prefix_lm_mask(seq_len, cache_len, 0, dtype=torch.float32, device=DEVICE)

    causal = torch.full((seq_len, seq_len), float('-inf'), dtype=torch.float32, device=DEVICE).triu(1)
    cache = torch.zeros(seq_len, cache_len, dtype=torch.float32, device=DEVICE)
    reference = torch.cat([cache, causal], dim=1)[None, None]

    assert torch.equal(mask, reference)

