from paligemma.gemma.config import GemmaConfig
from paligemma.gemma.model import GemmaForCausalLM
from paligemma.gemma.kvcache import KVCache
import pytest
import torch

@pytest.fixture(scope="session")
def model():
    device = "cuda"
    config = GemmaConfig()
    return GemmaForCausalLM(config).to(device)

def test_model_create(model):
    n_params = sum(p.numel() for p in model.parameters())
    print(f"embedding model parameters: {n_params}")

    assert 4_000_000 < n_params < 500_000_000

def test_forward_shape(model):
    device = "cuda"
    input_ids = torch.randint(
        0,
        30000,
        (4, 50)
    ).to(device)

    output = model(input_ids)

    assert output.shape == (4, 50, model.config.vocab_size)

def test_deterministic_in_eval(model):
    device = "cuda"
    input_ids = torch.randint(
        0,
        30000,
        (4, 50)
    ).to(device)
    model.eval()

    with torch.no_grad():
        output1 = model(input_ids)
        output2 = model(input_ids)

    assert torch.allclose(output1, output2)

def test_model_backprop(model):
    device = "cuda"
    input_ids = torch.randint(
        0,
        30000,
        (4, 50)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters())
    optimizer.zero_grad()
    out = model(input_ids)
    
    loss = torch.nn.functional.mse_loss(out, torch.zeros_like(out))
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"NaN gradient in {name}"

def test_output_nan(model):
    device = "cuda"
    input_ids = torch.randint(
        0,
        30000,
        (4, 50)
    ).to(device)

    out = model(input_ids)
    assert torch.isfinite(out).all() 

def test_kv_cache_shape(model):
    device = "cuda"
    input_ids = torch.randint(0, 30000, (2, 5)).to(device)
    kvcache = KVCache()
    _ = model(input_ids, kv_cache=kvcache)

    assert kvcache.num_items == 5

def test_kv_cache_consistency(model):
    device = "cuda"
    input_ids = torch.randint(0, 30000, (2, 6)).to(device)
    model.eval()
    kvcache = KVCache()

    with torch.no_grad():
        full_out =  model(input_ids)
        _ = model(input_ids[:, :5], kv_cache=kvcache)
        next_out = model(input_ids[:, 5:], kv_cache=kvcache)

    assert torch.allclose(full_out[:, 5:], next_out, atol=1e-4)
