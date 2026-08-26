from paligemma.gemma.kvcache import KVCache
import torch

def test_model_create(gemmamodel):
    n_params = sum(p.numel() for p in gemmamodel.parameters())
    print(f"embedding gemmamodel parameters: {n_params}")

    assert 4_000_000 < n_params < 500_000_000

def test_forward_shape(gemmamodel):
    device = "cuda"
    input_ids = torch.randint(
        0,
        30000,
        (4, 50)
    ).to(device)

    output = gemmamodel(input_ids)

    assert output.shape == (4, 50, gemmamodel.config.vocab_size)

def test_deterministic_in_eval(gemmamodel):
    device = "cuda"
    input_ids = torch.randint(
        0,
        30000,
        (4, 50)
    ).to(device)
    gemmamodel.eval()

    with torch.no_grad():
        output1 = gemmamodel(input_ids)
        output2 = gemmamodel(input_ids)

    assert torch.allclose(output1, output2)

def test_model_backprop(gemmamodel):
    device = "cuda"
    input_ids = torch.randint(
        0,
        30000,
        (4, 50)
    ).to(device)

    optimizer = torch.optim.Adam(gemmamodel.parameters())
    optimizer.zero_grad()
    out = gemmamodel(input_ids)
    
    loss = torch.nn.functional.mse_loss(out, torch.zeros_like(out))
    loss.backward()

    for name, param in gemmamodel.named_parameters():
        assert param.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"NaN gradient in {name}"

def test_output_nan(gemmamodel):
    device = "cuda"
    input_ids = torch.randint(
        0,
        30000,
        (4, 50)
    ).to(device)

    out = gemmamodel(input_ids)
    assert torch.isfinite(out).all() 

def test_kv_cache_shape(gemmamodel):
    device = "cuda"
    input_ids = torch.randint(0, 30000, (2, 5)).to(device)
    kvcache = KVCache()
    _ = gemmamodel(input_ids, kv_cache=kvcache)

    assert kvcache.num_items == 5

def test_kv_cache_consistency(gemmamodel):
    device = "cuda"
    input_ids = torch.randint(0, 30000, (2, 6)).to(device)
    gemmamodel.eval()
    kvcache = KVCache()

    with torch.no_grad():
        full_out =  gemmamodel(input_ids)
        _ = gemmamodel(input_ids[:, :5], kv_cache=kvcache)
        next_out = gemmamodel(input_ids[:, 5:], kv_cache=kvcache)

    assert torch.allclose(full_out[:, 5:], next_out, atol=1e-4)
