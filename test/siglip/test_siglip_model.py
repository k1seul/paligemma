import torch

def test_model_create(siglipmodel):
    n_params = sum(p.numel() for p in siglipmodel.parameters())
    print(f"siglipmodel parameters: {n_params}")

    assert 300_000 < n_params < 400_000

def test_forward_shape(siglipmodel):
    device = "cuda"
    config = siglipmodel.config
    num_patches = (config.image_size // config.patch_size) ** 2
    input_image = torch.randn(4, 3, config.image_size, config.image_size).to(device)

    with torch.no_grad():
        output = siglipmodel(input_image)

    assert output.shape == (4, num_patches, config.hidden_size)

def test_deterministic_in_eval(siglipmodel):
    device = "cuda"
    siglipmodel.eval()
    x = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        out1 = siglipmodel(x)
        out2 = siglipmodel(x)

    assert torch.allclose(out1, out2)

def test_model_backprop(siglipmodel):
    device = "cuda"
    siglipmodel.train()
    x = torch.randn(2, 3, 224, 224).to(device)

    out = siglipmodel(x)
    loss = torch.nn.functional.mse_loss(out, torch.zeros_like(out))
    loss.backward()

    try:
        for name, param in siglipmodel.named_parameters():
            assert param.grad is not None, f"no gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"NaN gradient in {name}"
    finally:
        siglipmodel.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

def test_output_nan(siglipmodel):
    device = "cuda"
    x = torch.randn(2, 3, 224, 224).to(device)

    with torch.no_grad():
        out = siglipmodel(x)
    assert torch.isfinite(out).all()
