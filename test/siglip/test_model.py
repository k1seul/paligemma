from paligemma.siglip.config import SiglipVisionConfig
from paligemma.siglip.model import SiglipVisionModel
import pytest
import torch

@pytest.fixture
def model():
    device = "cuda"
    default_config = SiglipVisionConfig()
    return SiglipVisionModel(default_config).to(device)

def test_model_create(model):
    n_params = sum(p.numel() for p in model.parameters())
    print(f"embedding model parameters: {n_params}")

    assert 85_000_000 < n_params < 90_000_000

def test_forward_shape(model):
    device = "cuda"
    input_image = torch.randn(30, 3, 224, 224).to(device)
    output = model(input_image)

    assert output.shape == (30, 196, 768)

def test_deterministic_in_eval(model):
    device = "cuda"
    model.to(device)
    model.eval()
    x = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)

    assert torch.allclose(out1, out2)

def test_model_backprop(model):
    device = "cuda"
    model.train()
    x = torch.randn(2, 3, 224, 224).to(device)

    optimizer = torch.optim.Adam(model.parameters())
    optimizer.zero_grad()
    out = model(x)
    loss = torch.nn.functional.mse_loss(out, torch.zeros_like(out))
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"NaN gradient in {name}"

def test_output_nan(model):
    device = "cuda"
    x = torch.randn(2, 3, 224, 224).to(device)

    out = model(x)
    assert torch.isfinite(out).all()
