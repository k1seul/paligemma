from paligemma.siglip.config import SiglipVisionConifg
from paligemma.siglip.embeddings import SiglipVisionEmbeddings
import pytest
import torch

@pytest.fixture
def model():
    deafult_embedding_config = SiglipVisionConifg()
    return SiglipVisionEmbeddings(deafult_embedding_config)

def test_model_create(model):
    n_params = sum(p.numel() for p in model.parameters())
    print(f"embedding model parameters: {n_params}")

    assert n_params < 500_000_000

def test_forward_shape(model):
    input_image = torch.randn(128, 3, 224, 224).to(model.device)
    output = model(input_image)

    assert output.shape == (128, 14 * 14, 768)
