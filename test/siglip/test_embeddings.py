import torch
from paligemma.siglip.config import SiglipVisionConfig
from paligemma.siglip.embeddings import SiglipVisionEmbeddings


def test_num_patches_follows_config():
    config = SiglipVisionConfig()
    embedding = SiglipVisionEmbeddings(config)

    assert embedding.num_patches == (config.image_size // config.patch_size) ** 2


def test_forward_shape():
    device = "cuda"
    config = SiglipVisionConfig()
    embedding = SiglipVisionEmbeddings(config).to(device)

    input_image = torch.randn(4, 3, config.image_size, config.image_size).to(device)
    with torch.no_grad():
        output = embedding(input_image)

    assert output.shape == (4, embedding.num_patches, config.hidden_size)


def test_position_embedding_is_added():
    device = "cuda"
    config = SiglipVisionConfig()
    embedding = SiglipVisionEmbeddings(config).to(device)

    x = torch.zeros(1, 3, config.image_size, config.image_size).to(device)
    with torch.no_grad():
        out = embedding(x)
        pos = embedding.pos_embedding(embedding.position_ids)
        bias = embedding.patch_embedding.bias.view(1, 1, -1)

    assert torch.allclose(out, pos + bias, atol=1e-5)
