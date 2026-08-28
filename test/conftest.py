import pytest
import torch
from transformers import AutoTokenizer

from paligemma.gemma.config import GemmaConfig
from paligemma.gemma.model import GemmaForCausalLM
from paligemma.siglip.config import SiglipVisionConfig
from paligemma.siglip.model import SiglipVisionModel
from paligemma.multimodal.config import PaligemmaConfig
from paligemma.multimodal.model import PaliGemmaForConditionalGeneration
from paligemma.multimodal.input_processer import PaliGemmaProcessor

DEVICE = "cuda"


def _release(model):
    """Drop a fixture's model and hand the VRAM back to the allocator."""
    del model
    torch.cuda.empty_cache()

def _tiny_gemma():
    return GemmaConfig(
        vocab_size=257216,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4
    )

def _tiny_vision():
    return SiglipVisionConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layer=2,
        num_attention_heads=4,
        patch_size=14
    )


@pytest.fixture(scope="session")
def gemmamodel():
    model = GemmaForCausalLM(_tiny_gemma()).to(DEVICE)
    yield model
    _release(model)


@pytest.fixture(scope="session")
def siglipmodel():
    model = SiglipVisionModel(_tiny_vision()).to(DEVICE)
    yield model
    _release(model)


@pytest.fixture(scope="session")
def model_and_processor():
    vision_config = _tiny_vision()
    config = PaligemmaConfig(_tiny_gemma(), vision_config)
    model = PaliGemmaForConditionalGeneration(config).to(DEVICE)

    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
    processor = PaliGemmaProcessor(
        tokenizer,
        num_image_tokens=(vision_config.image_size // vision_config.patch_size) ** 2,
        image_size=vision_config.image_size,
    )
    yield model, processor
    _release(model)
