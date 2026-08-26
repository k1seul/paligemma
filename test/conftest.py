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


@pytest.fixture(scope="session")
def gemmamodel():
    model = GemmaForCausalLM(GemmaConfig()).to(DEVICE)
    yield model
    _release(model)


@pytest.fixture(scope="session")
def siglipmodel():
    model = SiglipVisionModel(SiglipVisionConfig()).to(DEVICE)
    yield model
    _release(model)


@pytest.fixture(scope="session")
def model_and_processor():
    vision_config = SiglipVisionConfig()
    config = PaligemmaConfig(GemmaConfig(), vision_config)
    model = PaliGemmaForConditionalGeneration(config).to(DEVICE)

    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
    processor = PaliGemmaProcessor(
        tokenizer,
        num_image_tokens=(vision_config.image_size // vision_config.patch_size) ** 2,
        image_size=vision_config.image_size,
    )
    yield model, processor
    _release(model)
