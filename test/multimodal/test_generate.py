from transformers.models.got_ocr2.processing_got_ocr2 import preprocess_box_annotation
from paligemma.multimodal.config import PaligemmaConfig, GemmaConfig, SiglipVisionConfig
from paligemma.multimodal.model import PaliGemmaForConditionalGeneration
from paligemma.multimodal.input_processer import PaliGemmaProcessor
from paligemma.multimodal.inference import generate
from paligemma.multimodal import inference
from transformers import AutoTokenizer
from PIL import Image
import pytest
import torch
import numpy as np


@pytest.fixture(scope="session")
def model_and_processor():
    device = "cuda"
    config = PaligemmaConfig(GemmaConfig(), SiglipVisionConfig())
    model = PaliGemmaForConditionalGeneration(config).to(device)
    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens=196, image_size=224)
    return model, processor

def test_generate_return(model_and_processor):
    model, processor = model_and_processor
    prompt = "What is in the image?"
    image = Image.open("/home/seul/Desktop/study/google/paligemma/data/demo/cat.jpg")

    output = generate(model, processor, prompt, image, max_new_tokens=5)

    assert len(output[0]) == 5

def test_doced_feed_single_token(model_and_processor, monkeypatch):
    model, processor = model_and_processor
    image = Image.open("/home/seul/Desktop/study/google/paligemma/data/demo/cat.jpg")

    seq_lens = []
    original_forward = model.forward

    def spy(**kwargs):
        seq_lens.append(kwargs["input_ids"].shape[1])
        return original_forward(**kwargs)

    monkeypatch.setattr(model, "forward", spy)
    generate(model, processor, "test", image, max_new_tokens=4)

    assert seq_lens[0] > 1
    assert all(n == 1 for n in seq_lens[1:])

def test_stop_on_eos(model_and_processor, monkeypatch):
    model, processor = model_and_processor
    image = Image.open("/home/seul/Desktop/study/google/paligemma/data/demo/cat.jpg")
    eos = processor.tokenizer.eos_token_id

    monkeypatch.setattr(
        inference, "top_p_sample",
        lambda logits, top_p : torch.full((logits.shape[0], 1),  eos, dtype=torch.long, device=logits.device)
    )

    output = generate(model, processor, "test", image, max_new_tokens=10)

    assert output[0] == ""


def test_greedy_is_deterministic(model_and_processor):
    model, processor = model_and_processor
    image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

    first = generate(model, processor, "test", image, max_new_tokens=5, do_sample=False)
    second = generate(model, processor, "test", image, max_new_tokens=5, do_sample=False)

    assert first == second

def test_generates_exact_token_count(model_and_processor, monkeypatch):
    model, processor = model_and_processor
    image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    max_new_tokens = 5

    calls = []
    original_forward = model.forward

    def spy(**kwargs):
        calls.append(1)
        return original_forward(**kwargs)

    monkeypatch.setattr(model, "forward", spy)
    generate(model, processor, "test", image, max_new_tokens=max_new_tokens, do_sample=False)

    assert len(calls) == max_new_tokens + 1

