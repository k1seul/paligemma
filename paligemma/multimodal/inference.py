import PIL
from PIL import Image
from paligemma.multimodal.input_processer import PaliGemmaProcessor
from paligemma.gemma.kvcache import KVCache
from paligemma.multimodal.model import PaliGemmaForConditionalGeneration


def generate (
    model: PaliGemmaForConditionalGeneration,
    processor: PaliGemmaProcessor,
    prompt: str,
    image: Image.Image,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    do_sample: bool = True
):
    encoded = processor(prompt, image, max_new_tokens)
    kv_cahce = KVCache()

    output, cache = model(input_ids = encoded["input_ids"], 
                          pixel_values = encoded["pixel_values"],
                          kv_cache = )

