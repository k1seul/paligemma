from paligemma.multimodal.model import PaliGemmaForConditionalGeneration
from paligemma.multimodal.config import SiglipVisionConfig, GemmaConfig, PaligemmaConfig
from safetensors import safe_open
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
import torch
import glob
import re

HUGGINGFACE_MODEL_NAME = "google/paligemma-3b-mix-224"

RULES = [
    # --- vision : HF vision_model. to our implementation ---
    (r'^vision_tower\.vision_model\.embeddings\.position_embedding', 'vision_tower.embedding.pos_embedding'),
    (r'^vision_tower\.vision_model\.embeddings\.', 'vision_tower.embedding.'),
    (r'^vision_tower\.vision_model\.post_layernorm', 'vision_tower.layernorm'),
    (r'^vision_tower\.vision_model\.', 'vision_tower.'),
    (r'\.self_attn\.', '.attention.'),
    (r'\.layer_norm(\d)\.', r'.layernorm\1.'),
    # --- gemma ---
    (r'\.input_layernorm\.', '.rms_norm1.'),
    (r'\.post_attention_layernorm\.', '.rms_norm2.'),
    (r'\.attention\.o_proj\.', '.attention.out_proj.'),
    (r'^language_model\.model\.norm\.', 'language_model.model.rms_norm.'),
    # --- projector ---
    (r'^multi_modal_projector\.linear\.', 'multi_modal_projector.fc.'),
]

def map_key(key : str) -> str:
    for pattern, replacement in RULES:
        key = re.sub(pattern, replacement, key)

    return key

def make_load_tokenizer_and_paligemma_model(device="cuda", dtype=torch.bfloat16):
    snap = snapshot_download(HUGGINGFACE_MODEL_NAME, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(snap)
    vision_config = SiglipVisionConfig()
    gemma_config = GemmaConfig()
    paligemma_config = PaligemmaConfig(
        text_config=gemma_config,
        vision_config=vision_config,
    )

    model = PaliGemmaForConditionalGeneration(paligemma_config)

    for p in model.parameters():
        p.data = p.data.to(dtype=dtype)

    model.to(device=device)
    state = {}

    for shard in glob.glob(f"{snap}/*.safetensors"):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for k in f.keys():
                state[map_key(k)] = f.get_tensor(k).to(dtype=dtype)

    missing, unexpected = model.load_state_dict(state, strict=False)

    assert unexpected == [], f"unpaded keys: {unexpected[:5]}"
    assert missing == ['language_model.lm_head.weight'], f"missing keys: {missing[:5]}"

    return tokenizer, model


if __name__ == '__main__':
    from paligemma.multimodal.inference import generate
    from paligemma.multimodal.input_processer import PaliGemmaProcessor
    from PIL import Image

    tokenizer, model = make_load_tokenizer_and_paligemma_model()

    processor = PaliGemmaProcessor(tokenizer, (224 // model.config.vision_config.patch_size) ** 2, 224)
    prompt = "What is in the image?"
    image = Image.open("/home/seul/Desktop/study/google/paligemma/data/demo/cat.jpg")

    output = generate(
        model = model,
        processor=processor,
        prompt=prompt,
        image=image,
    )

    print(output)
