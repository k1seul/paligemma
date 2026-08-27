import torch
import torch.nn as nn
from paligemma.gemma.model import GemmaForCausalLM
from paligemma.siglip.model import SiglipVisionModel
from paligemma.multimodal.projector import PaliGemmaMultiModalProjector
from paligemma.multimodal.config import PaligemmaConfig


class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config : PaligemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.language_model = GemmaForCausalLM(config.text_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)

    def forward(self, input_ids : torch.Tensor, pixel_values : torch.Tensor, kv_cache=None, attention_mask=None, prefix_len : int = -1):
        text_embedding = self.language_model.model.embed_tokens(input_ids)

        if pixel_values is not None:
            image_embedding = self.vision_tower(pixel_values)
            image_embedding = self.multi_modal_projector(image_embedding)
            image_tokens_mask = (input_ids == self.config.img_token_id)
            text_embedding[image_tokens_mask] = image_embedding.reshape(-1, image_embedding.shape[-1])
            seq_len = text_embedding.shape[-2]

        if attention_mask is None:
            cache_len = kv_cache.num_items if kv_cache is not None else 0
            seq_len = input_ids.shape[-1]
            if prefix_len == -1:
                prefix_len = seq_len + cache_len

            attention_mask = build_prefix_lm_mask(
                seq_len, cache_len, prefix_len,
                dtype=text_embedding.dtype,
                device=text_embedding.device
            )

        output = self.language_model(
            input_embedding=text_embedding,
            kv_cache=kv_cache,
            attention_mask=attention_mask,
        )
        
        return output

def build_prefix_lm_mask(seq_len, cache_len, prefix_len, dtype, device):
    q_pos = cache_len + torch.arange(seq_len, device=device)
    k_pos = torch.arange(cache_len + seq_len, device=device)

    allowed = (k_pos[None, :] < prefix_len) | (k_pos[None, :] <= q_pos[:, None])
    mask = torch.zeros(seq_len, cache_len + seq_len, dtype=dtype, device=device)
    mask.masked_fill_(~allowed, float('-inf'))

    return mask[None, None]


if __name__ == '__main__':
    from paligemma.gemma.config import GemmaConfig
    from paligemma.siglip.config import SiglipVisionConfig
    from transformers import AutoTokenizer
    from PIL import Image
    from paligemma.multimodal.input_processer import PaliGemmaProcessor

    gemma_config = GemmaConfig()
    siglip_config = SiglipVisionConfig()
    config = PaligemmaConfig(gemma_config, siglip_config)

    model = PaliGemmaForConditionalGeneration(config)
    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

    demo_dir = "../../data/demo/"

    with open(demo_dir + "text.txt", "r") as f:
        text = f.readline()

    img = Image.open(demo_dir + "cat.jpg")

    paligemma_processor = PaliGemmaProcessor(
        tokenizer=tokenizer,
        num_image_tokens=(224 // siglip_config.patch_size)  ** 2,
        image_size=224,
    )

    out = paligemma_processor(text, img)

    output = model(out["input_ids"], out["pixel_values"])

    print(f"{output.shape}")

