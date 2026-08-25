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

    def forward(self, input_ids : torch.Tensor, pixel_values : torch.Tensor, kv_cache=None, attention_mask=None):
        text_embedding = self.language_model.model.embed_tokens(input_ids)
        image_embedding = self.vision_tower(pixel_values)

        image_embedding = self.multi_modal_projector(image_embedding)
        image_tokens_mask = (input_ids == self.config.img_token_id)
        text_embedding[image_tokens_mask] = image_embedding.reshape(-1, image_embedding.shape[-1])

        output = self.language_model(
            input_embedding=text_embedding,
            kv_cache=kv_cache,
            attention_mask=attention_mask,
        )
        
        return output


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
        num_image_tokens=196,
        image_size=224,
    )

    out = paligemma_processor(text, img)

    output, cache = model(out["input_ids"], out["pixel_values"])

    print(f"{output.shape}")

