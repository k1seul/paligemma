import torch
import torch.nn as nn
from torchvision import transforms
from transformers import PreTrainedTokenizer
from PIL import Image
from transformers.models.pixtral.image_processing_pixtral import _num_image_tokens


IMAGE_TOKENS = "<image>"

class PaliGemmaProcessor(nn.Module):
    def __init__(self, tokenizer : PreTrainedTokenizer, num_image_tokens : int, image_size : int):
        super().__init__()
        self.tokenizer = tokenizer
        self.preprocess = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

        self.num_image_tokens = num_image_tokens

    def __call__(self, text, images, max_length=None):

        pixel_values = self.preprocess(images)

        image_tokens = IMAGE_TOKENS * self.num_image_tokens

        prompt = image_tokens + text

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }


if __name__ == '__main__':
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

    demo_dir = "../../data/demo/"
    
    with open(demo_dir + "text.txt", "r") as f:
        text = f.readline()

    img = Image.open(demo_dir + "cat.jpg")

    paligemma_processor = PaliGemmaProcessor(
        tokenizer=tokenizer,
        num_image_tokens=256,
        image_size=224,
    )

    out = paligemma_processor(text, img)

    print(out["input_ids"])
