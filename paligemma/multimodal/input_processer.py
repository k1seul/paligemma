from torchvision import transforms
from transformers import PreTrainedTokenizer
from PIL import Image


IMAGE_TOKENS = "<image>"

class PaliGemmaProcessor:
    def __init__(self, tokenizer : PreTrainedTokenizer, num_image_tokens : int, image_size : int):
        self.tokenizer = tokenizer
        self.preprocess = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

        self.num_image_tokens = num_image_tokens

    def __call__(self, text, images):

        pixel_values = self.preprocess(images)
        
        if len(pixel_values.shape) == 3:
            pixel_values = pixel_values.unsqueeze(0)

        image_tokens = IMAGE_TOKENS * self.num_image_tokens

        prompt = image_tokens + self.tokenizer.bos_token + text + "\n"

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False
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
        num_image_tokens=(16) ** 2,
        image_size=224,
    )

    out = paligemma_processor(text, img)

    print(out["input_ids"])
