import torch
import torch.nn.functional as F
from PIL import Image
from paligemma.multimodal.input_processer import PaliGemmaProcessor
from paligemma.gemma.kvcache import KVCache
from paligemma.multimodal.model import PaliGemmaForConditionalGeneration

def top_p_sample(logits : torch.Tensor, top_p : float):
    logits_sorted, logits_indicies = torch.sort(logits, dim=-1, descending=True)

    sorted_probs = torch.softmax(logits_sorted, dim=-1)
    cumulated_probs = torch.cumsum(sorted_probs, dim=-1)
    remove = cumulated_probs > top_p
    remove[:, 1:] = remove[:, :-1].clone()
    remove[:, 0] = False

    logits_sorted[remove] = float("-inf")
    filtered_logits = torch.full_like(logits_sorted, float("-inf"))
    filtered_logits = filtered_logits.scatter(1, logits_indicies, logits_sorted)
    probs = torch.softmax(filtered_logits, dim=-1)

    return torch.multinomial(probs, num_samples=1)

@torch.no_grad()
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
    model.eval()
    encoded = processor(prompt, image)
    kv_cache = KVCache()
    generated = []
    device = next(model.parameters()).device
    encoded["input_ids"] = encoded["input_ids"].to(device)
    encoded["pixel_values"] = encoded["pixel_values"].to(device)

    output = model(input_ids = encoded["input_ids"], 
                          pixel_values = encoded["pixel_values"],
                          kv_cache=kv_cache)

    logits = output[:, -1, :]
    logits = logits / temperature

    if do_sample:
        idx = top_p_sample(logits, top_p)
    else:
        idx = torch.argmax(logits, dim=-1, keepdim=True)

    for _ in range(max_new_tokens):
        if idx == processor.tokenizer.eos_token_id:
            break
        generated.append(idx)
        output = model(input_ids=idx,
                       pixel_values=None,
                       kv_cache=kv_cache)
        logits = output[:, -1, :] / temperature
        idx = top_p_sample(logits, top_p) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)

    if not generated:
        return [""]

    generated = torch.cat(generated, dim=-1)
    return processor.tokenizer.decode(generated, skip_special_tokens=True)

if __name__ == "__main__":
    from paligemma.multimodal.config import PaligemmaConfig, GemmaConfig, SiglipVisionConfig
    from transformers import AutoTokenizer

    siglipconfig = SiglipVisionConfig()
    gemmaconfig = GemmaConfig()
    config = PaligemmaConfig(text_config=gemmaconfig,
                             vision_config=siglipconfig)
    
    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
    processor = PaliGemmaProcessor(tokenizer=tokenizer,
                                   num_image_tokens=196,
                                   image_size=224,)

    model = PaliGemmaForConditionalGeneration(config).to("cuda")
    prompt = "What is in the image?"
    image = Image.open("/home/seul/Desktop/study/google/paligemma/data/demo/cat.jpg")

    output = generate(model=model,
             processor=processor,
             prompt=prompt,
             image=image)

    print(output)
