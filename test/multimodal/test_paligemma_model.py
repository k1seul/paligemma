from PIL import Image
import torch
import numpy as np

def test_forward_shape(model_and_processor):
    model, processor = model_and_processor
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    out = processor("describe this image", img)
    logits = model(out["input_ids"].cuda(), out["pixel_values"].cuda())

    assert logits.shape[0] == 1
    assert logits.shape[2] == model.config.text_config.vocab_size

def test_image_tokens_replaced(model_and_processor):
    model, processor = model_and_processor
    img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    out = processor("test", img)
    assert (out["input_ids"] == model.config.img_token_id).sum() == processor.num_image_tokens

def test_pixel_values_range(model_and_processor):
    _, processor = model_and_processor
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    out = processor("test", img)
    pv = out["pixel_values"]
    assert pv.min() >= -1.0 and pv.max() <= 1.0

def test_non_square_image_resize(model_and_processor):
    _, processor = model_and_processor
    portrait_img = Image.fromarray(np.zeros((400, 200, 3), dtype=np.uint8))
    landscape_img = Image.fromarray(np.zeros((200, 600, 3), dtype=np.uint8))
    for img in [portrait_img, landscape_img]:
        out = processor("test", img)
        assert out["pixel_values"].shape == (1, 3, 224, 224)

def test_image_tokens_at_start(model_and_processor):
    model, processor = model_and_processor
    out = processor("hello", Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)))
    ids = out["input_ids"][0]
    image_token_id = model.config.img_token_id
    num_image_tokens = processor.num_image_tokens
    image_positions = (ids == image_token_id).nonzero(as_tuple=True)[0]
    assert image_positions[0].item() == 0
    assert image_positions[-1].item() == num_image_tokens - 1
    assert (ids[num_image_tokens:] != image_token_id).all() # the rest is not image token

def test_eval_model_deterministic(model_and_processor):
    model, processor = model_and_processor
    model.eval()
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    out = processor("test", img)
    ids, pv = out["input_ids"].cuda(), out["pixel_values"].cuda()
    with torch.no_grad():
        logits1 = model(ids, pv)
        logits2 = model(ids, pv)

    assert torch.allclose(logits1, logits2)

def test_batch_size_2(model_and_processor):
    model, processor = model_and_processor
    imgs = [Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)) for _ in range(2)]
    outs = [processor("test", img) for img in imgs]
    input_ids = torch.cat([out["input_ids"] for out in outs], dim=0).cuda()
    pixel_values = torch.cat([out["pixel_values"] for out in outs], dim=0).cuda()
    logits = model(input_ids, pixel_values)
    assert logits.shape[0] == 2
