from paligemma.multimodal.inference import generate
from PIL import Image

def test_demo_output(pretrained_model_and_processor):
    model, processor = pretrained_model_and_processor
    prompt = "What is in the image?"
    image = Image.open("/home/seul/Desktop/study/google/paligemma/data/demo/cat.jpg")

    output = generate(
        model = model,
        processor=processor,
        prompt=prompt,
        image=image,
    )

    assert output in ["cat", "kitten"]
