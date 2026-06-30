import torch
import torch.nn as nn
from paligemma.gemma.config import GemmaConfig
from paligemma.siglip.config import SiglipVisionConfig
from paligemma.multimodal.config import PaligemmaConfig


class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config : PaligemmaConfig):
        super().__init__()
        self.fc = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size)

    def forward(self, siglip_output : torch.Tensor):
        return self.fc(siglip_output)


if __name__ == '__main__':
    from paligemma.gemma.model import GemmaForCausalLM
    from paligemma.siglip.model import SiglipVisionModel

    gemma_default_config = GemmaConfig()
    siglip_default_config = SiglipVisionConfig()
    config = PaligemmaConfig(gemma_default_config, siglip_default_config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gemma_model = GemmaForCausalLM(gemma_default_config).to(device)
    siglip_model = SiglipVisionModel(siglip_default_config).to(device)
    projector = PaliGemmaMultiModalProjector(config).to(device)

    input_image = torch.randn(3, 3, 224, 224).to(device)

    siglip_out = siglip_model(input_image)
    projected = projector(siglip_out).to(torch.int64)

    print(projected.shape)
