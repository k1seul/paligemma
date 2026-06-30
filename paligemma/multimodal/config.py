from paligemma.gemma.config import GemmaConfig
from paligemma.siglip.config import SiglipVisionConfig
from dataclasses import dataclass

@dataclass
class PaligemmaConfig:
    text_config : GemmaConfig
    vision_config : SiglipVisionConfig
    img_token : str = "<image>"
    img_token_id : int = 257152
