> **Goal**: Deeply understand the internals of a Vision-Language Model by reimplementing PaliGemma (SigLIP + Gemma) entirely in PyTorch from scratch.
>
> **Workflow**: Read each problem → Write your own code → Compare with the reference implementation

Entire codebase, except the README.md file, were written by hand coding.

---

## 📁 Target File Structure

```
paligemma/
├── siglip/
│   ├── config.py          # SigLIP config
│   ├── embeddings.py      # Patch embeddings
│   ├── encoder.py         # Transformer encoder
│   └── model.py           # Full SigLIP model
├── gemma/
│   ├── config.py          # Gemma config
│   ├── attention.py       # GQA + RoPE
│   ├── mlp.py             # GeGLU FFN
│   ├── decoder.py         # Decoder layer
│   └── model.py           # Full Gemma model
├── projector.py           # Multimodal linear projector
├── processor.py           # Image + text preprocessing
├── model.py               # PaliGemma integrated model
└── inference.py           # Inference script
```

---

## CHAPTER 1 — SigLIP: Vision Encoder

> SigLIP (Sigmoid Loss for Language-Image Pre-training) is PaliGemma's visual front-end.
> Built on the ViT (Vision Transformer) architecture, it converts an image into a sequence of patch embeddings.

---

### 🟢 Problem 1-1: SigLIP Config Class

**Requirements**

Create a `SiglipVisionConfig` class (dataclass or plain class) that holds the following hyperparameters:

| Parameter | Description | Default |
|---|---|---|
| `hidden_size` | Embedding dimension | 768 |
| `intermediate_size` | FFN hidden dimension | 3072 |
| `num_hidden_layers` | Number of Transformer layers | 12 |
| `num_attention_heads` | Number of attention heads | 12 |
| `num_channels` | Number of image channels | 3 |
| `image_size` | Input image resolution | 224 |
| `patch_size` | Patch size | 16 |
| `layer_norm_eps` | LayerNorm epsilon | 1e-6 |
| `attention_dropout` | Attention dropout rate | 0.0 |

**Hint**: `num_patches = (image_size // patch_size) ** 2`

---

### 🟢 Problem 1-2: Patch Embedding

**Requirements**

Implement `SiglipVisionEmbeddings(nn.Module)`.

- Input: `pixel_values` — shape `(B, C, H, W)`
- Output: patch embeddings — shape `(B, num_patches, hidden_size)`
- Use a single **Conv2d** to extract all patches at once
  - `kernel_size = patch_size`, `stride = patch_size`
- Unlike vanilla ViT, SigLIP has **no [CLS] token**
- Add a learnable 2D position embedding via `nn.Embedding`

**Key question**: How does `nn.Conv2d` implement patch embedding? Why is it mathematically equivalent to `nn.Linear` applied to each patch?

---

### 🟡 Problem 1-3: Multi-Head Attention

**Requirements**

Implement `SiglipAttention(nn.Module)`.

- Projections: `q_proj`, `k_proj`, `v_proj`, `out_proj` — all `nn.Linear`
- Scaled Dot-Product Attention: `softmax(QK^T / sqrt(d_k)) V`
- Multi-head reshape: `(B, seq_len, hidden) → (B, num_heads, seq_len, head_dim)`
- SigLIP Vision Encoder is **bidirectional** — no causal mask
- Apply attention dropout

**Implementation steps**
```
1. Project Q, K, V
2. Split into heads: reshape + transpose
3. Compute attention scores
4. Softmax + Dropout
5. Weighted sum over V, then merge heads
6. Apply out_proj
```

---

### 🟡 Problem 1-4: MLP (Feed-Forward Network)

**Requirements**

Implement `SiglipMLP(nn.Module)`.

- `fc1`: `hidden_size → intermediate_size`
- `fc2`: `intermediate_size → hidden_size`
- Activation: **GELU** — use `nn.GELU(approximate='tanh')`

---

### 🟡 Problem 1-5: Encoder Layer & Encoder

**Requirements**

① `SiglipEncoderLayer(nn.Module)` — a single Transformer layer

```
x → LayerNorm → Attention → residual +
  → LayerNorm → MLP       → residual +
```

② `SiglipEncoder(nn.Module)` — stack of N layers

- `self.layers = nn.ModuleList([SiglipEncoderLayer(...) for _ in range(num_hidden_layers)])`

**Key question**: What is the difference between Pre-LayerNorm and Post-LayerNorm? Which one does SigLIP use?

---

### 🔴 Problem 1-6: Complete SigLIP Vision Model

**Requirements**

Implement `SiglipVisionTransformer(nn.Module)` and `SiglipVisionModel(nn.Module)`.

- `SiglipVisionTransformer`:
  - `embeddings` → `encoder` → `post_layernorm`
  - Output: `last_hidden_state` of shape `(B, num_patches, hidden_size)`

- `SiglipVisionModel`: a thin wrapper around `SiglipVisionTransformer` that accepts a config

**Verify the final data flow**:
```
(B, 3, 224, 224)
  → SiglipVisionEmbeddings → (B, 196, 768)
  → SiglipEncoder × 12     → (B, 196, 768)
  → LayerNorm               → (B, 196, 768)
```

---

## CHAPTER 2 — Gemma: Language Model Decoder

> Gemma is PaliGemma's language brain. It is a Decoder-Only Transformer (GPT-style)
> that uses Grouped Query Attention (GQA) and RoPE positional encoding.

---

### 🟢 Problem 2-1: Gemma Config Class

**Requirements**

Create a `GemmaConfig` class with the following parameters:

| Parameter | Description | Default |
|---|---|---|
| `vocab_size` | Vocabulary size | 257152 |
| `hidden_size` | Model dimension | 2048 |
| `intermediate_size` | FFN hidden dimension | 16384 |
| `num_hidden_layers` | Number of layers | 18 |
| `num_attention_heads` | Number of query heads | 8 |
| `num_key_value_heads` | Number of KV heads (GQA) | 1 |
| `head_dim` | Dimension per head | 256 |
| `max_position_embeddings` | Maximum sequence length | 8192 |
| `rms_norm_eps` | RMSNorm epsilon | 1e-6 |
| `rope_theta` | RoPE base frequency | 10000.0 |

---

### 🟢 Problem 2-2: RMSNorm

**Requirements**

Implement `GemmaRMSNorm(nn.Module)`.

Formula:
```
RMSNorm(x) = x / RMS(x) * weight
RMS(x) = sqrt(mean(x^2) + eps)
```

- Unlike LayerNorm: **no mean subtraction** (no centering)
- `weight`: learnable parameter of shape `(hidden_size,)`
- Gemma adds 1 to the weight before scaling: `output * (1 + self.weight)`

---

### 🟡 Problem 2-3: RoPE (Rotary Position Embedding)

**Requirements**

Implement `precompute_freqs_cis` and `apply_rotary_emb`.

**Concept**: RoPE encodes relative position by rotating Q and K vectors by a position-dependent angle.

① `precompute_freqs_cis(dim, max_seq_len, theta=10000.0)`
```python
# Compute per-dimension frequencies
theta_i = 1 / (theta ** (2i / dim))  # i = 0, 1, ..., dim/2-1
# Outer product with positions
freqs = outer(positions, theta_i)
# Convert to complex numbers (cos + i*sin)
freqs_cis = complex(cos(freqs), sin(freqs))
```

② `apply_rotary_emb(x, freqs_cis)`
- Interpret x as complex: `(x_r, x_i)` → apply rotation → convert back to real

**Key question**: How does RoPE differ from Absolute PE and Relative PE? Why does it generalize better to longer sequences (extrapolation)?

---

### 🔴 Problem 2-4: Grouped Query Attention (GQA)

**Requirements**

Implement `GemmaAttention(nn.Module)`.

**GQA concept**:
- Query heads: `num_attention_heads = 8`
- KV heads: `num_key_value_heads = 1`
- Multiple query heads **share a single KV head**
- → Reduces KV Cache memory footprint

**Key implementation**:
```python
# Expand KV heads to match query head count
num_groups = num_attention_heads // num_key_value_heads
key = key.repeat_interleave(num_groups, dim=1)
value = value.repeat_interleave(num_groups, dim=1)
```

**Must include**:
- `q_proj`, `k_proj`, `v_proj`, `o_proj`
- RoPE applied to Q and K only
- KV Cache support: manage `cache_k` and `cache_v`
- Causal attention mask support

---

### 🟡 Problem 2-5: GeGLU MLP

**Requirements**

Implement `GemmaMLP(nn.Module)`.

Unlike a standard FFN, this uses a **Gated Linear Unit** structure:
```
output = (gate_proj(x) * GELU(up_proj(x))) @ down_proj
```

- `gate_proj`: `hidden_size → intermediate_size`
- `up_proj`: `hidden_size → intermediate_size`
- `down_proj`: `intermediate_size → hidden_size`

**Key question**: What is the advantage of GLU-family activations over a standard FFN?

---

### 🟡 Problem 2-6: Gemma Decoder Layer & Full Model

**Requirements**

① `GemmaDecoderLayer(nn.Module)`:
```
x → RMSNorm → GemmaAttention → residual +
  → RMSNorm → GemmaMLP       → residual +
```

② `GemmaModel(nn.Module)`:
- `embed_tokens`: `nn.Embedding(vocab_size, hidden_size)`
- `layers`: N × `GemmaDecoderLayer`
- `norm`: final `GemmaRMSNorm`
- Scale embeddings by `sqrt(hidden_size)` (Gemma-specific normalization)

③ `GemmaForCausalLM(nn.Module)`:
- `GemmaModel` + `lm_head` (`nn.Linear`, no bias)
- Tie weights: `lm_head.weight = embed_tokens.weight`

---

## CHAPTER 3 — Projector & Integration

> The Vision Encoder and Language Model operate in different embedding spaces.
> A Linear Projector bridges the two by mapping from the vision dimension to the language dimension.

---

### 🟡 Problem 3-1: Multimodal Linear Projector

**Requirements**

Implement `PaliGemmaMultiModalProjector(nn.Module)`.

- Input: SigLIP output — shape `(B, num_patches, siglip_hidden_size)` = `(B, 256, 768)`
- Output: Gemma input — shape `(B, num_patches, gemma_hidden_size)` = `(B, 256, 2048)`
- Architecture: **a single Linear layer** (with bias)

---

### 🔴 Problem 3-2: Input Processor

**Requirements**

Implement the `PaliGemmaProcessor` class.

**Role**: Convert raw image + text into model-ready tensors.

```python
class PaliGemmaProcessor:
    def __init__(self, tokenizer, num_image_tokens, image_size):
        ...

    def __call__(self, text, images, max_length=None):
        # 1. Preprocess image: resize → normalize → tensor
        # 2. Prepend image placeholders to text:
        #    "<image>" * num_image_tokens + text
        # 3. Tokenize
        # 4. Return attention_mask and input_ids
```

**Key points**:
- Image token placeholder: `IMAGE_TOKEN = "<image>"`
- Input sequence layout: `[img_token × 256][BOS][text_tokens][EOS]`
- Image normalization: `mean=[0.5, 0.5, 0.5]`, `std=[0.5, 0.5, 0.5]`

---

### 🔴 Problem 3-3: PaliGemma Integrated Model

**Requirements**

Implement `PaliGemmaForConditionalGeneration(nn.Module)`.

```python
class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config):
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.language_model = GemmaForCausalLM(config.text_config)
```

**`forward()` implementation steps**:
1. Extract text embeddings from `input_ids`
2. Encode `pixel_values` with SigLIP → image features
3. Project image features to language model dimension via the projector
4. **Insert** image embeddings at the correct positions in the text embedding sequence
   - Replace positions marked by `IMAGE_TOKEN_INDEX` with image embeddings
5. Feed the merged embeddings into Gemma and return logits

**Key question**: Where exactly in the token sequence are image tokens inserted? What is the prefix-LM approach?

---

## CHAPTER 4 — Inference Pipeline

---

### 🔴 Problem 4-1: KV Cache

**Requirements**

Implement the `KVCache` class.

```python
class KVCache:
    def __init__(self):
        self.key_cache: List[Tensor] = []
        self.value_cache: List[Tensor] = []

    def update(self, key, value, layer_idx):
        # Accumulate K and V tensors per layer
        ...

    def get(self, layer_idx):
        ...

    @property
    def num_items(self):
        # Return the number of tokens currently cached
        ...
```

**Key question**: Without KV Cache, what redundant computation happens during autoregressive generation? What is the time complexity with and without it?

---

### 🔴 Problem 4-2: Autoregressive Text Generation

**Requirements**

Implement the `generate()` function.

```python
def generate(
    model,
    processor,
    prompt: str,
    image: PIL.Image,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    do_sample: bool = True,
):
```

**Implementation steps**:
1. Preprocess inputs with the Processor
2. **Prefill phase**: process the full image + prompt in one forward pass, initialize KV Cache
3. **Decode phase**: process one new token at a time
   - Extract `logits[:, -1, :]`
   - Apply temperature scaling
   - Top-p (nucleus) sampling or greedy decoding
   - Append new token to input, update KV Cache
4. Stop when EOS token is generated
5. Decode the generated token sequence and return the string

---

### 🟡 Problem 4-3: Loading Pretrained Weights

**Requirements**

Load PaliGemma pretrained weights from HuggingFace and apply them to your custom model.

```python
def load_hf_model(model_path: str, device: str):
    # 1. Load safetensors files
    # 2. Map HF state_dict keys to your model's keys
    # 3. Call load_state_dict()
```

**Hints**:
- Use the `safetensors` library
- Inspect HF parameter names via: `from transformers import PaliGemmaForConditionalGeneration`

---

## 📚 Progress Checklist

### CHAPTER 1 — SigLIP
- [X] 1-1 SiglipVisionConfig
- [X] 1-2 SiglipVisionEmbeddings (patch embedding)
- [X] 1-3 SiglipAttention (Multi-Head Attention)
- [X] 1-4 SiglipMLP (GELU FFN)
- [X] 1-5 SiglipEncoderLayer + SiglipEncoder
- [X] 1-6 SiglipVisionModel + shape verification

### CHAPTER 2 — Gemma
- [X] 2-1 GemmaConfig
- [X] 2-2 GemmaRMSNorm
- [X] 2-3 RoPE (precompute + apply)
- [X] 2-4 GemmaAttention (GQA + KV Cache)
- [X] 2-5 GemmaMLP (GeGLU)
- [ ] 2-6 GemmaModel + GemmaForCausalLM

### CHAPTER 3 — Integration
- [ ] 3-1 MultiModalProjector
- [ ] 3-2 PaliGemmaProcessor
- [ ] 3-3 PaliGemmaForConditionalGeneration

### CHAPTER 4 — Inference
- [ ] 4-1 KVCache
- [ ] 4-2 generate() function
- [ ] 4-3 Load HF weights and run real inference

---

## 🔗 References

| Resource | Link |
|---|---|
| PaliGemma Paper | [arxiv.org/abs/2407.07726](https://arxiv.org/abs/2407.07726) |
| SRDdev original implementation | [github.com/SRDdev/PaliGemma](https://github.com/SRDdev/PaliGemma) |
| hkproj implementation (YouTube walkthrough) | [github.com/hkproj/pytorch-paligemma](https://github.com/hkproj/pytorch-paligemma) |
| stojk notebook implementation | [github.com/stojk/paligemma-from-scratch](https://github.com/stojk/paligemma-from-scratch) |
| Attention is All You Need | [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762) |
| RoPE Paper | [arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864) |
| SigLIP Paper | [arxiv.org/abs/2303.15343](https://arxiv.org/abs/2303.15343) |

---

