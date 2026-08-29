# Chapter 4 (Inference) — 완료 기록

> **Chapter 4 전체 완료.** 사전학습 3B 가중치로 실제 캡션 생성까지 확인.
> 최초 작성 2026-08-26 · 갱신 2026-08-27 · **완료 2026-08-30**

---

## 현재 상태 — 4-3 완료

`google/paligemma-3b-mix-224` 가중치로 실제 이미지 캡셔닝이 동작한다.

```
$ uv run python -m paligemma.multimodal.loader        # 고양이 사진

'What is in the image?'   ->  cat
'caption en'              ->  In this image we can see a cat.
'Describe the image.'     ->  In this image we can see a cat.
```

| 항목 | 상태 |
|---|---|
| 4-1 KVCache | ✅ |
| 4-2 `generate()` | ✅ |
| prefix-LM mask | ✅ `build_prefix_lm_mask` + 단위 테스트 |
| RoPE 규약 | ✅ HF `rotate_half`, 수치 일치 확인 |
| SigLIP / Gemma config | ✅ 실제 3B 값 |
| BICUBIC 리사이즈 | ✅ |
| 테스트 OOM | ✅ |
| **4-3 가중치 로딩** | ✅ **완료** |

> 문장이 나온다는 것은 **prefix-LM 마스크·RoPE 규약·키 매핑·이미지 전처리·bf16 캐스팅이 전부 맞았다는 뜻**이다. 하나라도 틀리면 이런 문장은 나오지 않는다.

---

## 4-3 이전에 정리한 것

### ① prefix-LM attention mask (신규 구현)

PaliGemma의 핵심 특징 — **이미지 토큰과 프롬프트 텍스트 전체가 하나의 prefix이고 그 안에서는 양방향**, 생성 구간만 causal.

```
         img[0..255]  prompt   생성분
img     [   전부 봄   |  전부 봄 |  ✗  ]   ← prefix 내부는 양방향
prompt  [   전부 봄   |  전부 봄 |  ✗  ]
gen[0]  [   전부 봄   |  전부 봄 |  ✗  ]   ← 여기서부터 causal
gen[1]  [   전부 봄   |  전부 봄 | gen[0] ]
```

**구현 위치** — `paligemma/multimodal/model.py`의 모듈 레벨 함수. `PaliGemmaForConditionalGeneration.forward`가 호출한다. 이미지 토큰 위치를 아는 유일한 계층이기 때문.

**구현 방식** — 마스크 행렬을 직접 그리지 않고 쿼리/키 위치 벡터의 브로드캐스팅 비교로 만든다:

```python
q_pos = cache_len + torch.arange(seq_len, device=device)   # (S,)
k_pos = torch.arange(cache_len + seq_len, device=device)   # (C+S,)

allowed = (k_pos[None, :] < prefix_len) | (k_pos[None, :] <= q_pos[:, None])
#          └─ prefix 는 양방향 ─┘         └─ 그 외는 causal ─┘
```

이 한 식이 세 경우를 모두 처리한다:

| 상황 | `prefix_len` | 결과 |
|---|---|---|
| 추론 prefill (260 토큰, 캐시 0) | 260 | `k_pos < 260`이 항상 참 → **전부 0** |
| 추론 decode (1 토큰, 캐시 N) | 무엇이든 | `k_pos <= q_pos`가 항상 참 → **전부 0** |
| 학습 (prefix + suffix) | 프롬프트 길이 | 우상단 블록만 `-inf` |

> **추론 경로에서 마스크는 항상 전부 0이다.** 생성 토큰의 causality는 KV 캐시가 이미 보장하기 때문이다. 이전 구현이 전체 causal이었다는 것은 이미지 패치가 좌→우로만 읽히고 있었다는 뜻이다.

**`cache_len` vs `prefix_len`** — 헷갈리기 쉬운데 서로 다른 축을 잰다.

- `cache_len` = **시간축**: "이미 처리해서 캐시에 넣었는가"
- `prefix_len` = **위치축**: "절대 위치 0부터 몇 번째까지가 양방향인가"

prefill 때는 prefix가 전부 *새 토큰* 안에 있고, decode 때는 prefix가 전부 *캐시* 안에 있다. 같은 `prefix_len` 값이 두 상황에서 다른 곳을 가리킬 뿐이다.

**`prefix_len`의 소유자** — 모델이 아니라 `generate()`. `KVCache`를 caller가 소유하게 만든 것과 같은 원칙이다.

```python
prefix_len = encoded["input_ids"].shape[-1]   # prefill 직전 한 번
```

`input_ids`에는 이미 `<image>` 토큰 256개가 들어 있으므로(프로세서가 프롬프트 문자열 단계에서 삽입) 이 길이로 충분하다. **이미지는 시퀀스를 늘리지 않는다** — 뚫려 있는 자리의 임베딩을 갈아끼울 뿐이다.

나중에 파인튜닝 단계에서는 프로세서가 `token_type_ids`(0 = prefix, 1 = suffix)를 내보내게 하고 거기서 유도한다. `forward`의 인터페이스는 그대로라 바꿀 게 없다.

### ② RoPE 규약 전환 — 4-3 최대 걸림돌 제거

이전 구현은 `view_as_complex`로 **인접 쌍** `(x₀,x₁), (x₂,x₃), …`을 회전시켰다(LLaMA 원논문 방식). HF Gemma는 `rotate_half` 방식으로 **앞뒤 절반** `(x₀, x₁₂₈), (x₁, x₁₂₉), …`을 묶는다. 둘 다 올바른 RoPE지만 가중치는 호환되지 않는다.

현재 `apply_rotary_emb`는 `chunk(2, dim=-1)` 기반으로 바뀌었다:

```python
x1, x2 = x.chunk(2, dim=-1)
return torch.cat((x1 * cos - x2 * sin,
                  x2 * cos + x1 * sin), dim=-1)
```

HF의 `x * cos_full + rotate_half(x) * sin_full`과 대수적으로 동일하다. 실제로 확인:

```
RoPE HF-compatible : True | maxdiff 2.4e-07
```

`torch.polar`로 `freqs_cis`를 만들고 `.real`/`.imag`를 cos/sin으로 쓰는 방식은 유지되었으므로 코드는 여전히 간결하다.

### ③ 테스트 OOM 해결

`test/conftest.py`에 세션 스코프 fixture 3개(`gemmamodel`, `siglipmodel`, `model_and_processor`)를 통합하고, `_release()`로 `torch.cuda.empty_cache()`를 호출하도록 정리. 배치 크기도 축소.

### ④ SigLIP config → 실제 3B 값

`hidden_size=1152`, `intermediate_size=4304`, `num_hidden_layer=27`, `num_attention_heads=16`, **`patch_size=14`**.

`num_image_tokens`는 대부분의 호출부에서 `(image_size // patch_size) ** 2`로 유도하도록 바뀌어 자동으로 256이 된다.

### ⑤ GELU / BICUBIC

- `GemmaMLP`, `SiglipMLP` 모두 `approximate="tanh"` — 원본 config의 `gelu_pytorch_tanh`와 일치 ✅
- `transforms.Resize(..., interpolation=InterpolationMode.BICUBIC)` — 원본 `resample=3`과 일치 ✅

---

## 가중치 현황

| 리포 | 캐시 상태 |
|---|---|
| `google/paligemma-3b-**mix**-224` | **가중치 있음** — safetensors 3개, **603 텐서** |
| `google/paligemma-3b-pt-224` | 17 MB — config/토크나이저만 |

**`mix`를 사용한다.** instruction-tuned라 `"What is in the image?"` 같은 자연어 질문에 바로 답한다. `pt`는 `"caption en"` 같은 태스크 프리픽스가 필요하다. 토크나이저·아키텍처는 동일하다.

> ⚠️ 체크포인트는 **fp32 11.1 GB**(shard 3개). 모델로 올리면 **10.89 GiB**인데 GPU 가용량이 11.42 GiB라 파라미터만으로 95%를 먹는다. `bfloat16` 캐스팅이 필수다(**5.45 GiB**). 방법은 4-3 체크리스트 ① 참조 — `model.to(dtype)`은 쓰면 안 된다.

### 키 대조 결과

정규식 매핑을 적용해 체크포인트와 대조한 결과 **완전히 일치**한다.

```
매핑된 키   : 603 / 모델 키 604
missing     : 1   ['language_model.lm_head.weight']   ← weight tying, 정상
unexpected  : 0
shape 불일치: 0
```

매핑 규칙과 검증 방법은 아래 4-3 체크리스트 ③ 참조.

---

## 4-3 가중치 로딩 — 구현 기록

> `config.json`의 값과 `GemmaConfig`/`SiglipVisionConfig`가 vision·text 양쪽 모두 **전부 일치**함을 확인했다.
> 아래 ①~⑤가 로딩 자체, ⑥~⑧이 로딩 후 실제로 걸린 함정이다.

### ① ⚠️ bfloat16 캐스팅의 함정 — 로딩보다 먼저 해결

3B를 fp32로 올리면 GPU에 안 들어간다.

```
params   : 2.92 B
fp32     : 10.89 GiB      ← GPU free 11.42 GiB. 파라미터만으로 95%
bfloat16 :  5.45 GiB
```

그런데 **`model.to(torch.bfloat16)`을 쓰면 안 된다.** `nn.Module.to`는 부동소수점뿐 아니라 **복소수 텐서도** 캐스팅 대상에 넣기 때문에, `GemmaAttention.freqs_cis`(complex64)가 bfloat16으로 변환되며 허수부가 날아간다.

```
A) model.to(bfloat16)
   freqs_cis : torch.bfloat16
   forward   : RuntimeError  imag is not implemented for tensors with non-complex dtypes

B) parameters() 만 캐스팅
   freqs_cis : torch.complex64          ← 버퍼 보존 성공
   forward   : RuntimeError  expected m1 and m2 to have the same dtype: float != BFloat16
```

**A**는 `.imag`가 실수 텐서에서 예외를 던져 조용히 틀리진 않지만, 이 경로는 못 쓴다.
**B**는 버퍼는 지켰는데 `apply_rotary_emb`에서 fp32 `cos` × bf16 `x`가 **fp32로 승격**되고, 그게 bf16 가중치인 `out_proj`에 들어가면서 터진다.

**수정 두 가지:**

```python
# ① apply_rotary_emb — cos/sin 을 x 의 dtype 으로
cos = freqs_cis.real[None, None, :, :].to(x.dtype)
sin = freqs_cis.imag[None, None, :, :].to(x.dtype)

# ② 캐스팅은 파라미터만 (parameters() 는 버퍼를 포함하지 않는다)
for p in model.parameters():
    p.data = p.data.to(torch.bfloat16)
model.to(DEVICE)
```

둘 다 적용 후 확인:
```
freqs_cis : torch.complex64
output    : torch.bfloat16  torch.Size([1, 8, 1000])  | finite: True
```

> `model.to(device=..., dtype=...)` 한 줄로 처리하고 싶다면 `freqs_cis`를 complex 대신 **실수 `cos`/`sin` 버퍼 두 개**로 바꾸면 된다. HF Gemma가 그 방식이다. 지금 `torch.polar` 기반 코드가 간결하므로 위 2줄 수정이 더 작다.

### ② 테스트용 작은 config 분리 — config 교체와 세트

`GemmaConfig`가 3B가 된 순간 `conftest.py`의 세션 스코프 fixture 3개가 전부 3B를 만든다. bf16으로 줄여도 동시 상주하면 터진다. 그리고 **랜덤 가중치 3B는 테스트 가치가 0이다** — 지금 테스트가 검증하는 것은 배선(shape, 캐시 일관성, 마스크, 생성 루프)이지 표현력이 아니다.

```python
def _tiny_gemma():
    return GemmaConfig(
        vocab_size=257216,          # ← 줄이면 안 됨 (img_token_id = 257152)
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
    )

def _tiny_vision():
    return SiglipVisionConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layer=2,
        num_attention_heads=4,
        patch_size=14,              # ← 유지 (이미지 토큰 256개 전제 테스트가 있음)
    )
```

`vocab_size`만 실제 값을 유지해야 한다. 프로세서가 진짜 토크나이저로 `<image>` = 257152를 만들기 때문. 임베딩이 257216 × 128 = 33M로 전체의 대부분이지만 fp32로도 133 MB라 문제없다.

실제 3B 가중치 로딩 검증은 `@pytest.mark.slow`를 붙여 기본 실행에서 빠지게 하고 필요할 때만 `-m slow`로 돌린다.

### ③ state_dict 키 매핑 — 검증 완료

순서 있는 정규식 치환. **실제 체크포인트와 대조 완료.**

```python
RULES = [
    # --- vision : HF 는 vision_model. 계층이 하나 더 있음 ---
    (r'^vision_tower\.vision_model\.embeddings\.position_embedding', 'vision_tower.embedding.pos_embedding'),
    (r'^vision_tower\.vision_model\.embeddings\.',                   'vision_tower.embedding.'),
    (r'^vision_tower\.vision_model\.post_layernorm',                 'vision_tower.layernorm'),
    (r'^vision_tower\.vision_model\.',                               'vision_tower.'),
    (r'\.self_attn\.',                                               '.attention.'),
    (r'\.layer_norm(\d)\.',                                          r'.layernorm\1.'),
    # --- gemma ---
    (r'\.input_layernorm\.',                                         '.rms_norm1.'),
    (r'\.post_attention_layernorm\.',                                '.rms_norm2.'),
    (r'\.attention\.o_proj\.',                                       '.attention.out_proj.'),
    (r'^language_model\.model\.norm\.',                              'language_model.model.rms_norm.'),
    # --- projector ---
    (r'^multi_modal_projector\.linear\.',                            'multi_modal_projector.fc.'),
]
```

**순서 의존성 세 군데:**

- `position_embedding` → `pos_embedding` 규칙이 `embeddings.` → `embedding.` 규칙보다 **먼저** 와야 한다. 뒤집으면 접두사가 먼저 잘려 매칭이 안 된다.
- `post_layernorm`을 `vision_model.` 제거보다 먼저 처리해야 한다.
- `.self_attn.` → `.attention.`을 먼저 돌린 뒤에야 `.attention.o_proj.` → `.out_proj.`가 걸린다. vision 쪽은 이미 `out_proj`라 이 규칙에 안 걸린다 — 그래서 `o_proj`만 콕 집어야 한다.

**검증 결과** (safetensors 헤더에서 shape만 읽어 데이터 로드 없이 대조):

```
매핑된 키   : 603 / 모델 키 604
missing     : 1   ['language_model.lm_head.weight']
unexpected  : 0
shape 불일치: 0
```

missing 1개는 weight tying 때문에 체크포인트에 없는 것이므로 정상이다.

> 가중치를 올리기 전에 이 대조를 먼저 돌리면 매핑 오류를 데이터 로드 없이 잡을 수 있다. `safe_open(...).get_slice(k).get_shape()`는 헤더만 읽는다.

### ④ 로더 구현

`paligemma/multimodal/loader.py`에 `load_hf_model(model_path, device, dtype)`.

```
1. 모델을 CPU 에 생성                             10.9 GiB fp32
2. 파라미터를 bf16 으로 캐스팅                     →  5.5 GiB     ← 먼저 줄인다
3. safe_open 으로 shard 3개 순회
     키 매핑 + .to(dtype) 하며 state dict 구성      +5.5 GiB
4. load_state_dict(state, strict=False)
5. 반환값 검증
6. model.to(device).eval()                        GPU 5.45 GiB
```

**2번을 3번보다 먼저** 한다. 뒤집으면 fp32 모델(10.9) + bf16 state dict(5.5)가 동시에 CPU RAM에 상주한다. RAM 58 GB라 어차피 되지만 습관을 들여두는 편이 좋다.

```python
with safe_open(shard, framework="pt", device="cpu") as f:
    for k in f.keys():
        state[map_key(k)] = f.get_tensor(k).to(dtype)
```

> 더 아끼려면 shard마다 `load_state_dict(부분_dict, strict=False)`를 호출하고 dict를 버리는 방식도 된다. CPU 피크가 8 GB 정도로 내려간다. 다만 `missing_keys`가 매번 나오므로 검증은 마지막에 따로 해야 한다.

**검증** — `strict=True`는 tying 때문에 무조건 실패하므로 `strict=False` 후 반환값을 직접 본다.

```python
missing, unexpected = model.load_state_dict(state, strict=False)

assert unexpected == [], f"매핑 안 된 키: {unexpected[:5]}"
assert missing == ["language_model.lm_head.weight"], f"빠진 키: {missing[:5]}"
```

`unexpected == []`가 오타를 잡아주는 실질적인 안전장치다. `strict=False`를 그냥 쓰고 넘어가면 규칙 하나가 틀려도 그 레이어만 랜덤 가중치인 채 조용히 돌아간다.

**`lm_head`는 따로 처리할 필요 없다.** `gemma/model.py:58`에서 `self.lm_head.weight = self.model.embed_tokens.weight`로 같은 `Parameter` 객체를 공유하므로, `embed_tokens`에 값이 복사되면 `lm_head`도 함께 갱신된다. bf16 캐스팅 때도 `parameters()`가 공유 파라미터를 중복 없이 순회하므로 tie가 유지된다.

### ⑤ 자잘한 참고

- 체크포인트는 **fp32 11.1 GB** 3개 shard. bf16으로 캐스팅하며 읽는 것이 필수다.
- 토크나이저도 mix-224 스냅샷에 같이 있다. 로더에서 `model_path` 하나로 가중치·토크나이저를 함께 처리하면 경로가 한 곳으로 모인다.

### ⑥ 🔴 이미지 임베딩 스케일 — 로딩 후 실제로 걸린 함정

가중치를 다 올리고 실행했는데 **이미지를 전혀 못 보는** 출력이 나왔다. 캡션이 `\n`만 나오거나 `<eos>`가 즉시 튀어나오는 증상.

```
baseline (수정 전)          top5 = '<eos>', 'empty', 'blank', ' empty', 'page'
+ 이미지 스케일 보정         top5 = 'cat',   'kitten', 'Cat',  'orange', 'k'
```

"empty / blank / page" — **빈 입력으로 취급**하고 있었다.

**원인** — `GemmaModel.forward`가 `x = x * math.sqrt(hidden_size)`로 임베딩 **전체**에 45.25를 곱한다. 텍스트 임베딩에는 이게 Gemma 원래 설계지만, 이미지 feature는 프로젝터를 통과한 최종값이라 곱하면 안 된다. 그런데 지금 구조는 이미지를 먼저 끼워 넣고 나중에 통째로 곱하므로 **이미지만 45배 뻥튀기**된다.

HF는 구조로 이 문제를 피한다 — 스케일링을 `GemmaScaledWordEmbedding` **안에** 두어 임베딩 조회에만 적용하고(`embed_scale=hidden_size**0.5`), 이미지 feature는 그 **뒤에** `masked_scatter`로 끼워 넣는다. 즉 이미지는 스케일링 경로를 아예 타지 않는다.

**수정** — 프로젝터 출력을 미리 나눠서 상쇄시킨다. `PaliGemmaForConditionalGeneration.forward`:

```python
image_embedding = self.multi_modal_projector(image_embedding) / math.sqrt(self.config.text_config.hidden_size)
```

> 이 버그의 무서운 점은 **크래시도 shape 오류도 없다**는 것이다. RMSNorm이 pre-norm이라 각 레이어 입력은 정규화되어 정상으로 보이고, 잔차 스트림의 비율만 틀어진다. 랜덤 가중치에서는 전혀 티가 안 나고, 실제 가중치를 올려야만 "캡션이 이상하다"로 드러난다.

### ⑦ `pixel_values` dtype

프로세서의 `ToTensor()`는 fp32를 만드는데 모델은 bf16이다. SigLIP 첫 Conv2d에서 터진다.

```
RuntimeError: Input type (torch.FloatTensor) and weight type (CPUBFloat16Type) should be the same
```

`generate()`에서 모델 dtype을 따라가게 한다:

```python
device = next(model.parameters()).device
dtype  = next(model.parameters()).dtype
encoded["input_ids"]    = encoded["input_ids"].to(device)          # int64 유지
encoded["pixel_values"] = encoded["pixel_values"].to(dtype).to(device)
```

`torch.bfloat16` 하드코딩보다 모델에서 꺼내는 편이 낫다 — fp32 작은 모델을 쓰는 테스트에서도 그대로 동작한다.

| 대상 | dtype | 조치 |
|---|---|---|
| tokenizer | — | 없음. 정수 id만 만듦 |
| `input_ids` | int64 | 그대로 |
| **`pixel_values`** | fp32 → bf16 | **변환 필요** |
| `attention_mask` | 자동 | `text_embedding.dtype`을 따라감 |
| 로짓 / 샘플링 | bf16 | 그대로 동작 (`multinomial`이 bf16 지원, cumsum 정밀도도 실용상 문제없음) |

### ⑧ RULES 정규식 — 매칭이 없어도 조용하다

`re.sub`는 매치가 없으면 문자열을 **그대로 돌려준다.** 에러가 안 난다. 실제로 겪은 실수들:

| 실수 | 증상 |
|---|---|
| `(\d\.` — 괄호 안 닫힘 | `re.PatternError` (즉시 터짐, 그나마 나음) |
| 치환 문자열 누락 `(r'...')` | 튜플이 아니라 문자열 → `ValueError: too many values to unpack` |
| `^\.self_attn\.` — 불필요한 `^` | **절대 매치 안 됨.** `.self_attn.`은 문자열 중간에 나온다 |
| `vison_tower` 오타 | 조용히 무시 |
| 우리 쪽 이름을 `position_embedding`으로 착각 | 실제로는 `pos_embedding` |
| 패턴엔 `\.` 없는데 치환엔 `.` 있음 | `out_proj..weight` 생성 |

`^` 앵커 실수는 연쇄로 번진다 — `self_attn` → `attention` 변환이 죽으면 `.attention.o_proj` 규칙도 같이 죽는다.

**그래서 `unexpected_keys` 검증이 필수다.** 규칙 하나가 죽어도 `load_state_dict(strict=False)`는 그냥 넘어가고, 그 레이어만 랜덤 가중치인 채로 돌아간다. 가중치를 올리기 전에 키 집합만 대조하면 11 GB 로드 없이 몇 초 만에 잡힌다.

---

## 남은 것 (선택)

### 1-indexed 위치

HF는 RoPE 위치를 **1부터** 쓴다 — `modeling_paligemma.py:236`:

```python
position_ids = position_ids.unsqueeze(0) + 1  # Paligemma positions are 1-indexed
```

현재 구현은 `freqs_cis[cache_len : cache_len + num_tokens]`로 0부터 쓴다. 실측해보니 **출력 차이는 거의 없었다** (top5 순서만 미세하게 바뀜):

```
baseline        top5 = '<eos>', 'empty', 'blank', 'page',   ' empty'
+ 1-indexed     top5 = '<eos>', 'empty', 'blank', ' empty', 'page'
```

원본 파리티를 맞추려면 슬라이스를 한 칸 밀면 된다. `max_position_embeddings=8192`라 버퍼 여유는 충분하다.

```python
freq_cis = self.freqs_cis[cache_len + 1 : cache_len + 1 + num_tokens]
```

### `do_sample` 기본값

캡셔닝은 greedy가 안정적이다. 위 출력은 전부 `do_sample=False`로 얻은 것. 기본값을 바꾸거나 최소한 예제는 greedy로 두는 편이 낫다.

### `freqs_cis` 중복

레이어마다 8192×128 complex64 = 8.4 MB를 똑같이 들고 있어 18개면 **151 MB**. `GemmaModel`에서 한 번 만들어 공유하도록 옮기면 깔끔하다.

---

## 부록 A: 마스크 테스트 설계

`build_prefix_lm_mask`는 순수 함수라 **모델도 GPU도 없이** 검증된다. 마스크는 눈으로 봐야 확신이 서는 코드이므로 작은 크기로 찍어보는 것이 핵심.

| 테스트 | 검증 내용 |
|---|---|
| prefill 전부 0 | **핵심 회귀 테스트** — 이미지 패치가 causal로 읽히던 버그 |
| `prefix_len=3, seq_len=5` 패턴 일치 | 로직 정확성 |
| `prefix_len=0` ≡ `GemmaModel` causal | **두 곳의 마스크 규약이 일치하는가** |
| decode 스텝 전부 0 | shape `(1,1,1,C+1)` |
| 캐시 열은 항상 열림 | `prefix_len` 무관 |
| prefix가 캐시 경계를 넘을 때 | 양방향이 실제로 작동하는 증거 |
| dtype 보존 (bf16/fp16) | 4-3에서 터지는 것 미리 차단 |

`torch.where(allowed, 0.0, -inf)` 대신 `zeros(dtype=...)` + `masked_fill_`을 쓰면 dtype 관리가 깔끔하다. `torch.arange`에 `device=`를 빼먹으면 CPU 텐서가 만들어져 비교 연산에서 에러가 난다.

단위 테스트만으로는 **`forward`가 그 함수를 실제로 부르는지**를 검증할 수 없다. 배선이 끊겨도 전부 통과한다. `model.language_model.forward`를 spy로 감싸 prefill 마스크가 전부 0인지 확인하는 통합 테스트를 하나 두면 좋다.

---

## 부록 B: 이미 해결된 항목

재발 방지용 기록.

| 항목 | 내용 |
|---|---|
| KV cache pre-RoPE 저장 | Critical 버그. RoPE 적용 후 캐시에 저장하도록 수정 |
| `num_items` 레이어 내부 읽기 | 레이어 0이 먼저 갱신해 레이어 1부터 RoPE 위치가 한 칸 밀림. `cache_len`을 `GemmaModel`에서 한 번 계산해 전달 |
| 프롬프트 포맷 | `[BOS][img][text]` → `[img×N][BOS][text][\n]`. 원본 `build_string_from_input`과 일치 확인 |
| `max_length` / `truncation` | 원본에 없는 파라미터. 제거 |
| `if pixel_values:` | 다중 원소 텐서는 `bool()`이 `RuntimeError`. `is not None`으로 수정 |
| `tokenizer.eos` | 존재하지 않는 속성. `eos_token_id`(=1) |
| `argmax` shape | `keepdim=True` 없으면 `(B,)`가 되어 greedy 경로에서 터짐 |
| `top_p` 하드코딩 | `cumulated_probs > 0.9`가 파라미터를 무시하고 있었음 |
| device 누락 | `encoded` 텐서가 GPU로 안 올라가고 있었음. `next(model.parameters()).device` 한 번으로 통일 |
| 빈 `generated` | 첫 토큰이 EOS면 `torch.cat([])`가 `ValueError`. 조기 반환 가드 추가 |
| 테스트 OOM | fixture 중복 정의 + 과도한 배치 크기 |
| RoPE 규약 | interleaved → rotate_half 전환, HF와 수치 일치 확인 |
| BILINEAR 리사이즈 | BICUBIC으로 수정 |
| SigLIP config | paligemma-3b 실제 값으로 교체 (patch 14) |
| `from sys import prefix` | 자동완성 사고. 제거됨 |
| `GemmaConfig` 토이 값 | 실제 3B 값으로 교체 |
| `model.to(bfloat16)` | complex64 버퍼(`freqs_cis`)를 파괴. `parameters()`만 캐스팅해야 함 |
| RoPE cos/sin dtype | fp32 cos × bf16 x → fp32 승격 → bf16 Linear 에서 dtype mismatch |
| **이미지 임베딩 스케일** | **`sqrt(hidden)`이 이미지에도 곱해져 45배 뻥튀기. 실제 가중치에서만 드러남** |
| `pixel_values` dtype | `ToTensor()` fp32 vs 모델 bf16 → Conv2d 에서 터짐 |
| RULES `^` 앵커 | `^\.self_attn\.`은 절대 매치 안 됨. 매칭 실패는 조용하다 |
| `generate` 반환 타입 | 2D 텐서를 `decode`에 넘기면 `list`. 수정됨 |

### 참고: 랜덤 가중치에서 `\n`만 생성되는 이유

프롬프트 마지막 토큰이 `\n`(id 108)인데, 학습되지 않은 모델은 residual stream + weight tying 때문에 **방금 본 토큰을 그대로 예측**한다. 가중치를 올린 뒤에도 `\n`만 나온다면 그건 다른 문제다 — 위 ⑥번을 볼 것. 배선이 정상이라는 신호이기도 하다. 이 때문에 "토큰 1개 = 문자 1개"가 되어, 문자 길이로 토큰 수를 세는 테스트가 우연히 통과할 수 있으니 주의.
