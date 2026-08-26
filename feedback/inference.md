# Chapter 4 (Inference) 진행 상황 & 4-3 준비 체크리스트

> 4-1 KVCache 통일, 4-2 `generate()` 구현 완료 시점의 정리.
> 4-3(사전학습 가중치 로딩)에 들어가기 전에 확인·수정해야 할 항목을 우선순위별로 기록한다.
> 조사 기준일: 2026-08-26

---

## 현재 상태

| 항목 | 상태 | 비고 |
|---|---|---|
| 4-1 KVCache | ✅ 완료 | 클래스 + 전 계층 배선 + 정확성 검증 |
| 4-2 `generate()` | ✅ 완료 | prefill/decode 분리, top-p 샘플링, EOS 정지 |
| 4-3 가중치 로딩 | ❌ 미착수 | 아래 체크리스트 참조 |
| prefix-LM mask | ❌ 미구현 | 체크리스트에 없는 숨은 항목 |

### 4-1에서 잡은 핵심 버그

**`num_items`를 레이어 내부에서 읽으면 안 된다.** 공유 `KVCache` 객체로 바꾸면 `num_items`는 항상 레이어 0번의 길이를 본다. 레이어 0이 먼저 갱신해버리므로 레이어 1부터 RoPE 위치가 한 칸씩 밀린다. 크래시 없이 값만 미세하게 틀어지는 종류의 버그다.

→ `GemmaModel.forward`에서 루프 시작 **전에** `cache_len`을 한 번 계산해 각 레이어로 전달하도록 수정. `test_kv_cache_consistency`("캐시 있는 경로 == 캐시 없는 경로")가 이를 지켜준다.

---

## 🔴 즉시 수정: 전체 테스트가 OOM으로 실패

```
test/siglip 단독 실행  →  7 passed
test/ 전체 실행        →  1 failed, 22 passed, 3 errors  (torch.OutOfMemoryError)
```

**원인** — `test_generate.py`가 `test_paligemma_model.py`와 동일한 `scope="session"` 모델 fixture를 중복 정의하고 있다. PaliGemma 모델 2개 + Gemma 모델 1개가 세션 내내 GPU에 상주해서, 나중에 도는 siglip 테스트가 자리를 못 잡는다.

**해결** — `test/conftest.py`로 fixture를 하나만 두고 두 파일이 공유하게 한다. 3B 모델을 올리면 메모리가 훨씬 빡빡해지므로 4-3 전에 반드시 정리한다.

---

## 가중치 현황

| 리포 | 캐시 상태 |
|---|---|
| `google/paligemma-3b-**mix**-224` | **가중치 있음** — safetensors 3개, 603 텐서, **11.69 GB (fp32)** |
| `google/paligemma-3b-pt-224` | 17 MB — config/토크나이저만 |

**`mix`를 사용한다.** instruction-tuned라 `"What is in the image?"` 같은 자연어 질문에 바로 답한다. `pt`는 사전학습 체크포인트라 `"caption en"` 같은 태스크 프리픽스가 필요하다. 토크나이저·아키텍처는 동일하므로 현재 프로세서를 그대로 쓸 수 있다.

> ⚠️ **fp32 11.69 GB는 12 GB GPU(RTX 4070)에 들어가지 않는다.** 로드 시 `bfloat16`으로 캐스팅해야 한다(~5.9 GB). 활성값까지 감안하면 이것이 유일한 선택지.

---

## 4-3 준비 체크리스트

### ① Config 값 교체

| vision | 현재 → 원본 | text | 현재 → 원본 |
|---|---|---|---|
| `hidden_size` | 768 → **1152** | `vocab_size` | 300000 → **257216** |
| `intermediate_size` | 3072 → **4304** | `hidden_size` | 1024 → **2048** |
| `num_hidden_layer` | 12 → **27** | `intermediate_size` | 4096 → **16384** |
| `num_attention_heads` | 12 → **16** | `num_hidden_layers` | 8 → **18** |
| **`patch_size`** | 16 → **14** | `num_attention_heads` | 4 → **8** |
| `image_size` | 224 (동일) | `num_key_value_heads` | 1 (동일) |
| | | `max_position_embeddings` | 2048 → **8192** |

- `head_dim`은 원본이 256이고 `2048 / 8 = 256`이라 현재의 `emb_dim // num_heads` 계산과 일치한다.
- `rope_theta`는 config에 없어 HF 기본값 10000.0 — 현재 값과 같다.

### ② `num_image_tokens` 196 → 256

`patch_size`가 14가 되면 `(224/14)² = 256`이다. 현재 **5곳에 하드코딩**되어 있다:

- `paligemma/multimodal/inference.py:82`
- `paligemma/multimodal/input_processer.py:59`
- `paligemma/multimodal/model.py:58`
- `test/multimodal/test_paligemma_model.py:17`
- `test/multimodal/test_generate.py:20`

→ `(image_size // patch_size) ** 2`로 vision config에서 유도하도록 변경한다.

### ③ state_dict 키 매핑

체크포인트 키와 현재 구현의 키를 대조한 결과.

| 우리 | HF 체크포인트 |
|---|---|
| `vision_tower.embedding.patch_embedding` | `vision_tower.`**`vision_model`**`.embeddings.patch_embedding` |
| `vision_tower.embedding.`**`pos`**`_embedding` | `...embeddings.`**`position`**`_embedding` |
| `vision_tower.encoder.layers.N.`**`attention`** | `...encoder.layers.N.`**`self_attn`** |
| `vision_tower.encoder.layers.N.`**`layernorm`**`1/2` | `...layers.N.`**`layer_norm`**`1/2` |
| `vision_tower.`**`layernorm`** | `vision_tower.vision_model.`**`post_layernorm`** |
| `language_model.model.layers.N.`**`attention`**`.out_proj` | `...`**`self_attn`**`.`**`o_proj`** |
| `language_model.model.layers.N.`**`rms_norm1`** | `...`**`input_layernorm`** |
| `language_model.model.layers.N.`**`rms_norm2`** | `...`**`post_attention_layernorm`** |
| `language_model.model.`**`rms_norm`** | `language_model.model.`**`norm`** |
| `multi_modal_projector.`**`fc`** | `multi_modal_projector.`**`linear`** |
| `language_model.lm_head.weight` | **체크포인트에 없음** (weight tying) |

**이미 일치하는 것** — `q/k/v_proj`, `mlp.{gate,up,down}_proj`, `mlp.fc1/fc2`, vision의 `out_proj`.

**`freqs_cis`** — `persistent=False`로 등록되어 있어 state_dict에 들어가지 않는다. unexpected key 문제 없음. ✅

**`SiglipVisionModel`이 `SiglipVisionTransformer`를 상속**하는 구조 때문에 `vision_model.` 계층이 한 단계 빠져 있다. 매핑에서 끼워 넣으면 된다.

### ④ ⚠️ RoPE 규약 불일치 — 가장 위험

`apply_rotary_emb`가 `view_as_complex`로 **인접 쌍** `(x₀,x₁), (x₂,x₃), …`을 회전시킨다(LLaMA 원논문 방식). 반면 **HF Gemma는 `rotate_half`** 방식으로 앞뒤 절반 `(x₀, x₁₂₈), (x₁, x₁₂₉), …`을 묶는다.

수학적으로 둘 다 올바른 RoPE지만 **가중치는 호환되지 않는다.** 어떤 채널이 어떤 회전 주파수를 받는지가 달라지기 때문이다. 랜덤 초기화에서는 차이가 드러나지 않으므로 지금까지 문제가 보이지 않았다. 사전학습 q/k 가중치를 그대로 올리면 **크래시도 shape 오류도 없이 조용히 틀린 출력**이 나온다.

**해결** — `apply_rotary_emb`를 rotate_half 방식으로 바꾸거나(권장), 로딩 시 `q_proj`/`k_proj` 가중치 행을 permute한다.

### ⑤ GELU 근사식

원본 config는 vision·text 양쪽 모두 **`gelu_pytorch_tanh`** 를 쓴다. 현재 `GemmaMLP`는 `nn.functional.gelu`(exact erf)를 쓰고 있다.

→ `nn.GELU(approximate='tanh')` 또는 `F.gelu(x, approximate='tanh')`로 변경. SigLIP MLP도 함께 확인한다.

### ⑥ prefix-LM attention mask — 미구현

현재 `GemmaModel`이 전체 시퀀스에 causal mask를 일괄 적용한다. PaliGemma의 핵심 특징은 **이미지 토큰과 프롬프트 텍스트 전체가 하나의 prefix이고, 그 안에서는 양방향**이라는 점이다. 생성 구간만 causal이다.

```
         img[0..255]  prompt   생성분
img     [   전부 봄   |  전부 봄 |  ✗  ]   ← prefix 내부는 양방향
prompt  [   전부 봄   |  전부 봄 |  ✗  ]
gen[0]  [   전부 봄   |  전부 봄 |  ✗  ]   ← 여기서부터 causal
gen[1]  [   전부 봄   |  전부 봄 | gen[0] ]
```

랜덤 가중치에서는 티가 안 났지만, **실제 가중치에서는 이미지 토큰이 서로를 못 봐서 품질이 크게 떨어진다.** "가중치는 맞는데 캡션이 이상하다"의 유력한 원인.

마스크는 이미지 토큰 위치를 아는 `PaliGemmaForConditionalGeneration.forward`에서 만들어 `GemmaForCausalLM`으로 전달해야 한다.

### ⑦ 리사이즈 보간법

원본 image processor는 `resample=3`, 즉 **BICUBIC**이다. `transforms.Resize`의 기본값은 BILINEAR.

→ `interpolation=transforms.InterpolationMode.BICUBIC` 명시.

---

## 권장 작업 순서

1. **`conftest.py`로 fixture 통합** → 전체 테스트 초록불 복구
2. **커밋** (KV cache 통일 + generate + 테스트)
3. **④ RoPE, ⑤ GELU, ⑦ BICUBIC** — 가중치 없이 지금 고칠 수 있는 것들. 기존 테스트가 지켜준다
4. **⑥ prefix-LM mask** + 테스트
5. **① config + ② `num_image_tokens` 유도화**
6. **③ 키 매핑 + `load_state_dict`** (bfloat16 캐스팅)
7. 실제 이미지로 캡션 생성

> **3~5번을 6번보다 먼저 하는 것이 중요하다.** 가중치를 올린 뒤에 이것들을 고치면 출력이 이상할 때 "가중치 로딩이 틀렸나 / RoPE가 틀렸나 / mask가 틀렸나"가 뒤엉켜 원인 분리가 매우 어려워진다.

---

## 부록: 이미 해결된 항목

작업 과정에서 발견하고 수정한 것들. 재발 방지용 기록.

| 항목 | 내용 |
|---|---|
| KV cache pre-RoPE 저장 | 이전에 지적된 Critical 버그. 현재는 RoPE 적용 후 캐시에 저장 — 해결됨 |
| `num_items` 레이어 내부 읽기 | `cache_len`을 `GemmaModel`에서 한 번 계산해 전달하도록 수정 |
| 프롬프트 포맷 | `[BOS][img][text]` → `[img×N][BOS][text][\n]`. 원본 `build_string_from_input`과 일치 확인 |
| `max_length` / `truncation` | 원본에 없는 파라미터. 제거. 이미지 토큰이 잘리는 사고 경로를 차단 |
| `if pixel_values:` | 다중 원소 텐서는 `bool()`이 `RuntimeError`. `is not None`으로 수정 |
| `tokenizer.eos` | 존재하지 않는 속성. `eos_token_id`(=1)로 수정 |
| `argmax` shape | `keepdim=True` 없으면 `(B,)`가 되어 greedy 경로에서 터짐 |
| 빈 `generated` | 첫 토큰이 EOS이거나 `max_new_tokens=0`이면 `torch.cat([])`가 `ValueError`. 조기 반환 가드 필요 |
| `decode` 2D 입력 | `(1,N)` 텐서를 넘기면 `str`이 아니라 `list`가 반환됨. 1D로 넘겨야 함 |

### 참고: 랜덤 가중치에서 `\n`만 생성되는 이유

프롬프트 마지막 토큰이 `\n`(id 108)인데, 학습되지 않은 모델은 residual stream + weight tying 때문에 **방금 본 토큰을 그대로 예측**한다. 배선이 정상이라는 신호이기도 하다. 이 때문에 "토큰 1개 = 문자 1개"가 되어, 문자 길이로 토큰 수를 세는 테스트가 우연히 통과할 수 있으니 주의.
