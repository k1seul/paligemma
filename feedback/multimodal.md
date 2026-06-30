# Multimodal (PaliGemma) Implementation Feedback (v2)

> v1 피드백 반영 후 재평가. Critical/High/Medium 버그가 전부 수정되고 pytest가 추가됐다. kv_cache와 causal masking 관련 항목(chapter 4)은 미완성 상태로 평가에서 제외하고 별도 표기.

## 점수: 79 / 100

| 항목 | v1 | v2 | 비고 |
|---|---|---|---|
| 전체 아키텍처 이해 (20) | 16 | 16 | prefix LM attention 미구현(chapter 4 보류) |
| 수학/알고리즘 정확도 (20) | 11 | 18 | reshape 수정, Resize 튜플 수정 |
| API 설계 (20) | 13 | 18 | 타입 어노테이션 수정, kv_cache/attention_mask 전달 추가 |
| 코드 품질 (20) | 14 | 19 | config 오타 수정, nn.Module 제거, dead import 제거 |
| 테스트 (20) | 5 | 15 | pytest 2개 추가, __main__ 블록에 `int64` 잔존 |

---

## v1 → v2 변경사항

### ✅ 수정 완료

| 우선순위 | 항목 | 파일 |
|---|---|---|
| 🔴 Critical | `pixel_values = torch.Tensor` → `: torch.Tensor` 타입 어노테이션 | `model.py:17` |
| 🔴 Critical | `image_embedding.reshape(-1, D)` shape 불일치 수정 | `model.py:23` |
| 🟠 High | `Resize((image_size, image_size))` 튜플로 수정 | `input_processer.py:12` |
| 🟠 High | `kv_cache`, `attention_mask` forward 인자로 받고 반환 | `model.py:17,25-31` |
| 🟡 Medium | `text_conifg` → `text_config` 오타 수정 | `config.py`, `projector.py` |
| 🟡 Medium | `PaliGemmaProcessor` nn.Module 상속 제거 | `input_processer.py` |
| 🟡 Medium | 미사용 pixtral import 제거 | `input_processer.py` |
| 🟢 Minor | `__main__` 블록 `num_image_tokens=256 → 196` 수정 | `model.py:57` |

### ❌ 미수정

- **`projector.py:33` `.to(torch.int64)` 잔존** — float 임베딩을 정수로 버리는 코드가 아직 남아 있다. shape 확인 목적의 `__main__` 블록이라도 삭제해야 한다.

```python
# 현재 (잘못됨)
projected = projector(siglip_out).to(torch.int64)

# 수정
projected = projector(siglip_out)
```

---

## 테스트 추가 (신규)

`test/multimodal/test_paligemma_model.py`에 pytest 2개가 추가됐다.

**잘 된 것:**
- `scope="session"` fixture로 모델을 한 번만 로드 — 메모리 효율적
- `test_forward_shape`: logits shape `(B, seq_len, vocab_size)` 검증
- `test_image_tokens_replaced`: 이미지 토큰 196개가 input_ids에 정확히 들어가는지 확인

**부족한 점:**
- device를 `"cuda"`로 하드코딩 — CPU 환경에서 실패한다. `pytest.mark.skipif` 또는 `device = "cuda" if torch.cuda.is_available() else "cpu"` 사용 권장
- causal masking 동작 검증이 없음 (chapter 4 이후 추가 예정)

---

## 추가 권장 테스트

현재 2개 테스트로는 파이프라인의 외형(shape)만 확인된다. 아래는 각 컴포넌트의 **동작 정확성**을 검증하는 케이스들이다.

### 1. 전처리 정규화 범위 — `test_pixel_values_range`

```python
def test_pixel_values_range(model_and_processor):
    _, processor = model_and_processor
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    out = processor("test", img)
    pv = out["pixel_values"]
    assert pv.min() >= -1.0 and pv.max() <= 1.0
```

`Normalize(mean=0.5, std=0.5)` 가 `[0, 1] → [-1, 1]`로 매핑하는지 확인한다. 정규화 파라미터를 잘못 쓰면 범위가 벗어나서 SigLIP이 out-of-distribution 입력을 받는다. 가장 자주 틀리는 전처리 버그 중 하나다.

---

### 2. 비정방형 이미지 입력 — `test_non_square_image_resize`

```python
def test_non_square_image_resize(model_and_processor):
    _, processor = model_and_processor
    portrait_img = Image.fromarray(np.zeros((400, 200, 3), dtype=np.uint8))  # H > W
    landscape_img = Image.fromarray(np.zeros((200, 600, 3), dtype=np.uint8))  # W > H
    for img in [portrait_img, landscape_img]:
        out = processor("test", img)
        assert out["pixel_values"].shape == (1, 3, 224, 224)
```

`Resize(int)` 대신 `Resize((H, W))` 튜플로 수정한 것이 실제로 동작하는지 검증한다. 이 테스트 없이는 수정이 맞는지 실행 전까지 알 수 없다.

---

### 3. 이미지 토큰 위치 — `test_image_tokens_at_start`

```python
def test_image_tokens_at_start(model_and_processor):
    model, processor = model_and_processor
    out = processor("hello", Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)))
    ids = out["input_ids"][0]
    image_token_id = model.config.img_token_id
    image_positions = (ids == image_token_id).nonzero(as_tuple=True)[0]
    assert image_positions[0].item() == 0           # 첫 번째 토큰이 이미지
    assert image_positions[-1].item() == 195        # 196번째까지 이미지
    assert (ids[196:] != image_token_id).all()      # 이후엔 이미지 토큰 없음
```

현재 `test_image_tokens_replaced`는 개수(196)만 센다. 이 테스트는 이미지 토큰이 **앞에** 모여 있는지 확인한다. `image_tokens + text` 순서가 바뀌면 개수는 맞아도 embedding 교체 로직이 어긋난다.

---

### 4. Projector 출력 dtype — `test_projector_output_dtype`

```python
def test_projector_output_dtype(model_and_processor):
    model, _ = model_and_processor
    device = next(model.parameters()).device
    dummy_siglip_out = torch.randn(1, 196, model.config.vision_config.hidden_size, device=device)
    projected = model.multi_modal_projector(dummy_siglip_out)
    assert projected.dtype == torch.float32
    assert projected.shape == (1, 196, model.config.text_config.hidden_size)
```

Projector가 float을 유지하는지 확인한다. `projector.py:33`의 `.to(torch.int64)` 버그가 실제 forward에 섞여 들어오면 이 테스트가 즉시 잡는다. shape도 같이 검증해서 projector 단독 테스트가 된다.

---

### 5. kv_cache 구조 — `test_kv_cache_structure`

```python
def test_kv_cache_structure(model_and_processor):
    model, processor = model_and_processor
    img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    out = processor("test", img)
    _, kv_cache = model(out["input_ids"].cuda(), out["pixel_values"].cuda())
    assert kv_cache is not None
    assert len(kv_cache) == model.config.text_config.num_hidden_layers  # 레이어 수만큼
    k, v = kv_cache[0]
    assert k.shape[1] == model.config.text_config.num_key_value_heads   # head 수
```

forward가 kv_cache를 반환하기는 하는지, 레이어 수와 구조가 맞는지 검증한다. chapter 4에서 cache를 재사용할 때 이 구조가 틀려 있으면 즉시 shape mismatch가 난다.

---

### 6. eval 모드 결정론적 출력 — `test_eval_mode_deterministic`

```python
def test_eval_mode_deterministic(model_and_processor):
    model, processor = model_and_processor
    model.eval()
    img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    out = processor("test", img)
    ids, pv = out["input_ids"].cuda(), out["pixel_values"].cuda()
    with torch.no_grad():
        logits1, _ = model(ids, pv)
        logits2, _ = model(ids, pv)
    assert torch.allclose(logits1, logits2)
```

같은 입력에 같은 출력이 나오는지 확인한다. Dropout이 eval에서 꺼지지 않거나 내부에 stochastic 연산이 있으면 이 테스트가 실패한다. 현재 구현에 Dropout은 없지만, 나중에 추가할 때 실수로 train 모드로 실행하는 것을 방지한다.

---

### 7. 배치 크기 2 — `test_batch_size_2`

```python
def test_batch_size_2(model_and_processor):
    model, processor = model_and_processor
    imgs = [Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)) for _ in range(2)]
    outs = [processor("test", img) for img in imgs]
    input_ids = torch.cat([o["input_ids"] for o in outs], dim=0).cuda()
    pixel_values = torch.cat([o["pixel_values"] for o in outs], dim=0).cuda()
    logits, _ = model(input_ids, pixel_values)
    assert logits.shape[0] == 2
```

배치 연산이 실제로 동작하는지 확인한다. `image_embedding.reshape(-1, D)` 수정이 올바른지 배치 > 1에서 검증하는 유일한 테스트다. 배치 1에서는 Critical 버그가 숨어 있어도 통과할 수 있었다. (GPU 11.6 GiB 기준 배치 2는 안전하다.)

---

## Chapter 4 이후 구현할 항목

다음 두 기능은 chapter 4 미진행으로 현재 평가에서 제외하고 별도 추적한다.

### prefix LM attention mask

PaliGemma의 핵심 특징: **이미지 토큰은 bidirectional attention, 텍스트 토큰은 causal attention**.
현재 구현은 `GemmaModel` 내부에서 전체 시퀀스에 causal mask를 일괄 적용하고 있다.

```
올바른 attention 패턴 (이미지 196토큰 + 텍스트 10토큰):
          img[0..195]  txt[0..9]
img[0]  [  1  1  1 |  0  0  0  ]  ← 이미지끼리 서로 다 봄
img[1]  [  1  1  1 |  0  0  0  ]
txt[0]  [  1  1  1 |  1  0  0  ]  ← 이미지 전체 + 이전 텍스트만
txt[1]  [  1  1  1 |  1  1  0  ]
```

이 mask를 `PaliGemmaForConditionalGeneration.forward`에서 생성해서 `GemmaForCausalLM`에 전달해야 한다.

### `generate` 메서드

현재 `forward`만 있어서 prefill 1번만 실행된다. Autoregressive 추론을 위해 필요:

```python
def generate(self, input_ids, pixel_values, max_new_tokens=50):
    output, kv_cache = self.forward(input_ids, pixel_values)
    tokens = [output[:, -1:, :].argmax(dim=-1)]
    for _ in range(max_new_tokens - 1):
        next_logits, kv_cache = self.language_model(input_ids=tokens[-1], kv_cache=kv_cache)
        next_token = next_logits[:, -1:, :].argmax(dim=-1)
        tokens.append(next_token)
        if next_token.item() == eos_token_id:
            break
    return torch.cat(tokens, dim=1)
```

---

## 전체 평가

v1에서 실행 시 즉시 크래시가 나던 Critical 버그 2개가 수정되고, shape 오류와 Resize 버그도 고쳐졌다. kv_cache/attention_mask 전달 구조도 올바르게 잡혔고, 코드 품질 문제(오타, 불필요한 상속, dead import)도 모두 정리됐다.

pytest 추가는 큰 개선이다. `forward_shape`과 `image_tokens_replaced` 두 테스트가 파이프라인의 핵심 경로를 검증한다.

남은 과제는 chapter 4 항목인 prefix LM attention mask와 `generate` 메서드다. 이 두 가지가 완성되면 실제 추론까지 동작하는 구현이 된다.
