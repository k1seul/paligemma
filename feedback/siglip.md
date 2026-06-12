# SigLIP Implementation Feedback (v1)

> 첫 번째 리뷰. 수학적으로 정확하고 구조도 탄탄하다. Gemma와 달리 critical 버그가 없다.

## 점수: 81 / 100

| 항목 | 점수 | 비고 |
|---|---|---|
| 전체 아키텍처 이해 (20) | 18 | ViT 구조 정확, 불필요한 상속 |
| 수학/알고리즘 정확도 (20) | 18 | 전반적으로 정확, 연산자 우선순위 OK |
| API 설계 (20) | 16 | SiglipVisionModel이 사실상 불필요 |
| 코드 품질 (20) | 15 | hardcoded cuda, 명명 관례 일부 이탈 |
| 테스트 (20) | 14 | fixture scope 없음, OOM 가능 배치, 검증 범위 얕음 |

---

## 잘 된 것들

- **GELU approximate="tanh"** — SigLIP 공식 구현과 일치 (`attention.py:57`)
- **LayerNorm 위치** — pre-norm 적용 순서 정확 (`encoder.py:19, 23`)
- **patch embedding** — Conv2d로 정확하게 구현, `padding="valid"` 명시 (`embeddings.py:10-16`)
- **position_ids** — `register_buffer(persistent=False)` 적절 (`embeddings.py:21-25`)
- **attention scaling** — `q @ k_T * scale` 연산자 우선순위 확인 OK (`attention.py:34`)
- **attention dropout** — `training=self.training` 으로 train/eval 분기 정확 (`attention.py:37`)
- **parameter count** — 테스트 범위 85M~90M이 ViT-B/16 실제 값(87M)에 정확히 맞음

---

## 버그 및 개선 목록

### 🟠 High — SiglipVisionModel이 불필요한 상속

**파일:** `paligemma/siglip/model.py:24-29`

```python
class SiglipVisionModel(SiglipVisionTransformer):
    def __init__(self, config : SiglipVisionConfig | None):
        if config is None:
            config = SiglipVisionConfig()
        super().__init__(config)
```

`SiglipVisionModel`은 `config=None` 처리 외에 아무것도 추가하지 않는다.
현재 구조에서 `SiglipVisionTransformer`가 공개 API이고 `SiglipVisionModel`은 그 위에 있는데,
실제 HuggingFace SigLIP에서는 `SiglipVisionModel`이 곧 전체 모델이다.
두 클래스를 분리해야 할 이유가 없으니 `SiglipVisionModel`로 합치는 게 낫다:

```python
class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig | None = None):
        super().__init__()
        if config is None:
            config = SiglipVisionConfig()
        self.config = config
        self.embedding = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.encoder(x)
        return self.layernorm(x)
```

---

### 🟡 Medium — `__main__` hardcoded "cuda"

**파일:** `paligemma/siglip/model.py:37`

```python
img = torch.randn((10, 3, 224, 224)).to("cuda")  # cuda가 없으면 RuntimeError
```

테스트 파일들은 이미 `"cuda"` 고정이지만 `__main__`은 여전히 그대로:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
img = torch.randn((10, 3, 224, 224)).to(device)
model = SiglipVisionModel(config).to(device)
```

---

### 🟡 Medium — `num_hidden_layer` 명명 관례

**파일:** `paligemma/siglip/config.py:6`

```python
num_hidden_layer : int = 12  # 's' 없음
```

PyTorch 및 HuggingFace 전체 관례가 `num_hidden_layers`(복수). 내부적으로는 일관되지만 나중에 HuggingFace 가중치를 로드할 때 필드 매핑이 어긋날 수 있다.

---

### 🟡 Medium — `test_embeddings.py` 배치 크기 OOM 가능

**파일:** `test/siglip/test_embeddings.py:21`

```python
input_image = torch.randn(128, 3, 224, 224).to(device)
```

embedding layer 자체는 작지만 입력 tensor가 128 × 3 × 224 × 224 × 4 bytes ≈ 1.5 GiB.
GPU에서 다른 모델 테스트와 동시에 실행되면 OOM 가능. `(4, 3, 224, 224)`로 줄여도 shape 검증에 충분하다.

---

### 🟢 Minor — fixture `scope` 미지정

**파일:** `test/siglip/test_siglip_model.py:6`

```python
@pytest.fixture          # scope 없음 → 테스트마다 모델 새로 생성
def model():
```

모델이 4번 생성·소멸된다. `scope="session"`으로 바꾸면 한 번만 생성된다:

```python
@pytest.fixture(scope="session")
def model():
```

Gemma 테스트 (`test_gemma_model.py:6`)는 이미 `scope="session"` 적용 중 — 같은 패턴으로 통일.

---

### 🟢 Minor — `test_deterministic_in_eval` 중복 device 이동

**파일:** `test/siglip/test_siglip_model.py:27`

```python
def test_deterministic_in_eval(model):
    device = "cuda"
    model.to(device)  # ← fixture에서 이미 cuda로 만들었음, 불필요
```

---

### 🟢 Minor — `embeddings.py __main__` device 미지정

**파일:** `paligemma/siglip/embeddings.py:38`

```python
img = torch.randn(100, 3, 224, 224)   # CPU에서 실행 — 배치 100이라 느림
embed = embedding(img)
```

`device` 지정 없이 배치 100으로 실행하면 CPU에서 매우 느리다.

---

## 테스트 보강 제안

현재 테스트는 shape·파라미터 수·결정론성·역전파·NaN만 확인한다. 다음을 추가하면 구현 신뢰도가 높아진다:

```python
def test_patch_count(model):
    # 224/16 = 14, 14*14 = 196 패치
    x = torch.randn(1, 3, 224, 224).to("cuda")
    out = model(x)
    assert out.shape[1] == 196

def test_attention_weights_sum_to_one():
    # softmax 이후 attention weights가 각 query에 대해 합 1
    from paligemma.siglip.attention import SiglipAttention
    attn = SiglipAttention(SiglipVisionConfig()).to("cuda")
    x = torch.randn(2, 196, 768).to("cuda")
    _, weights = attn(x)
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)), atol=1e-5)
```

---

## 수정 우선순위

```
1. model.py:24-29     SiglipVisionModel 상속 제거      ← 구조 명확성
2. config.py:6        num_hidden_layers 복수형          ← HuggingFace 호환
3. model.py:37        hardcoded cuda 제거               ← 이식성
4. test_embeddings:21 배치 크기 축소                    ← OOM 방지
5. test_siglip:6      fixture scope="session" 추가      ← 테스트 속도
```

---

## 전체 평가

SigLIP 구현은 안정적이다. ViT의 핵심 요소(patch embedding, pre-norm, MHSA, GELU-tanh, LayerNorm)가 모두 정확하게 구현됐고 수학적으로 틀린 부분이 없다.

남은 이슈들은 대부분 구조·관례·편의성에 관한 것이라 Gemma의 Critical 버그와는 성격이 다르다.
`SiglipVisionModel` 상속 정리와 `num_hidden_layers` 복수형 수정이 가장 의미있는 개선이다.
