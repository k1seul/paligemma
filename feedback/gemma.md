# Gemma Implementation Feedback (v3)

> v2(79점) → v3. 코드 품질 측면에서 의미있는 진전이 있었지만, `k`에 RoPE가 전혀 적용되지 않는 regression이 생겼다.

## 점수: 82 / 100

| 항목 | v2 | v3 | 변화 이유 |
|---|---|---|---|
| 전체 아키텍처 이해 (20) | 19 | 19 | 변화 없음 |
| 수학/알고리즘 정확도 (20) | 13 | 11 | k에 RoPE 없음(regression), cache concat dead code |
| API 설계 (20) | 18 | 18 | 변화 없음 |
| 코드 품질 (20) | 15 | 18 | bias=False 수정, shadowing 수정, device hardcoding 수정 |
| 테스트 (20) | 14 | 16 | OOM 배치 크기 수정 |

---

## v2 대비 잘 고친 것들

- `MLP bias=False` 추가 (`modules.py:25-27`) — 체크포인트 로드 시 shape mismatch 해결
- `new_kv_cache` 변수명 사용 (`decoder.py:20`) — 입력값 shadowing 제거
- 테스트 배치 크기 `(100, 50)` → `(4, 50)` — OOM 방지
- `__main__` 전체에 `"cuda" if torch.cuda.is_available() else "cpu"` 적용

---

## 남은 버그 목록

### 🔴 Critical — attention의 k에 RoPE가 전혀 없음 (regression)

**파일:** `paligemma/gemma/attention.py:67-81`

현재 코드를 실행 순서대로 추적하면:

```python
k_states = k_states.reshape(batch_size, num_tokens, self.num_key_value_head, self.head_dim)
k_origin = k_states.transpose(1, 2)           # (B, kv_heads, T, head_dim) — pre-RoPE 저장
k_states = apply_rotary_emb(k_states, freq_cis)  # ← (B, T, kv_heads, head_dim)에 잘못 적용
                                               #   freqs_cis broadcasting이 잘못되어
                                               #   결과가 (B, T, T, head_dim)으로 오염됨
# ... kv_cache concat ...
k_states = k_origin.repeat_interleave(self.num_groups, dim=1)  # ← k_origin (pre-RoPE!)으로 덮어씀
                                                                #   위 cat 결과는 버려짐
```

**실제로 attention에 들어가는 k는 `k_origin`으로, RoPE가 전혀 없다.**
q에는 RoPE가 있고 k에는 없으니 dot-product에서 위치 정보가 아예 사용되지 않는다.

v2에서는 `k_origin.repeat_interleave` → `apply_rotary_emb` 순서라 비효율적이었지만 RoPE는 적용됐다. 현재 코드는 그보다 오히려 퇴보.

**수정 방법:**

```python
# 1. k/v는 kv_heads 해상도에서 transpose 후 RoPE 적용
k_states = k_states.reshape(batch_size, num_tokens, self.num_key_value_head, self.head_dim)
k_states = k_states.transpose(1, 2)              # (B, kv_heads, T, head_dim)
k_states = apply_rotary_emb(k_states, freq_cis)  # ← 올바른 shape, RoPE 적용

# 2. cache concat (RoPE 이후, expand 이전)
if kv_cache is not None:
    k_cache, v_cache = kv_cache
    k_states = torch.cat([k_cache, k_states], dim=2)
    v_states = torch.cat([v_cache, v_states], dim=2)

# 3. concat 이후에 expand
k_expanded = k_states.repeat_interleave(self.num_groups, dim=1)
v_expanded = v_states.repeat_interleave(self.num_groups, dim=1)

# 4. expand 이전 (kv_heads 레벨) 상태로 캐시 저장
return attention_outputs, (k_states, v_states)
```

---

### 🔴 Critical — kv_cache concat이 dead code

**파일:** `paligemma/gemma/attention.py:75-81`

```python
if kv_cache is not None:
    k_cache, v_cache = kv_cache
    k_states = torch.cat([k_cache, k_states], dim=2)  # 결과가 다음 줄에서 바로 덮어써짐
    v_states = torch.cat([v_cache, v_states], dim=2)  # 결과가 다음 줄에서 바로 덮어써짐

k_states = k_origin.repeat_interleave(...)  # ← k_states 완전 교체
v_states = v_origin.repeat_interleave(...)  # ← v_states 완전 교체
```

kv_cache가 있어도 없어도 결과가 동일 — autoregressive generation이 작동하지 않는다.
위 Critical 1번 수정 방법대로 고치면 이 버그도 함께 해결된다.

---

### 🟠 High — kv_cache에 pre-RoPE k 저장

**파일:** `paligemma/gemma/attention.py:96`

```python
return attention_outputs, (k_origin, v_origin)  # k_origin은 RoPE 없는 상태
```

1번 수정과 세트로 해결된다. 캐시는 RoPE 적용 이후 상태로 저장해야 한다.

---

### 🟡 Medium — `if kv_caches:` falsy check

**파일:** `paligemma/gemma/model.py:24`

```python
kv_cache = kv_caches[i] if kv_caches else None
```

`kv_caches`가 빈 리스트 `[]`이면 `False`로 평가 → 캐시가 있어도 None으로 취급.

```python
kv_cache = kv_caches[i] if kv_caches is not None else None
```

---

### 🟡 Medium — kv_cache 테스트 없음

**파일:** `test/gemma/test_gemma_model.py`

현재 테스트는 single-pass forward만 검증한다. Autoregressive generation이 핵심 기능인데 다음 케이스가 없다:

```python
def test_kv_cache_shape(model):
    device = "cuda"
    input_ids = torch.randint(0, 30000, (2, 5)).to(device)
    _, caches = model(input_ids)
    assert len(caches) == model.config.num_hidden_layers
    assert caches[0][0].shape[2] == 5  # T_prev == 5

def test_kv_cache_consistency(model):
    # cache 사용 결과가 full context 결과와 일치해야 함
    device = "cuda"
    input_ids = torch.randint(0, 30000, (2, 6)).to(device)
    model.eval()
    with torch.no_grad():
        full_out, _ = model(input_ids)
        step_out, caches = model(input_ids[:, :5])
        next_out, _ = model(input_ids[:, 5:], kv_cache=caches)
    assert torch.allclose(full_out[:, 5:], next_out, atol=1e-4)
```

---

### 🟢 Minor — `__main__` 변수 오타

**파일:** `paligemma/gemma/model.py:56`

```python
out, cahces = model(input_ids)  # cahces → caches
```

---

## 수정 우선순위

```
1. attention.py:67-81  k RoPE 없음 + dead cache code  ← inference 전체 망가짐, 최우선
2. attention.py:96     pre-RoPE cache 저장             ← 1번 수정과 세트
3. model.py:24         kv_caches falsy check            ← cache 로직 전반 신뢰성
4. test/gemma/         kv_cache 테스트 추가             ← 수정 검증 필수
5. model.py:56         오타                             ← trivial
```

---

## 전체 평가

코드 품질 면에서 확실한 진전 — `bias=False`, 변수 shadowing, device hardcoding이 모두 해결됐다.
그러나 attention 로직을 손대면서 v2에선 비효율적으로나마 작동하던 k RoPE가 완전히 사라졌다.
현재 모델은 q에만 위치 정보가 있고 k에는 없어서, attention score에 위치가 전혀 반영되지 않는다.

Critical 2개(k RoPE + dead cache code)가 사실상 한 묶음이니, 수정 방법 섹션대로 attention.py를 고치면
수학/알고리즘 점수가 11 → 17 이상으로 오르고 총점도 88점대로 올라갈 것이다.
