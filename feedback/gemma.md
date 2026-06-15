# Gemma Implementation Feedback (v4)

> v3(82점) → v4. Critical 버그 3개 모두 수정, kv_cache 테스트 추가. 수학 정확도가 11 → 17로 회복됐다.

## 점수: 91 / 100

| 항목 | v3 | v4 | 변화 이유 |
|---|---|---|---|
| 전체 아키텍처 이해 (20) | 19 | 19 | 변화 없음 |
| 수학/알고리즘 정확도 (20) | 11 | 17 | k RoPE 복구, cache concat 살아남, GELU variant 미세 차이 잔존 |
| API 설계 (20) | 18 | 18 | 변화 없음 |
| 코드 품질 (20) | 18 | 18 | `apply_rotary_emb` 주석 오류 잔존 |
| 테스트 (20) | 16 | 19 | kv_cache shape·consistency 테스트 추가 |

---

## 점수 히스토리

| 리뷰 | 종합 | 비고 |
|------|------|------|
| v2 (2026-06-12) | 79 | MLP bias=False 누락, OOM 배치, shadowing |
| v3 (2026-06-12) | 82 | 코드 품질 개선, but k RoPE regression |
| v4 (2026-06-15) | 91 | Critical 3개 모두 수정, kv_cache 테스트 추가 |

---

## v3 대비 잘 고친 것들

### ✅ k에 RoPE 적용 (Critical 수정)

**파일:** `paligemma/gemma/attention.py:66-68`

```python
k_states = k_states.reshape(batch_size, num_tokens, self.num_key_value_head, self.head_dim)
k_states = k_states.transpose(1, 2)          # (B, kv_heads, T, head_dim)
k_states = apply_rotary_emb(k_states, freq_cis)  # ← 올바른 shape에서 RoPE 적용
```

transpose 이후 올바른 shape에서 RoPE를 적용한다. v3의 regression이 완전히 해결됐다.

### ✅ kv_cache concat이 live code로 복구 (Critical 수정)

**파일:** `paligemma/gemma/attention.py:75-81`

```python
if kv_cache is not None:
    k_cache, v_cache = kv_cache
    k_states = torch.cat([k_cache, k_states], dim=2)  # RoPE 이후, expand 이전
    v_states = torch.cat([v_cache, v_states], dim=2)

k_expanded = k_states.repeat_interleave(self.num_groups, dim=1)  # concat 이후 expand
v_expanded = v_states.repeat_interleave(self.num_groups, dim=1)
```

순서가 정확히 맞다: RoPE → concat → expand → cache 저장.

### ✅ post-RoPE 상태로 cache 저장 (High 수정)

**파일:** `paligemma/gemma/attention.py:96`

```python
return attention_outputs, (k_states, v_states)  # k_states는 post-RoPE, pre-expand 상태
```

kv_heads 레벨 (expand 이전) 저장이라 다음 pass에서 concat 후 expand할 수 있다.

### ✅ kv_caches falsy check 수정

**파일:** `paligemma/gemma/model.py:24`

```python
cache_len = kv_caches[0][0].shape[2] if kv_caches is not None else 0  # ← is not None
```

### ✅ kv_cache 테스트 추가

**파일:** `test/gemma/test_gemma_model.py`

```python
def test_kv_cache_shape(model): ...      # cache 개수, T 차원 확인
def test_kv_cache_consistency(model): ... # full_out[:, 5:] == step_out 확인
```

v3 feedback에서 제안한 테스트 두 개가 그대로 구현됐다. `test_kv_cache_consistency`가 통과하면 autoregressive generation이 수학적으로 올바르다는 걸 검증한다.

### ✅ 오타 수정

`cahces` → `caches` (model.py:64)

---

## 남은 이슈

### 🟡 Medium — GELU variant 불일치

**파일:** `paligemma/gemma/modules.py:30`

```python
output = (self.up_proj(x) * nn.functional.gelu(self.gate_proj(x)))
#                                               ↑ 기본값 = exact GELU
```

Gemma 스펙은 `gelu_pytorch_tanh` (tanh 근사)를 사용한다.

```python
# 수정
nn.functional.gelu(self.gate_proj(x), approximate='tanh')
```

수치 차이는 작아서 silent failure이고, 사전학습 가중치 로드 시 미세하게 틀린 결과를 낸다.

---

### 🟢 Minor — `apply_rotary_emb` 주석이 틀림

**파일:** `paligemma/gemma/attention.py:26`

```python
) # (batch_size, seq_len, head_dim, hidden_dim // 2, 2) --> ...
```

실제 shape은 `(batch_size, num_head, seq_len, head_dim//2, 2)` — `seq_len`과 `num_head` 순서가 바뀌어 있다. 현재 입력이 `(B, num_head, seq_len, head_dim)`인데 주석이 v2 시절 순서를 그대로 반영하고 있다.

---

### 🟢 Minor — `__main__` 배치 크기 OOM 위험

**파일:** `paligemma/gemma/attention.py:102`, `paligemma/gemma/decoder.py:32`

```python
x = torch.randn((100, 200, 2048))  # attention.py: B=100, T=200 — OOM 위험
x = torch.randn((100, 200, ...))   # decoder.py:   동일
```

테스트는 배치 2~4로 수정됐는데 `__main__` 블록은 아직 100. `(2, 50, 2048)` 수준으로 낮추면 충분하다.

---

### 🟢 Minor — `kv_cache` / `kv_caches` 이름 불일치

**파일:** `model.py:17, 47`

`GemmaForCausalLM.forward`의 파라미터는 `kv_cache` (단수), `GemmaModel.forward`는 `kv_caches` (복수). 동작에는 무관하지만 API를 읽는 사람이 헷갈린다.

---

## 수정 우선순위

```
1. modules.py:30   GELU approximate='tanh' 추가   ← 스펙 정확도
2. attention.py:26 apply_rotary_emb 주석 수정     ← 혼동 방지
3. attention.py:102, decoder.py:32  __main__ 배치 크기  ← trivial
4. model.py:17/47  kv_cache 이름 통일            ← trivial
```

---

## 전체 평가

Critical 3개 (k RoPE, dead cache concat, pre-RoPE 저장)가 한 번에 깔끔하게 수정됐고, kv_cache consistency 테스트까지 추가해서 수학적 정확성이 검증된다. 91점은 학습 목적 구현으로서 매우 높은 수준이다.

남은 이슈 중 실질적으로 의미 있는 건 GELU variant 하나뿐이고, 나머지는 모두 사소하다. GELU를 고치면 93점 수준으로 올라간다.
