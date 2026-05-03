# DepthMem: Scope–Frequency Memory Hierarchy for Hardware-Aware Iterative ISP

**Paper proposal v5 — final consolidated version**

---

## 0. One-line thesis

반복형 low-light denoising / restoration ISP 모델의 loop-depth hidden state는 spatial scope와 spatial frequency 축에서 균질하지 않다. DepthMem은 이 inhomogeneity를 4-bank memory hierarchy로 명시적으로 관리하여, edge NPU 환경에서 화질–메모리–지연시간 Pareto를 개선한다.

---

## 1. Problem framing

### 1.1 The gap

기존 looped Transformer / iterative restoration network는 같은 block을 반복 적용하는 구조다.

```
h_0 → h_1 → h_2 → ... → h_T
```

이 구조는 parameter footprint를 줄이는 대신 activation을 반복 통과시킨다. 그러나 **각 loop에서 생성되는 hidden state는 다음 loop 입력으로만 사용되고, 과거 loop의 정보를 어떤 형태로 보존할지에 대한 explicit design은 없다.** 다음 loop에 넘어가는 순간, 이전 loop의 정보 일부는 손실되거나 덮어쓰인다.

### 1.2 Research question

> _Iterative ISP의 loop-depth hidden state는 scope × frequency 축에서 inhomogeneous한가? 만약 그렇다면, 이 inhomogeneity를 hardware-aware memory hierarchy로 활용하여 quality–memory–latency Pareto를 개선할 수 있는가?_

핵심 가설은 **observation-driven**이다. 먼저 hidden state의 inhomogeneity를 probing으로 실증한 뒤, 이 관찰에 기반한 architecture를 설계한다.

### 1.3 Scope (확정)

| 항목            | 값                                                  |
| --------------- | --------------------------------------------------- |
| Task            | Low-light enhancement / denoising                   |
| Primary dataset | LOL 또는 SID (1차 후보), SIDD (secondary)           |
| Base model      | Tiny iterative NAFNet-style block, weight-tied, T=4 |
| Hardware target | edge NPU SRAM/DRAM hierarchy (가정 spec)            |
| Compute budget  | Kaggle P100/T4, 주당 약 45h                         |

LLM, Mamba, Universal Transformer는 related work에서 conceptual reference로만 인용한다. 실제 baseline 비교에서는 제외한다.

---

## 2. Related work positioning

### 2.1 Depth-recurrent computation

Universal Transformer는 Transformer에 depth-wise recurrence를 도입하고 ACT로 position별 halting을 학습했다. DepthMem은 이 개념적 계보 위에 있으나, **반복 자체가 아니라 반복 사이의 hidden-state memory management**를 다룬다.

### 2.2 Adaptive computation

ACT, PonderNet은 *언제 멈출 것인가*를 다룬다. DepthMem은 *반복 사이에 무엇을 기억할 것인가*를 다룬다. 두 축은 직교하며, early-exit은 본 논문의 메인 contribution이 아니다.

### 2.3 Sequence-time vs loop-depth memory

Mamba/SSM은 sequence-time recurrence의 state를 다룬다. DepthMem은 loop-depth recurrence의 state를 다룬다. 두 축은 다음과 같이 명확히 구별된다.

```
Mamba axis  : token / sequence-time
DepthMem axis : iterative refinement depth
```

### 2.4 Feature modulation

FiLM, adaLN, hypernetwork 계열은 modulation의 일반 형태를 제공한다. DepthMem은 modulation을 새로 제안하지 않는다. **Modulation은 압축된 memory bank를 읽기 위한 하드웨어 친화적 readout 구현**일 뿐이며, novelty는 modulation 자체가 아니라 4-bank memory decomposition과 lifetime policy에 있다.

### 2.5 Image restoration baselines

NAFNet, Restormer, MIRNet, DRUNet 등을 ISP 분야 reference로 둔다. 실제 비교 baseline은 ISP 내부 구조(iterative weight-tied NAFNet variants)로 한정한다.

### 2.6 Looped LM stability

Parcae는 looped language model의 residual stream을 nonlinear time-variant dynamical system으로 보고 injection parameter의 spectral norm 안정화를 제안했다. DepthMem은 동일 문제를 풀지 않는다. Parcae가 looped LM 안정화를 다룬다면, DepthMem은 **iterative ISP에서 memory-feedback path의 안정화**를 다룬다.

---

## 3. Pre-architectural probing study (the core)

DepthMem을 제안하기 전, 다음을 먼저 검증한다.

> _Do loop-depth hidden states in iterative ISP exhibit structured scope–frequency inhomogeneity?_

### 3.1 Probing setup

- Tiny iterative NAFNet-S (weight-tied, T=4)
- LOL training set, 128×128 patches
- Near-convergence까지 학습 (학습 미숙 시 hidden state가 random에 가까워 inhomogeneity가 가려짐)
- Probing은 epoch 10, 50, 100에서 반복 측정 — inhomogeneity가 학습 진행에 따라 emerge하는지 함께 관찰

### 3.2 Frequency probe — 2D radial spectrum

각 loop의 hidden state h_t ∈ ℝ^(H×W×C)에 대해 channel-averaged 2D FFT를 적용한다.

```
P_t(u, v) = mean_c |FFT2(h_t[:,:,c])|^2
E_t(r)    = Σ_{√(u²+v²)≈r} P_t(u, v)
```

측정 항목:

- Low-frequency energy ratio (r < r_low)
- High-frequency energy ratio (r > r_high)
- Loop-wise spectral shift

### 3.3 Scope probe — spatial autocorrelation length

```
Corr_t(Δx, Δy) = E[h_t(x,y) · h_t(x+Δx, y+Δy)]
ℓ_t = min{d : Corr_t falls below threshold at distance d}
```

### 3.4 Effective rank probe

H_t를 (H·W) × C로 펼친 뒤 SVD:

```
r_eff(t) = exp(-Σ_i p_i log p_i),   p_i = σ_i / Σ_j σ_j
```

### 3.5 Compressibility probe

각 loop hidden state를 다양한 압축률(1/2, 1/4, 1/8, 1/16) 및 precision(FP16/INT8/INT4)으로 압축한 뒤 다음 loop에 주입했을 때의 ΔPSNR/ΔSSIM/ΔLPIPS를 측정한다.

### 3.6 Decision tree

Phase 0의 결과에 따라 paper의 narrative가 갈라진다.

| Probing 결과                                    | Paper direction                                                           |
| ----------------------------------------------- | ------------------------------------------------------------------------- |
| Scope · frequency 모두 강한 inhomogeneity       | Full DepthMem 4-bank (계획대로)                                           |
| Scope만 강함                                    | Scope-only 2-bank로 축소                                                  |
| Frequency만 강함                                | Frequency-only 2-bank로 축소                                              |
| 둘 다 약함, but loop-position dependence는 존재 | Single memory + loop-position-conditioned readout으로 pivot               |
| 모든 loop이 거의 동일                           | DepthMem 폐기, "iterative ISP memory compression" 단일 주제 paper로 pivot |

이 decision tree를 사전에 명시하는 것 자체가 risk management의 일부다.

### 3.7 Probing tool로서의 standardization (선택적 격상)

만약 inhomogeneity가 강하게 나타나면, probing metric을 표준화된 diagnostic tool로 framing한다.

```
Spectral Inhomogeneity Score:    SIS = Var_t[E_t(r)] / Mean_t[E_t(r)]
Scope Drift:                     SD  = |ℓ_T − ℓ_0| / ℓ_0
Rank Compression Ratio:          RCR = r_eff(h_T) / r_eff(h_0)
```

이러면 contribution이 "DepthMem이라는 method 제안"에서 "iterative restoration network 일반에 적용 가능한 inhomogeneity diagnostic + 그것을 활용하는 method"로 격상된다.

---

## 4. DepthMem architecture

### 4.1 Two-axis decomposition

Probing이 scope × frequency inhomogeneity를 입증했다는 가정 하에, hidden state를 다음 4개 bank로 분해한다.

|                  | Low frequency | High frequency |
| ---------------- | ------------- | -------------- |
| **Global scope** | Global-Low    | Global-High    |
| **Local scope**  | Local-Low     | Local-High     |

각 bank의 low-light denoising 의미:

| Bank        | 역할 (LL+denoise)                            | 저장 정책              |
| ----------- | -------------------------------------------- | ---------------------- |
| Global-Low  | 전체 illumination level, 전반적 color cast   | 강한 압축, 긴 lifetime |
| Global-High | 전역 noise variance, texture/noise 분포 통계 | 통계 벡터만 저장       |
| Local-Low   | local shading, local exposure 변화           | tile summary           |
| Local-High  | local edge, weak texture, detail residual    | last-k buffer (k=1)    |

DepthMem은 4-bank를 *최종답*으로 주장하지 않는다. 본문에서는 **"scope와 frequency를 분리하는 minimal orthogonal decomposition"**으로 표현하며, 3-bank/8-bank와의 비교도 ablation에서 다룬다. 만약 phase 0의 hidden-state clustering(k-means/GMM)이 자연스럽게 4개 mode를 형성하면 그 자체가 4-bank의 empirical 정당화가 된다.

### 4.2 Compressors C^{s,f}

| Bank        | Compressor                                          | 출력 차원         |
| ----------- | --------------------------------------------------- | ----------------- |
| Global-Low  | `Linear(GlobalAvgPool(h_t))`                        | C_g (예: 32)      |
| Global-High | `ChannelStats(HighPass(h_t))` = [mean, var]/channel | 2C                |
| Local-Low   | `Conv1×1(TileAvgPool(h_t))` (tile=16)               | (H/16, W/16, C/4) |
| Local-High  | `Conv1×1(HighPass(h_t))`, last-k=1                  | (H, W, C/4)       |

`HighPass(h_t) = h_t − Upsample(TileAvgPool(h_t))`.

**Local-High precision 결정은 phase 0 결과에 의존한다.** Compressibility probe에서 INT8 quantization sensitivity를 먼저 측정한 뒤 결정한다. INT8이 PSNR을 0.3dB 이상 떨어뜨리면 FP16으로 둔다. 이 경우 INT8/INT4는 ablation only로 격하한다. SoC 강점을 살리려면 logarithmic quantization 또는 channel-wise scale을 도입한 INT8 variant도 별도 검토한다.

### 4.3 Readouts R_i (bank-specific)

**Low-frequency bank: broadcast modulation**

```
scale_i, bias_i = MLP(M_i)
F'_t = scale_i ⊙ Fθ(h_t) + bias_i
```

Global-Low은 [B, C] → [B, H, W, C] broadcast.  
Local-Low은 [B, H/16, W/16, C/4] → upsample → [B, H, W, C].

**High-frequency bank: additive residual**

```
r_i = Conv1×1(M_i)
h_{t+1} += γ_i · sigmoid(a_i) · r_i
```

Local-High은 spatial map 그대로 residual 추가.  
Global-High은 통계 벡터이므로, 직접 detail을 더하기보다 **Local-High readout strength를 조절하는 noise-level controller**로 사용한다.

```
noise_gate = MLP(M^{G,H})
h_{t+1} += noise_gate ⊙ r_{L,H}
```

이 design은 "global statistics가 local detail readout의 강도를 제어한다"는 mechanistic justification을 갖는다 — denoising 도메인에서 자연스럽고, reviewer의 "왜 4-bank인가" 공격에 대한 답이 된다.

### 4.4 Memory update — EMA + learnable β

```
M_{t+1}^{s,f} = (1 − β^{s,f}) M_t^{s,f} + β^{s,f} C^{s,f}(h_t)
```

β는 [0, 1]에 안전하게 머물도록 sigmoid로 reparameterize한다.

```
β^{s,f} = sigmoid(b^{s,f})       # b^{s,f}는 learnable scalar
```

| Bank        | b 초기값 | β 초기값 | Lifetime 의미  |
| ----------- | -------- | -------- | -------------- |
| Global-Low  | −2.2     | ≈ 0.10   | 가장 길게 유지 |
| Global-High | −0.85    | ≈ 0.30   | 중간           |
| Local-Low   | 0.0      | ≈ 0.50   | 중간           |
| Local-High  | 2.2      | ≈ 0.90   | 최근 위주      |

**학습 schedule**: β는 처음 N epoch 동안 frozen, 이후 unfreeze. 처음부터 함께 학습시키면 readout weight과 함께 변동이 커져 loss spike 위험이 있다.

β는 단순 hyperparameter가 아니라 **memory lifetime policy**이며, 동시에 **hardware schedule**의 명세다.

```
small β  →  long-lived storage (register / persistent SRAM)
large β  →  streaming / tile buffer
```

### 4.5 Stability — measurement + enforcement

DepthMem update:

```
h_{t+1} = Fθ(h_t) + Σ_i γ_i · sigmoid(a_i) · R_i(M_t^i)
```

**Enforcement 메커니즘:**

1. γ_i를 0 근처로 small initialization
2. Read strength sigmoid(a_i)로 [0, 1] bound
3. Compressors / readouts에 spectral normalization 또는 weight clipping
4. Local-High은 β=0.9로 short lifetime 강제
5. 학습 중 adaptive scaling: `||h_{t+1} − h_t|| > τ`이면 EMA로 γ_i 감쇠

**Measurement:**

- `||h_{t+1} − h_t||` (loop별 update norm)
- `||R_i(M_t^i)||` (bank별 readout magnitude)
- `γ_i · sigmoid(a_i)` (effective gate)
- `||h_t||` (state norm growth)

**Sufficient stability bound** (informal):

```
||J_F||₂ + Σ_i |γ_i · sigmoid(a_i)| · σ_max(R_i) · σ_max(C_i) · β_i / (1 − |1 − β_i|) < 1
```

엄밀한 만족은 어렵지만, 위 5가지 enforcement가 spectrum 측면에서 이 bound를 근사적으로 강제한다. Parcae의 negative diagonal parameterization과 동등한 수준의 strict guarantee는 아니며, 본 논문은 이를 **"empirically enforceable mechanism"**으로 정직하게 표현한다.

---

## 5. Hardware cost model

### 5.1 Activation R/W vs persistent state (정직한 회계)

| Method                 | Per-loop activation R/W | Persistent state across loops           |
| ---------------------- | ----------------------- | --------------------------------------- |
| Non-looped 4-block     | 1 × Y                   | 0                                       |
| Naive looped (T=4)     | T × Y                   | h_t only (current)                      |
| Full hidden checkpoint | T × Y                   | T × \|h\|                               |
| **DepthMem**           | T × Y                   | T × Σ_i \|M_i\| (with Σ\|M_i\| ≪ \|h\|) |

여기서 Y는 single-loop activation traffic, |h|는 single hidden state size.

**핵심 주장 (수정된 형태):**

> DepthMem은 naive looped 대비 per-loop activation R/W를 줄이지 않는다. DepthMem은 _full hidden checkpoint 대비 persistent loop-state memory를 줄이면서, 동일 budget에서 더 task-relevant한 정보를 보존한다._

### 5.2 Concrete numbers (가정값 기반)

가정:

- Feature map: 256 × 256 × 32, FP16
- Single hidden size: 256 × 256 × 32 × 2 bytes ≈ **4 MB**
- Loop count T = 4
- Edge NPU SRAM budget: 4 MB (가정), DRAM bandwidth: 10 GB/s (가정)

| Component                     | Size                        | Notes                                          |
| ----------------------------- | --------------------------- | ---------------------------------------------- |
| Single \|h\|                  | 4 MB                        | tile streaming 가정 시 SRAM에 잠재적 적재 가능 |
| Full checkpoint × 4           | 16 MB                       | SRAM 초과, DRAM spill 필수                     |
| **DepthMem total persistent** | target ≤ 256 KB or ≤ 512 KB | SRAM 잔여에 적재                               |

**DepthMem 4-bank 예상 크기:**

| Bank                                    | Size estimate                  |
| --------------------------------------- | ------------------------------ |
| Global-Low                              | 32 × 2 = 64 B                  |
| Global-High                             | 64 × 2 = 128 B                 |
| Local-Low (16×16 tile, C/4=8)           | 16 × 16 × 8 × 2 = 4 KB         |
| Local-High (full spatial, C/4=8, FP16)  | 256 × 256 × 8 × 2 = **1.0 MB** |
| Local-High (FP16, tile-pooled to 64×64) | 64 × 64 × 8 × 2 = **64 KB**    |
| Local-High (INT8, tile-pooled to 64×64) | 64 × 64 × 8 × 1 = **32 KB**    |

**중요한 관찰**: Local-High의 spatial resolution과 precision이 SRAM budget을 좌우한다. 256×256 full-spatial INT8조차 256KB(64×256×256×8b... = 0.5MB)를 넘는다. **Local-High은 반드시 spatial pooling 또는 channel reduction을 거쳐야** 256KB SRAM target에 들어간다. Phase 1 직전에 back-of-envelope로 다시 검산.

### 5.3 Roofline analysis

```
AI_eff = FLOPs_loop / (WeightBytes + ActivationBytes + MemoryStateBytes)
```

가상 NPU spec (공개 추정치 기반, 회사 IP 미사용):

- Peak: 4 TFLOPs FP16
- SRAM bandwidth: 50 GB/s
- DRAM bandwidth: 10 GB/s

Roofline plot에 다음 4점을 찍는다.

1. Non-looped — high weight bytes, low AI
2. Naive looped — low weight bytes, T배 activation, mid AI
3. Full checkpoint — low weight, large memory state, mid-low AI
4. DepthMem — low weight, compressed state, **highest AI**

DepthMem이 compute-bound region에 가장 가까이 위치한다는 것을 plot으로 보일 수 있으면, hardware claim의 격이 한 단계 올라간다.

### 5.4 Open-source NPU simulator로의 확장 (선택적 격상)

SCALE-Sim 또는 Timeloop으로 cycle/energy estimate까지 산출하면 MLSys-grade systems paper로 격상된다. 이는 SoC 출신의 unique 강점이다. 단 회사 internal IP 수치는 절대 사용하지 않으며, public NPU spec(Google Edge TPU, Apple Neural Engine 공개 추정치 등)을 기반으로만 한다.

---

## 6. Experiments — compute-realistic plan

### 6.1 Hero figure (단일 main result)

X-axis: persistent memory state bytes per inference  
Y-axis: PSNR / SSIM / LPIPS

곡선:

1. Non-looped baseline
2. Naive iterative (no memory)
3. Full hidden checkpoint
4. Scope-only 2-bank DepthMem
5. Frequency-only 2-bank DepthMem
6. **DepthMem 4-bank**
7. DepthMem 4-bank + INT8 Local-High (가능 시)

**Iso-X comparison 메시지**: 같은 memory state budget에서 DepthMem 4-bank가 더 높은 PSNR/SSIM을 달성한다. **Iso-Y comparison**: 같은 quality에서 DepthMem이 더 적은 memory state를 사용한다.

### 6.2 Iso-parameter / iso-FLOP 공정 비교

DepthMem은 4 compressors + 4 readouts + learnable β/γ/a를 추가한다. 이로 인한 parameter 증가가 무시할 수 없다. Reviewer의 "그냥 추가 파라미터 효과 아닌가" 공격을 막기 위해 다음 세 가지를 표로 제시한다.

| Comparison    | Baseline 보강                                    |
| ------------- | ------------------------------------------------ |
| Iso-parameter | Naive looped에 동등한 parameter의 추가 conv 삽입 |
| Iso-FLOP      | DepthMem과 동일 FLOP의 deeper / wider baseline   |
| Iso-memory    | 동일 memory state budget의 full-checkpoint 변형  |

세 비교 모두에서 DepthMem이 우위면 contribution이 강해진다.

### 6.3 Phased run schedule (P100/T4, 주 45h 기준)

**Phase 0 — Probing (Week 1, ~30h)**

- Tiny iterative NAFNet-S, T=4, C=32, patch=128
- LOL training set, near-convergence (1~2일 학습)
- Probing scripts (FFT spectrum, autocorrelation, effective rank, compressibility)
- Runs: 1~2

**판단 게이트**: 3.6의 decision tree에 따라 next-phase scope 결정.

**Phase 1 — Minimal DepthMem (Week 2, ~30h)**

- Naive iterative baseline
- Full hidden checkpoint
- DepthMem 4-bank (FP16 only)
- Runs: 3

**Phase 2 — Ablation (Week 3~4)**

- No memory
- Scope-only
- Frequency-only
- 4-bank
- 4-bank + INT8 Local-High (compressibility probe가 허용 시)
- Runs: 5

**Phase 3 — Robustness (선택)**

- T = 2, 4, 6
- β fixed vs learnable
- Memory budget sweep: 64 / 128 / 256 KB
- Runs: 3~4

**Phase 4 — Seed replication (best 3개만)**

- Baseline, DepthMem 4-bank, best ablation에 대해 seed 3개씩
- Runs: 9

**Total: 필수 9~10 runs, 선택 포함 22 runs.** 3 seeds 일괄 적용은 하지 않는다.

### 6.4 Dataset 결정

1차 후보: **LOL (Low-Light)** 또는 **SID (See-in-the-Dark)**.

이유:

- Low-light는 denoising + enhancement + color가 혼합된 ill-posed 문제로 iterative refinement의 의미가 강하다.
- 4-bank semantics(Global-Low = global illumination)와 직접 매핑된다.
- Single-pass NAFNet이 SOTA에 가까운 SIDD보다, iterative variant의 추가 이득 여지가 크다.

**SIDD는 secondary**로 generalization study에 둔다. SIDD에서 추가 실험이 필요하면 synthetic하게 noise level을 강하게 올린 SIDD-extreme variant를 사용한다.

### 6.5 Loss recipe (minimal)

```
L = L1 + γ_mem · L_memory_bitcost
```

이게 starting point다. Perceptual loss, color consistency, stability term 등은 ablation에서 추가하며 main recipe에 default로 포함하지 않는다. Multi-loss recipe는 reviewer의 "어떻게 tuning했냐"를 자초한다.

---

## 7. Failure modes (정직한 한계)

다음 시나리오를 paper 본문에 명시한다.

1. **T = 2 이하**: Memory bank 유지 비용이 이득보다 클 수 있다. DepthMem은 T ≥ 3에서 의미가 있다.
2. **약한 noise (σ ≤ 5)**: Single-pass가 충분히 좋아 iterative refinement 자체의 이득이 작다.
3. **INT4 Local-High**: low-light에서 weak signal과 noise의 magnitude가 비슷할 때 quantization이 신호를 잡아먹을 수 있다.
4. **Over-stabilization**: spectral norm constraint를 너무 강하게 걸면 refinement capacity 자체가 줄어든다.
5. **Frequency separation이 약한 task**: classification이나 high-level recognition에서는 4-bank 이점이 사라진다 (그래서 ISP-only로 좁혔다).
6. **Large baseline regime**: NAFNet-Large급에서 capacity gap이 줄어 DepthMem 이득이 축소될 가능성.

이런 negative result를 paper에 직접 보여주는 것이 modern reviewing에서 점수를 더 받는다. 첫 publication 단계에서는 1, 2, 3에 대한 ablation을 반드시 포함한다.

---

## 8. Risk management & pivot plans

### 8.1 Probing-driven pivot points (이미 §3.6에 명시)

### 8.2 Hardware claim pivot

만약 Local-High의 SRAM target 256KB가 spatial pooling 후에도 어렵다면:

- Larger SRAM (1MB) 가정으로 변경, 또는
- "memory state는 DRAM에 두되, last-k=1 버퍼만 SRAM resident"로 narrative 수정.

### 8.3 Baseline 강도 pivot

만약 iterative weight-tied NAFNet < non-tied deep NAFNet (parameter-matched)이면, looped architecture 자체의 정당성이 약하다. 이 경우:

- DepthMem을 looped 한정 method가 아닌 "**parameter-bounded NPU inference scenario**"로 framing.
- "Weight bandwidth가 제한된 edge NPU에서 parameter는 작게, compute는 늘리고 싶다"는 시나리오 강조.

### 8.4 Submission venue 단계

| 결과 강도                                     | 1차 venue                          | 2차 venue                                         |
| --------------------------------------------- | ---------------------------------- | ------------------------------------------------- |
| Probing 약함 + 부분 pivot                     | NeurIPS/ICLR efficient ML workshop | ECCV efficient computational photography workshop |
| Probing 강함 + DepthMem 작동                  | MLSys main                         | ICCP main                                         |
| 위 + Roofline + simulator + full 4-bank claim | NeurIPS/ICLR/CVPR main도 시도      | —                                                 |

**1차 목표는 MLSys 또는 efficient ML workshop**. SoC 출신의 hardware-aware ML positioning이 가장 살아나는 venue다.

---

## 9. Open issues / decisions to resolve before paper draft

1. **Dataset 최종 결정**: LOL vs SID — Phase 0 시작 전 선택. LOL이 더 작아서 P100/T4 친화적.
2. **Local-High precision 정책**: phase 0 compressibility probe 결과에 따라 FP16 / INT8 / log-INT8 결정.
3. **Local-High spatial resolution**: 256×256 full vs 64×64 tile-pooled — SRAM budget 검산 후 결정.
4. **β 학습 schedule**: frozen 기간 N epoch 결정 (default N=10 권장).
5. **Iso-comparison baselines**: Phase 1 시작 전 parameter-matched / FLOP-matched baseline의 정확한 spec 확정.
6. **회사 IP 충돌 확인**: 가정한 NPU spec(SRAM 4MB, DRAM 10GB/s 등)이 회사 internal IP와 무관함을 사전 확인.
7. **Affiliation / open-source 결정**: independent researcher vs academic affiliation. github.com/[ORG]/depthmem 공개 여부.

---

## 10. Two-week kickoff plan

### Week 1 — Probing baseline

| Day | Task                                                                            |
| --- | ------------------------------------------------------------------------------- |
| 1   | Tiny iterative NAFNet-S 구현, T=4, C=32                                         |
| 2   | LOL dataloader, 128×128 patch pipeline, training loop                           |
| 3–5 | Near-convergence 학습 (P100 1~2일)                                              |
| 6   | Probing scripts: FFT spectrum, autocorrelation, effective rank, compressibility |
| 7   | 결과 시각화, decision tree (§3.6) 적용해 next-phase scope 결정                  |

### Week 2 — Minimal DepthMem

| Day | Task                                                            |
| --- | --------------------------------------------------------------- |
| 1   | 4-bank compressor 구현 (§4.2)                                   |
| 2   | EMA + sigmoid β, bounded readout (§4.3, 4.4)                    |
| 3   | Stability measurement / enforcement 훅 (§4.5)                   |
| 4–5 | Phase 1 runs: naive iterative, full checkpoint, DepthMem 4-bank |
| 6   | Memory bytes 회계 script, hero figure 1차 plot                  |
| 7   | Phase 1 결과 확인 후 Phase 2 ablation 설계 확정                 |

이 2주가 끝나는 시점이 paper draft 작성 시작 시점이다. 그 전에는 본문 작성하지 않는다.

---

## 11. Abstract draft (실험 결과 가정 형태)

> Iterative image restoration models refine hidden representations over multiple recurrent steps, but existing looped designs treat intermediate states as transient activations. We hypothesize that loop-depth hidden states in iterative ISP exhibit structured inhomogeneity along two orthogonal axes — spatial scope and spatial frequency — and we verify this hypothesis through loop-wise probing on a tiny iterative NAFNet baseline trained for low-light denoising. Building on this observation, we introduce **DepthMem**, a hardware-aware loop-state memory hierarchy that decomposes hidden states into four scope–frequency memory banks (global-low, global-high, local-low, local-high), each with a distinct compressor, lifetime, precision, and storage policy. Banks are updated by learnable EMA rules and read through bounded, lightweight modulation or residual paths without reconstructing full hidden states. Under an edge-NPU memory model, DepthMem reduces persistent loop-state storage compared with full hidden checkpointing while preserving task-relevant refinement information, improving the quality–memory–latency Pareto. We further analyze memory-feedback stability, quantization-error compounding, and roofline-based arithmetic intensity to characterize the regime in which loop-state memory is beneficial.

**주의**: "improves the Pareto"는 실험 결과가 나온 후에만 사용한다. 결과 전 proposal 단계에서는 "is hypothesized to improve" 또는 "we evaluate whether DepthMem improves"로 둔다.

---

## 12. One-paragraph summary (한국어)

**DepthMem은 반복형 low-light denoising / restoration ISP 모델의 loop-depth hidden state를 spatial scope와 spatial frequency 축에서 분해해 4개 memory bank(global-low, global-high, local-low, local-high)로 관리하는 구조다. 각 bank는 서로 다른 compressor, lifetime(learnable EMA β), precision, storage policy를 가지며, full hidden state를 복원하지 않고 압축된 loop-state를 다음 반복의 modulation 또는 residual readout으로 사용한다. Edge NPU SRAM/DRAM 계층 가정 하에 full hidden checkpoint 대비 persistent memory state를 크게 줄이면서 task-relevant 정보를 보존하여, 화질–메모리–지연시간 Pareto를 개선하는 것이 목표다. 핵심 가설은 loop hidden state의 scope × frequency inhomogeneity이며, 이는 architecture 설계 전에 probing study로 먼저 실증한다 — 결과에 따라 4-bank, 2-bank, 또는 단일-memory + loop-position-conditioned readout으로 즉시 pivot할 수 있도록 decision tree가 사전에 정의되어 있다.**
