# RecurISP: Looped DyT-LoRA 기반 Mobile Neural ISP 구조 제안

## 0. 개요

본 문서는 Camera ISP(Image Signal Processing)에 **Looped Transformer**, **DyT/DyT-HDR**, **LoRA**, **Adaptive Early-Exit**를 결합한 새로운 Neural ISP 구조인 **RecurISP** 제안 내용을 정리한다.

초기 아이디어는 “Looped Transformer + DyT + LoRA를 neural ISP에 결합한다”는 형태였으나, 리뷰 과정에서 단순한 최신 기법 조합으로 보일 위험이 지적되었다. 이에 따라 최종 구조는 다음과 같이 재정의되었다.

> **RecurISP는 shared recurrent ISP backbone 위에 iteration-conditioned low-rank update를 결합하여, 적은 파라미터로 반복적 ISP refinement를 수행하고, 조건별 early-exit을 통해 모바일 NPU 친화적인 adaptive compute를 제공하는 Neural ISP 구조이다.**

핵심 novelty는 단순히 Looped Transformer, DyT, LoRA를 결합하는 것이 아니라, **비생성형 RAW-to-RGB neural ISP refinement 환경에서 shared recurrent backbone에 iteration-conditioned LoRA를 넣고, sensor/capture condition 및 low-iteration adaptive compute와 함께 검증하는 것**이다.

주의할 점은 다음이다.

```text
Per-iteration LoRA 자체는 완전히 새로운 일반 기법으로 주장하지 않는다.

Novelty boundary:
    recurrent neural ISP
    low-iteration RAW/HDR refinement
    sensor/capture-conditioned LoRA scaling
    shared LoRA vs per-iteration LoRA의 ISP-specific 분석
```

---

## 1. 배경 및 문제 정의

### 1.1 기존 Neural ISP의 한계

기존 Neural ISP는 RAW-to-RGB, denoising, demosaicing, color correction, tone mapping 등을 하나의 CNN 또는 Transformer 기반 네트워크로 처리한다.

일반적인 문제는 다음과 같다.

- 다양한 sensor, ISO, exposure, gain, color temperature 조건을 하나의 모델이 모두 처리해야 한다.
- 고품질 ISP를 위해 모델이 커지면 모바일 NPU에서 latency, SRAM traffic, power 문제가 커진다.
- 센서별 tuning이 필요할 경우 backbone 전체를 재학습하거나 별도 모델을 유지해야 한다.
- Transformer 기반 ISP는 표현력은 좋지만 LayerNorm, attention, activation memory, SRAM access 비용이 크다.
- 모든 입력에 동일한 compute를 쓰기 때문에 daylight scene과 low-light scene 간 계산량 차이를 반영하지 못한다.

---

### 1.2 목표

RecurISP의 목표는 다음과 같다.

1. **Small recurrent adaptive ISP engine**
   - 큰 ISP 모델 대신 작은 shared block을 반복적으로 사용한다.

2. **Iteration-conditioned low-rank refinement**
   - 동일 backbone을 공유하되 iteration별 LoRA를 사용하여 표현력을 확장한다.

3. **ISP-aware residual nonlinear stabilizer**
   - DyT-HDR를 normalization 자체의 대체가 아니라 RAW/HDR feature를 위한 residual soft-saturating nonlinearity로 검증한다.

4. **Sensor/ISO/scene adaptive ISP**
   - sensor_id, ISO, gain, exposure, CCT 등의 condition에 따라 LoRA scaling과 early-exit을 제어한다.

5. **Mobile NPU-friendly deployment**
   - offline-fused LoRA, adaptive iteration count, reduction-free DyT-HDR 등을 통해 latency, memory, power 측면의 효율성을 목표로 한다.

---

## 2. 최종 구조명

후보 이름은 여러 가지가 있었다.

- Loop-DyT-LoRA ISP
- AdaLoop-ISP
- RecurISP
- RecurISP-LoRA
- RecurISP-Adaptive
- IterISP
- LoopISP

최종 추천 이름은 다음과 같다.

```text
RecurISP
```

이유는 다음과 같다.

- 짧고 기억하기 쉽다.
- Recurrent ISP라는 핵심 구조가 바로 드러난다.
- 논문 제목, GitHub repo, 발표 자료에 모두 사용하기 좋다.
- LoRA, DyT-HDR, early-exit 등 세부 기술을 하위 모듈로 확장하기 쉽다.

추천 GitHub repository 이름:

```text
recurisp
```

---

## 3. 전체 아키텍처

### 3.1 High-level pipeline

```text
RAW Bayer / Linear RGB
        ↓
Lightweight ISP Stem
        ↓
CNN Local Feature Encoder
        ↓                         Condition metadata
RecurISP Refinement Block × T_max  ← sensor / ISO / gain / exposure / CCT / t
        ↓
ISP Reconstruction Head
        ↓
sRGB / YUV / Enhanced RGB
```

---

### 3.2 구성 요소

```text
RecurISP
 ├── ISP Stem
 │    ├── black level handling
 │    ├── shallow denoise
 │    ├── demosaic-aware convolution
 │    └── local feature extraction
 │
 ├── CNN Local Feature Encoder
 │    ├── depthwise convolution
 │    ├── pointwise projection
 │    └── residual local feature block
 │
 ├── Recurrent Refinement Core
 │    ├── shared DyT-HDR backbone
 │    ├── iteration-specific LoRA
 │    ├── condition encoder
 │    └── adaptive early-exit
 │
 └── ISP Reconstruction Head
      ├── color correction
      ├── tone/detail refinement
      └── RGB/YUV output projection
```

---

## 4. RecurISP Core

### 4.1 기본 반복 구조

입력 feature를 `F_0`라고 할 때, RecurISP는 동일한 shared block을 반복적으로 적용한다.

```text
F_0 = Encoder(x)

for t = 1 ... T_max:
    c_t = ConditionEncoder(sensor_id, ISO, gain, exposure, CCT, t)

    F_t = F_{t-1}
          + SharedDyTBackbone(F_{t-1})
          + LoRA_t(F_{t-1}, c_t)

    if EarlyExit(F_t, F_{t-1}, c_t):
        break

y = ISPHead(F_t)
```

---

### 4.2 핵심 설계 철학

기존 대형 Neural ISP:

```text
많은 layer와 많은 parameter로 다양한 조건을 한 번에 처리
```

RecurISP:

```text
작은 shared block을 반복 사용하고,
iteration별 LoRA와 condition scaling으로 필요한 refinement만 수행
```

즉, RecurISP는 다음 구조를 따른다.

```text
Shared backbone:
    공통 ISP prior 담당

Iteration-specific LoRA:
    반복 단계별 low-rank update 담당

Condition encoder:
    sensor / ISO / exposure / CCT / timestep 조건 반영

Early-exit:
    입력 난이도에 따른 adaptive compute 수행
```

---

## 5. Per-Iteration LoRA

### 5.1 왜 필요한가?

초기 구조에서는 같은 LoRA 또는 같은 Transformer block을 여러 번 반복하는 방식이었다.

이 경우 reviewer가 다음과 같이 질문할 수 있다.

> 같은 block을 T번 반복하는데, 왜 각 반복이 서로 다른 ISP 역할을 한다고 볼 수 있는가?

따라서 최종 구조에서는 **per-iteration LoRA**를 핵심 contribution으로 격상한다.

---

### 5.2 수식

기본 LoRA는 다음과 같이 표현된다.

```text
W_eff = W_shared + A B
```

RecurISP에서는 iteration마다 서로 다른 low-rank update를 사용한다.

```text
W_eff,t = W_shared + A_t B_t
```

condition-aware scaling을 추가하면 다음과 같다.

```text
W_eff,t,c = W_shared + s_t(c) · A_t B_t
```

여기서:

```text
t = iteration index
c = condition vector
A_t, B_t = iteration-specific low-rank matrices
s_t(c) = condition-dependent scaling factor
```

`s_t(c)`는 두 가지 설계가 가능하다.

```text
scalar scaling:
    s_t(c) ∈ R
    parameter와 runtime cost가 가장 작음

channel/vector scaling:
    s_t(c) ∈ R^d 또는 R^{d_out}
    sensor/ISO 조건별 표현력은 커지지만 fusion과 storage가 더 복잡함
```

초기 구현은 scalar 또는 per-output-channel vector 중 하나로 고정하고, ablation에서 둘을 비교한다.

---

### 5.3 핵심 claim

위험한 claim:

```text
t=1은 denoise, t=2는 texture, t=3은 color, t=4는 tone을 반드시 담당한다.
```

이는 보장되지 않는다.

최종 논문에서는 다음과 같이 안전하게 주장해야 한다.

```text
Primary claim:
In low-iteration recurrent neural ISP, per-iteration LoRA expands the expressive capacity of a shared backbone under a small parameter budget.

Secondary claim:
It may induce role specialization across ISP refinement steps, which we empirically analyze rather than assume.
```

즉, 역할 분화는 **주장**이 아니라 **실험적으로 관찰할 수 있는 현상**으로 다룬다.

---

### 5.4 Collapse 방지 regularization

per-iteration LoRA가 모두 비슷한 방향으로 수렴할 수 있다. 이를 막기 위해 다음 regularization을 추가할 수 있다.

#### 5.4.1 Orthogonality regularization

단순한 `||A_i^T A_j||_F²`는 `d × r` 행렬의 column 차원이 클 때 값이 자연스럽게 작아질 수 있다. 따라서 column-normalized basis 또는 subspace angle 기반 식을 권장한다.

```text
Q_t = orthonormal_basis(A_t)
L_ortho = Σ_{i≠j} || Q_i^T Q_j ||_F²
```

위 식은 principal angle의 `cos²` 합으로 해석할 수 있다. 논문에서는 아래 표현을 함께 쓰되, 별도 regularizer가 아니라 같은 목적의 해석 형태로 설명한다.

```text
L_angle = Σ_{i≠j} mean(cos² principal_angles(span(A_i), span(A_j)))
```

목적:

- iteration별 LoRA가 서로 다른 subspace를 학습하도록 유도
- 역할 분화 가능성 증가
- 단순 parameter duplication이 아니라는 근거 제공

---

#### 5.4.2 Residual magnitude balancing

```text
L_balance = Var( ||LoRA_t(x)||₂ across t )
```

목적:

- 특정 iteration에만 LoRA update가 몰리는 현상 방지
- 각 iteration이 일정 수준의 contribution을 갖도록 유도

---

#### 5.4.3 Rank sharing constraint

parameter 증가를 줄이기 위해 다음 구조도 고려할 수 있다.

```text
A_t = A_shared · diag(s_t)
```

장점:

- parameter overhead 감소
- memory reuse 증가
- hardware deployment에 유리

주의:

```text
rank sharing과 subspace orthogonality regularization은 동시에 쓰기 어렵다.
```

`A_t = A_shared · diag(s_t)`이면 모든 `A_t`가 같은 column space를 공유하므로, §5.4.1의 subspace orthogonality 목적과 충돌한다. 따라서 ablation은 다음처럼 mutually exclusive로 둔다.

```text
A: per-iteration A_t, B_t + no orthogonality
B: per-iteration A_t, B_t + subspace orthogonality
C: rank-shared A + per-iteration scaling
```

`L_balance`는 orthogonality와 별개의 axis로 둔다.

```text
regularization ablation:
    A/B/C 구조 각각에 대해 L_balance on/off 비교

minimum matrix:
    A
    A + L_balance
    B
    B + L_balance
    C
    C + L_balance
```

---

### 5.5 Parameter cost

per-iteration LoRA의 추가 parameter는 layer 하나에 LoRA를 붙일 때 다음과 같다.

```text
Params_LoRA = T_max × 2 × r × d
```

여기서:

```text
T_max = maximum iteration count
r     = LoRA rank
d     = hidden dimension
```

예를 들어:

```text
T_max = 4
r = 8
d = 256
```

이면:

```text
Params_LoRA = 4 × 2 × 8 × 256 = 16,384
```

하지만 실제 Transformer-style refinement block에서는 Q/K/V/O projection, FFN projection 등 여러 layer에 LoRA가 들어간다. 따라서 일반식은 다음처럼 써야 한다.

```text
Params_LoRA,total = T_max × Σ_{l∈L_lora} r_l × (d_in,l + d_out,l)
```

단순화해서 모든 LoRA target layer가 `d × d`이고 rank가 `r`이면:

```text
Params_LoRA,total ≈ T_max × N_lora_layers × 2 × r × d
```

예를 들어 `T_max=4`, `r=8`, `d=256`, `N_lora_layers=6`이면:

```text
Params_LoRA,total ≈ 4 × 6 × 2 × 8 × 256 = 98,304
```

이는 여전히 full non-shared Transformer block을 T개 두는 것보다 작을 수 있지만, 논문에서는 다음 표를 반드시 제시한다.

```text
shared recurrent block
shared recurrent block + shared LoRA
shared recurrent block + per-iteration LoRA
non-shared T-block Transformer
```

각 항목에 대해 parameter, trainable parameter, weight memory, activation memory, MAC, latency를 함께 보고한다.

필수 ablation:

```text
r ∈ {4, 8, 16, 32}
```

보고해야 할 항목:

- PSNR / SSIM / LPIPS
- ΔE / CIEDE2000
- parameter count
- latency
- SRAM access
- energy estimate

---

### 5.6 Prior art 비교 및 novelty boundary

Reviewer가 가장 먼저 비교할 prior art는 다음이다.

```text
LoRA:
    frozen backbone에 low-rank trainable update를 주입하는 기본 방법
    https://arxiv.org/abs/2106.09685

Universal Transformer / ALBERT / DEQ:
    weight sharing, recurrent-depth Transformer, implicit/infinite-depth 모델
    https://arxiv.org/abs/1807.03819
    https://arxiv.org/abs/1909.11942
    https://arxiv.org/abs/1909.01377

HyperNetworks / FiLM:
    condition에 따라 weight 또는 feature affine modulation을 생성
    https://arxiv.org/abs/1609.09106
    https://arxiv.org/abs/1709.07871

RAW/ISP/camera-domain LoRA 인접 사례:
    SpiralDiff / CamLoRA: RGB-to-RAW conversion에서 camera-specific ISP 특성 적응
    https://arxiv.org/abs/2603.14885
    검증: Codex web check on 2026-05-02.
    arXiv abstract에서 CamLoRA와 camera-specific ISP characteristics 적응을 명시.
    주의: 방향은 RGB-to-RAW이므로 direct RAW-to-RGB baseline이 아니라 opposite-direction adjacent prior art로만 사용.

    RDDM / CMB LoRA: RAW-domain diffusion restoration에서 Bayer pattern adaptation
    https://arxiv.org/abs/2508.19154
    검증: Codex web check on 2026-05-02.
    arXiv abstract에서 CMB LoRA와 RGGB/BGGR 등 Bayer pattern adaptation을 명시.

    Dr. RAW: RAW 기반 high-level vision에서 task-specific LoRA 사용
    https://openreview.net/forum?id=2Dwp12HmYI
    검증: Codex web check on 2026-05-02.
    OpenReview abstract에서 RAW task conditioning, SPP, LoRA를 명시.
```

최종 paper draft에서는 위 URL 메모를 BibTeX entry로 교체하고, 저자/제목/venue/year를 author가 한 번 더 확정한다.

따라서 논문에서는 다음을 주장하지 않는다.

```text
Per-iteration LoRA is the first use of LoRA for camera/RAW/ISP-related problems.
Conditioned low-rank adaptation is new in general.
Weight-shared recurrent Transformer is new in general.
```

대신 다음처럼 포지셔닝한다.

```text
RecurISP studies iteration-conditioned LoRA inside a low-iteration,
shared recurrent RAW-to-RGB neural ISP backbone, and evaluates whether
such updates improve ISP refinement quality, sensor adaptation, and
adaptive compute under mobile deployment constraints.
```

이 포지션에서 중요한 비교는 다음이다.

```text
vs Universal Transformer / DEQ:
    RecurISP는 일반 sequence modeling이 아니라 RAW/HDR ISP refinement에 맞춘
    finite low-iteration recurrent model이며, sensor/capture condition과 image quality metric을 중심으로 평가한다.

vs diffusion timestep-conditioned LoRA:
    RecurISP는 many-step generative denoising이 아니라 T≤4 정도의 deterministic ISP refinement이며,
    timestep LoRA의 목적도 generation style control이 아니라 RAW-to-RGB reconstruction quality와 device adaptation이다.

vs HyperNetwork / FiLM:
    RecurISP는 full weight generation이 아니라 deployable low-rank delta를 사용하고,
    offline-fused profile과 online condition scaling을 분리해 NPU 배포 가능성을 평가한다.

vs CamLoRA / CMB LoRA / Dr. RAW:
    이들은 camera/RAW domain에서 LoRA를 이미 사용한다.
    RecurISP의 차별점은 LoRA를 diffusion/VLM/high-level task adaptation이 아니라
    recurrent neural ISP refinement의 iteration axis에 배치한다는 점이다.
```

iteration axis의 차별점은 단순한 위치 차이가 아니라 다음 실험 질문으로 구체화한다.

```text
deterministic ISP refinement에서는 t가 generative diffusion timestep이 아니라
RAW-to-RGB correction이 누적되는 finite refinement step이다.

따라서 per-iteration LoRA가 실제로
    monotonic quality improvement,
    residual magnitude decay,
    frequency/color/detail correction 분리,
    collapse 없이 서로 다른 low-rank direction 사용
을 만드는지 shared LoRA와 직접 비교해야 한다.

이 점을 RecurISP의 핵심 empirical question으로 둔다.
```

논문 contribution은 "LoRA를 ISP에 처음 적용"이 아니라 다음처럼 잡는 것이 안전하다.

```text
First focused study of iteration-conditioned LoRA for shared recurrent neural ISP refinement.
```

---

## 6. DyT-HDR

### 6.1 기존 DyT

DyT는 LayerNorm/RMSNorm 등 reduction-based normalization layer를 대체하거나 보완하기 위해 제안된 element-wise nonlinear transform으로 볼 수 있다.

기본 형태:

```text
DyT(x) = tanh(αx)
```

장점:

- mean/variance reduction 없음
- rsqrt 없음
- element-wise 연산
- LUT 또는 polynomial approximation으로 구현 가능
- NPU Vector Engine에서 처리하기 쉬움

---

### 6.2 ISP에서의 문제

Camera ISP는 RAW/HDR 데이터를 다룬다.

이때 단순 tanh는 다음 문제를 가질 수 있다.

- high dynamic range 정보가 saturation될 수 있음
- extreme low-light 또는 highlight 영역에서 정보 손실 가능
- neural ISP는 LayerNorm보다 GroupNorm, InstanceNorm, no-norm을 자주 사용하므로 단순히 “LayerNorm보다 낫다”는 주장은 약함

---

### 6.3 DyT-HDR 제안

이를 해결하기 위해 residual linear path를 포함한 DyT-HDR를 사용한다.

```text
DyT-HDR(x) = x + λ · tanh(αx)
```

의미:

```text
x path:
    wide dynamic range 보존

tanh path:
    outlier suppression 및 feature stabilization

λ:
    nonlinear correction strength
```

이 구조는 normalization이라기보다 **residual-gated soft-saturating nonlinearity**로 포지셔닝하는 것이 안전하다.

중요한 표현:

```text
DyT-HDR is not a normalization layer in the strict statistical sense.
It is a reduction-free residual nonlinear stabilizer for RAW/HDR feature refinement.
```

따라서 논문에서는 "normalization을 대체했다"가 아니라 "reduction-free stabilizer로서 normalization/no-norm/activation family와 비교한다"고 쓴다.

---

### 6.4 비교해야 할 후보

DyT-HDR는 normalization family와 activation family를 모두 비교해야 한다.

```text
Normalization/no-norm family:
DyT-HDR vs LayerNorm
DyT-HDR vs GroupNorm
DyT-HDR vs InstanceNorm
DyT-HDR vs NoNorm
DyT-HDR vs Scale-only residual

Residual nonlinearity family:
DyT-HDR vs x + λ · GELU(x)
DyT-HDR vs x + λ · SiLU(x)
DyT-HDR vs x + λ · clamp(x)
DyT-HDR vs x + λ · softsign(x)
DyT-HDR vs learned soft-clipping
```

중요한 점은 다음이다.

> DyT-HDR가 항상 최선이라고 주장하지 않는다.
> ISP recurrent refinement block 내부에서 tanh 기반 residual nonlinear stabilizer가 HDR 정보를 보존하면서 outlier suppression에 유효한지 검증한다.

---

### 6.5 Stratified evaluation

DyT-HDR의 효과는 전체 평균 metric만으로 보면 약할 수 있다. 따라서 조건별로 나누어 평가해야 한다.

```text
low-light
high ISO
HDR scene
highlight recovery
skin tone region
banding-prone region
```

---

## 7. Condition Encoder

### 7.1 입력 condition

RecurISP는 sensor와 capture condition을 명시적으로 입력받는다.

```text
condition c =
{
    sensor_id,
    ISO,
    analog_gain,
    digital_gain,
    exposure_time,
    CCT,
    scene_type,
    timestep t
}
```

---

### 7.2 Encoding 방식

```text
sensor_id      → embedding lookup
ISO            → log-scale normalization
analog_gain    → normalized scalar
digital_gain   → normalized scalar
exposure_time  → log-scale scalar
CCT            → sinusoidal or normalized scalar encoding
timestep t     → timestep embedding
```

통합:

```text
c_t = MLP([sensor_emb, logISO, gain, exposure, CCT, t_emb])
```

---

### 7.3 Condition-aware LoRA scaling

condition vector는 LoRA scaling에 사용된다.

```text
s_t(c) = MLP_t(c)
```

`s_t(c)`의 형태는 명시해야 한다.

```text
scalar:
    W_eff,t,c = W_shared + s_t(c) · A_t B_t

vector:
    W_eff,t,c = W_shared + diag(s_t(c)) · A_t B_t
    또는
    y = xW_shared + (xA_tB_t) ⊙ s_t(c)
```

초기 논문 실험에서는 scalar scaling을 default로 둔다. vector scaling은 §11.2의 main ablation 중 하나로 비교하되, deployment 분석에서는 §9.1의 offline-fusion 제약을 함께 보고한다. 이후 별도 언급 없이 쓰는 `W_eff,t,c = W_shared + s_t(c) · A_t B_t`는 scalar scaling을 의미한다.

---

### 7.4 New sensor adaptation

RecurISP의 중요한 product engineering 장점은 다음이다.

```text
New sensor가 추가되면 backbone은 freeze하고,
sensor-specific LoRA만 fine-tune한다.
```

단, 이는 LoRA의 일반적인 활용 방식과 유사하므로 standalone contribution으로 과장하지 않는다. ISP 도메인에서 의미를 만들려면 다음을 보여야 한다.

```text
full fine-tuning 대비 catastrophic forgetting 감소
기존 sensor A/B 성능 유지
sensor C adaptation에 필요한 sample 수와 시간 감소
어떤 sensor/ISO/exposure 범위에서 LoRA-only adaptation이 충분한지 분석
```

실험 설계:

```text
Train:
    sensor A, sensor B

Test:
    sensor C

Adaptation:
    freeze backbone
    train LoRA for sensor C only
```

보고할 것:

- full fine-tuning 대비 품질
- 학습 parameter 수
- adaptation 시간
- sensor C에서의 ΔE, PSNR, LPIPS
- low-light/HDR generalization

---

## 8. Adaptive Early-Exit

### 8.1 동기

모든 이미지가 동일한 refinement depth를 필요로 하지 않는다.

예:

```text
Daylight / low ISO:
    1~2 iterations

Indoor / medium ISO:
    2~3 iterations

Night / high ISO:
    3~4 iterations

HDR / difficult scene:
    4 iterations
```

따라서 RecurISP는 condition-aware early-exit을 사용한다.

---

### 8.2 Residual energy 기반 exit

가장 단순한 방식:

```text
r_t = ||F_t - F_{t-1}||₂ / ||F_{t-1}||₂
```

exit condition:

```text
if r_t < τ(c):
    stop
```

여기서 `τ(c)`는 condition별 threshold이다.

---

### 8.3 Learned exit predictor

더 학습 가능한 형태:

```text
e_t = MLP(GlobalPool(F_t), c_t)
```

exit condition:

```text
if e_t < τ:
    stop
```

---

### 8.4 Training strategy

early-exit은 inference에서는 쉽지만, training에서는 문제가 생긴다.

선택지는 다음과 같다.

#### Option A: 모든 iteration에 supervision

```text
L = Σ_t w_t · L_t
```

장점:

- 학습 안정적
- 각 intermediate output이 의미를 갖게 됨

단점:

- 모든 iteration을 학습 중 계산해야 함

---

#### Option B: Expected loss over exits

```text
L = Σ_t p_t · L_t
```

여기서:

```text
p_t = learned halting probability
```

장점:

- exit 확률을 학습에 반영
- inference behavior와 training objective가 더 가까움

주의:

- `p_t = softmax(-energy_t)`처럼 residual energy만으로 exit probability를 만들면 품질과 계산량이 얽혀 collapse할 수 있다.
- 모든 입력이 너무 일찍 exit하거나, 특정 iteration에만 몰리는 현상을 막기 위한 compute regularization이 필요하다.
- inference의 hard threshold는 미분 불가능하므로 training surrogate를 명시해야 한다.

paper version에서 권장하는 형태:

```text
h_t = sigmoid(MLP([GlobalPool(F_t), c_t, t_emb]))
p_t = h_t · Π_{k<t}(1 - h_k)

L_quality = Σ_t p_t · L_t
```

regularization은 둘 중 하나를 기본으로 선택한다.

```text
PonderNet-style:
    L = L_quality + γ · KL(p_t || geometric prior)

ACT-style:
    L = L_quality + β · Σ_t p_t · t
```

`KL prior`와 `expected compute penalty`를 동시에 넣으면 둘 다 compute를 줄이는 방향의 gradient를 만들 수 있어 tuning이 불안정해질 수 있다. 따라서 paper main setting은 하나만 사용하고, 필요하면 다음을 ablation으로 둔다.

```text
KL only vs compute only vs KL + compute
```

이는 PonderNet/ACT 계열 halting probability에 가깝다.

참고 prior art:

```text
ACT: Adaptive Computation Time for Recurrent Neural Networks
https://arxiv.org/abs/1603.08983

PonderNet: Learning to Ponder
https://arxiv.org/abs/2107.05407

Universal Transformer의 dynamic per-position halting
https://arxiv.org/abs/1807.03819
```

대안:

```text
Gumbel-softmax / straight-through estimator:
    hard exit behavior를 training 중 근사

ACT-style halting:
    cumulative probability가 threshold를 넘으면 stop
```

---

#### Option C: Fixed training, dynamic inference

학습 중에는 항상 `T_max`까지 돌리고, inference에서만 early-exit을 적용한다.

장점:

- 구현이 가장 단순
- 초기 실험에 적합

단점:

- training/inference gap 존재

초기 연구 단계에서는 Option C로 시작할 수 있지만, paper version에서는 Option A + PonderNet/ACT-style Option B를 권장한다.

권장 개발 순서:

```text
v0:
    train fixed T_max
    inference only residual-energy exit
    quality-latency curve 확인

v1:
    모든 iteration output에 supervision
    learned halting head 추가

v2:
    expected loss + compute regularizer
    hard/soft exit gap 측정
```

---

### 8.5 Hardware 관점

Adaptive early-exit은 평균 latency와 energy를 줄일 수 있다.

하지만 dynamic iteration count는 batch inference에서 divergence를 유발할 수 있다.

단, mobile camera ISP는 일반적으로:

```text
batch size = 1
```

이므로 divergence 문제는 상대적으로 작다.

논문에서는 다음을 정직하게 명시해야 한다.

> Dynamic iteration reduces average latency but introduces control divergence.
> This is acceptable in mobile ISP deployment because the dominant use case is batch-1 real-time processing.

---

## 9. LoRA Deployment 정책

LoRA는 inference 시 처리 방식에 따라 두 가지 deployment mode가 있다.

---

### 9.1 Offline-fused LoRA

profile별로 미리 weight를 fuse한다.

```text
W_eff,t,profile = W_shared + s_t(profile) · A_t B_t
```

장점:

- runtime 추가 GEMM 없음
- NPU execution 단순
- profile별 최적화 가능
- deployment에 적합

단점:

- profile 수가 많으면 weight storage 증가
- profile switch 시 weight reload 필요
- continuous condition에 대해 매 입력마다 다른 `s_t(c)`가 나오면 offline fusion 불가

중요한 제약:

```text
offline fusion은 condition space를 discrete profile로 고정하거나
ISO/gain/exposure/CCT를 profile bucket으로 양자화한 경우에만 가능하다.

continuous scalar scaling:
    profile bucket으로 양자화하면 offline fusion 가능
    완전 연속 scaling이면 online 계산 필요

continuous vector scaling:
    입력마다 diag(s_t(c)) · A_tB_t가 달라지므로 online LoRA 또는 profile-bucket fusion 필요
```

적합한 경우:

```text
sensor-specific profile
fixed camera mode
discrete ISO/gain/exposure bucket
product deployment
```

---

### 9.2 Online LoRA

runtime에서 LoRA path를 따로 계산한다.

```text
y = xW_shared + xA_tB_t
```

장점:

- continuous ISO/gain adaptation 가능
- profile 조합 유연
- fine-grained condition control 가능

단점:

- 추가 GEMM 발생
- SRAM/BW/latency 증가
- NPU scheduling 복잡

적합한 경우:

```text
research prototype
continuous condition adaptation
dynamic ISP tuning
```

---

### 9.3 권장 정책

논문과 제품 관점에서는 다음이 좋다.

```text
Default:
    offline-fused LoRA

Optional:
    online LoRA for continuous condition experiments
```

---

## 10. Training Memory 문제

### 10.1 문제

RecurISP는 recurrent block을 `T_max`번 unroll한다.

따라서 training 시 activation memory가 증가한다.

```text
Training memory ≈ O(T_max × activation_size)
```

예:

```text
T_max = 4 → activation memory roughly 4×
```

---

### 10.2 대응 방법

사용 가능한 방법:

```text
gradient checkpointing
truncated BPTT
activation recomputation
mixed precision training
```

고급 옵션:

```text
implicit differentiation for fixed-point iteration
```

초기 구현에서는 gradient checkpointing을 우선 적용하는 것이 현실적이다.

단, checkpointing은 activation memory를 줄이는 대신 wall-clock training time을 늘린다.

초기 보고 항목:

```text
training memory peak:
    baseline vs T_max=2/3/4

training throughput:
    images/sec or sec/iter

checkpointing overhead:
    expected 1.3×~2.0× wall-clock depending on block size

LoRA overhead:
    parameter memory는 작지만 per-iteration target layer 수에 따라 optimizer state와 recomputation cost 증가
```

---

### 10.3 논문 framing

정직한 표현:

> RecurISP is designed primarily for inference efficiency.
> Training requires unrolled recurrent computation, and we mitigate the memory overhead using checkpointing.

---

## 11. 실험 설계

### 11.1 Baseline ladder

최종 ablation ladder:

```text
B0: CNN ISP baseline
B1: CNN + non-shared Transformer refinement
B2: CNN + shared looped Transformer
B3: B2 + timestep embedding
B4: B3 + shared LoRA
B5: B3 + per-iteration LoRA
B6: B5 + DyT-HDR
B7: B6 + adaptive early-exit
```

---

### 11.2 핵심 비교

```text
shared LoRA vs per-iteration LoRA
fixed T vs adaptive T
DyT-HDR vs GN/IN/NoNorm/Scale-only
DyT-HDR vs residual GELU/SiLU/soft-clipping
offline-fused LoRA vs online LoRA
rank sweep: r = 4, 8, 16, 32
T_max sweep: T = 1, 2, 3, 4
scalar condition scaling vs vector condition scaling
```

---

### 11.3 Dataset 후보

공개 데이터셋:

```text
Zurich RAW2RGB
MIT-Adobe FiveK
SIDD
SID
LOL / low-light enhancement datasets
```

내부 데이터가 있다면:

```text
S.LSI sensor RAW dataset
multi-sensor RAW-RGB paired dataset
low-light / HDR / skin-tone focused dataset
```

---

### 11.4 SOTA 비교 대상

비교 후보:

```text
PyNet
AWNet
MicroISP
LiteISP
Restormer-style ISP baseline
NAFNet-style ISP baseline
MIRNet-style ISP baseline
RMFA-Net
DiR / ISP degradation-independent representation
U-Net ISP
CNN ISP baseline
Transformer ISP baseline
ParamISP
SpiralDiff / CamLoRA
RDDM / CMB LoRA
Dr. RAW
```

비교 성격은 구분해야 한다.

```text
direct neural ISP baseline:
    PyNet, AWNet, MicroISP, LiteISP, RMFA-Net, U-Net ISP, Transformer ISP

camera-parameter / inverse ISP related:
    ParamISP

modern restoration backbones adapted to ISP:
    Restormer, NAFNet, MIRNet

camera ISP representation baseline:
    DiR

RAW/ISP/camera-domain LoRA adjacent work:
    SpiralDiff, RDDM, Dr. RAW
```

SpiralDiff, RDDM, Dr. RAW는 direct RAW-to-RGB neural ISP baseline이라기보다 "camera/RAW domain에서 LoRA가 이미 사용됨"을 보여주는 prior art로 둔다.

Restormer/NAFNet/MIRNet를 ISP baseline으로 쓸 때는 task adaptation 정책을 명시한다.

```text
input:
    packed RAW Bayer 4-channel 또는 demosaiced linear RGB

output:
    sRGB 3-channel

architecture adaptation:
    first projection과 reconstruction head를 RAW-to-RGB 입출력에 맞게 교체
    core restoration blocks는 가능한 원 구조 유지

training:
    RecurISP와 동일 paired RAW/sRGB data, loss, crop size, augmentation 사용

report:
    native restoration checkpoint를 직접 비교하지 않고,
    동일 ISP task로 재학습한 adapted baseline만 비교
```

참고 링크:

```text
Restormer: https://arxiv.org/abs/2111.09881
NAFNet: https://arxiv.org/abs/2204.04676
MIRNet: https://arxiv.org/abs/2003.06792
RMFA-Net: https://arxiv.org/abs/2406.11469
    검증: Codex web check on 2026-05-02.
    arXiv abstract에서 real RAW to RGB reconstruction 및 Mobile AI 2022 Learned Smartphone ISP Challenge 평가를 명시.

DiR: https://openaccess.thecvf.com/content/CVPR2024/html/Guo_Learning_Degradation-Independent_Representations_for_Camera_ISP_Pipelines_CVPR_2024_paper.html
    검증: Codex web check on 2026-05-02.
    CVF 페이지에서 저자 Yanhui Guo, Fangzhou Luo, Xiaolin Wu, CVPR 2024 pp.25774-25783 및 DiR contribution을 명시.
```

---

### 11.5 Image quality metric

일반 metric:

```text
PSNR
SSIM
LPIPS
```

ISP-specific metric:

```text
CIEDE2000 / ΔE
color constancy error
skin tone error
highlight recovery error
low-light noise metric
banding artifact score
demosaic artifact score
```

---

### 11.6 Skin tone metric

가능한 방법:

```text
ITA: Individual Typology Angle
CIEDE2000 on detected skin patches
ΔE on manually annotated skin regions
```

권장:

```text
skin detector → CIE Lab conversion → CIEDE2000
manual mask subset으로 detector reliability 확인
```

detector 후보:

```text
MediaPipe Face Mesh / face segmentation:
    얼굴 중심 skin patch 분석에 적합

semantic segmentation 기반 skin/person parsing:
    얼굴 외 피부 영역까지 포함 가능

ITA threshold:
    detector가 아니라 Lab 기반 diagnostic rule로 사용
    조도와 white balance bias가 크므로 단독 사용 금지
```

추가 reliability check:

```text
manual skin mask subset으로 detector precision/recall 확인
조도/인종/화이트밸런스별 detection bias 보고
skin metric은 전체 metric이 아니라 stratified diagnostic metric으로 사용
```

---

## 12. Per-Iteration LoRA 분석 실험

이 실험은 paper에서 매우 중요하다. 단, per-iteration LoRA만 시각화하면 "후반 iteration residual이 자연히 작아지는 수렴 현상"과 confound된다.

따라서 모든 분석은 다음 두 모델에 동일하게 적용한다.

```text
B4: shared LoRA
B5: per-iteration LoRA
```

핵심 질문:

```text
B5의 ΔF_t / ΔY_t / FFT pattern이 B4와 질적으로 다른가?
B5가 단순히 residual magnitude를 줄이는 것이 아니라 frequency/color/detail correction axis를 다르게 사용하는가?
```

이 분석은 전체 평균뿐 아니라 §6.5의 condition strata별로 반복한다.

```text
low-light
high ISO
HDR / highlight recovery
skin tone region
banding-prone region
```

목표는 "평균적으로 B5가 다르다"가 아니라, 어떤 capture condition에서 per-iteration LoRA가 shared LoRA와 다른 correction pattern을 만드는지 찾는 것이다.

### 12.1 ΔFeature visualization

```text
ΔF_t = F_t - F_{t-1}
```

각 iteration이 어떤 영역을 수정하는지 시각화한다.

---

### 12.2 ΔOutput visualization

```text
ΔY_t = Y_t - Y_{t-1}
```

이미지 공간에서 각 iteration의 변화량을 보여준다.

---

### 12.3 Frequency analysis

```text
FFT(ΔF_t)
```

분석 포인트:

```text
low frequency
mid frequency
high frequency
color channel sensitivity
edge/detail sensitivity
B4 vs B5 spectrum divergence
```

---

### 12.4 Weight norm analysis

```text
||A_t B_t||_F
```

또는:

```text
singular value spectrum of A_t B_t
```

분석 포인트:

- iteration별 rank utilization
- 특정 iteration collapse 여부
- update direction diversity
- B4 shared LoRA와 비교한 per-iteration update diversity

---

### 12.5 해석 전략

결과가 좋을 경우:

```text
Compared with shared LoRA, per-iteration LoRA induces meaningfully different correction patterns across refinement steps.
```

결과가 애매할 경우:

```text
Per-iteration LoRA improves expressive capacity under weight sharing,
even without explicit semantic specialization.
```

이 Plan B를 반드시 준비해야 한다.

---

## 13. Hardware Evaluation

### 13.1 Hardware claim의 위험

다음 주장은 실측 또는 시뮬레이션 없이 강하게 하면 위험하다.

```text
DyT-HDR is faster than normalization.
LoRA is NPU-friendly.
Looped block is efficient.
Early-exit reduces energy.
```

따라서 hardware contribution으로 주장하려면 반드시 정량 결과가 필요하다.

---

### 13.2 필요한 측정 항목

```text
latency (ms)
cycle count
energy (mJ)
SRAM access count
DRAM bandwidth
activation memory peak
weight memory
MAC count
Vector Engine utilization
Tensor Engine utilization
```

---

### 13.3 가능한 평가 환경

옵션 A: cycle-level simulator

```text
Py-V 기반 NPU simulator
DMA / SRAM / TE / VE / bus contention model 포함
```

옵션 B: 실제 NPU 측정

```text
S.LSI internal mobile NPU
commercial mobile NPU
Android NNAPI / vendor backend
```

옵션 C: proxy profiling

```text
TVM / TFLite / ONNXRuntime profiling
MAC + memory access analytical model
```

---

### 13.4 비교 대상

```text
CNN ISP baseline
Transformer ISP baseline
RecurISP fixed-T
RecurISP adaptive early-exit
RecurISP offline-fused LoRA
RecurISP online LoRA
```

---

## 14. 논문 Contribution 정리

최종 contribution은 다음과 같이 잡는 것이 안전하다.

```text
1. Iteration-conditioned low-rank adaptation for shared recurrent neural ISP refinement
2. DyT-HDR: reduction-free residual nonlinear stabilizer for RAW/HDR ISP features
3. Sensor/capture-conditioned halting for mobile RAW ISP refinement
```

3번은 ACT/PonderNet류 halting을 새로 제안한다는 뜻이 아니다. 차별점은 halting signal에 ISP-specific metadata를 넣고, RAW-to-RGB quality/latency trade-off에서 검증한다는 점이다.

```text
condition inputs:
    sensor_id, ISO, gain, exposure, CCT, timestep

evaluation:
    quality drop vs average iteration count
    low-light/high-ISO/HDR strata별 exit behavior
    mobile batch-1 latency/energy proxy
```

4번 후보는 결과가 확인된 뒤에만 contribution으로 올린다.

```text
If observed:
    Per-iteration LoRA produces correction patterns measurably distinct
    from shared LoRA in low-iteration ISP refinement.

If not observed:
    §12의 분석은 evidence/diagnostic으로만 두고 contribution에서는 제외한다.
```

Hardware-aware deployment은 실제 측정 또는 cycle-level simulation이 없으면 contribution이 아니라 discussion으로 둔다.

```text
Discussion:
    offline-fused vs online LoRA deployment
    expected SRAM/DRAM/latency trade-off
    mobile batch-1 early-exit behavior

Contribution으로 올릴 조건:
    real NPU measurement
    또는 cycle-level simulator
    또는 TVM/TFLite/ONNXRuntime + analytical memory model의 정량 결과
```

또한 다음 claim은 피한다.

```text
first LoRA for ISP
first sensor-aware LoRA
first timestep-conditioned low-rank adaptation
```

가능한 안전한 contribution 문장:

```text
We provide a focused empirical study of iteration-conditioned LoRA
inside a shared recurrent RAW-to-RGB neural ISP backbone, showing its
quality/parameter trade-off and correction behavior relative to shared LoRA.
```

---

## 15. 최종 논문 메시지

### 15.1 한 줄 정의

```text
RecurISP expands the expressive capacity of a shared ISP backbone via iteration-conditioned low-rank updates, while enabling condition-adaptive computation through early-exit.
```

한국어 표현:

```text
RecurISP는 shared recurrent ISP backbone에 iteration-conditioned LoRA를 결합하여,
작은 모델 크기로 반복적 ISP refinement를 수행하고,
입력 조건에 따라 필요한 만큼만 계산하는 mobile NPU-friendly neural ISP 구조이다.
```

Prior-art aware 표현:

```text
RecurISP does not claim low-rank adaptation or recurrent weight sharing as new in general.
It studies their combination on the iteration axis of RAW-to-RGB neural ISP refinement,
where low iteration count, sensor metadata, HDR preservation, and mobile deployment constraints
create a distinct evaluation setting.
```

---

### 15.2 안전한 주장

```text
RecurISP는 low-iteration recurrent neural ISP에서 per-iteration LoRA를 통해 weight-sharing backbone의 표현력과 parameter-quality trade-off를 개선한다.
```

---

### 15.3 강한 주장

실험에서 역할 분화가 보일 경우:

```text
Per-iteration LoRA induces emergent specialization across refinement steps.
```

---

### 15.4 피해야 할 주장

```text
각 iteration은 반드시 denoise, texture, color, tone 역할로 분화된다.
```

이 표현은 증거가 없으면 위험하다.

추가로 피해야 할 주장:

```text
LoRA를 ISP에 처음 적용했다.
DyT-HDR는 normalization layer다.
hardware 측정 없이 NPU-friendly라고 단정한다.
early-exit은 학습 없이 항상 품질 손실 없이 동작한다.
```

---

## 16. Venue별 포지셔닝

### 16.1 CVPR / ICCV / ECCV

강조점:

```text
image quality
novel architecture
per-iteration LoRA analysis
adaptive ISP refinement
multi-sensor generalization
```

필수:

```text
SOTA comparison
ablation
visual results
frequency / Δfeature analysis
```

---

### 16.2 ISSCC / JSSC / Hot Chips / MWSCAS

현실적 포지션:

```text
ISSCC / JSSC:
    실리콘 측정 또는 fully validated cycle-level simulator가 있을 때만 main target

Hot Chips / MWSCAS:
    NPU deployment analysis, simulator, proxy profiling까지 포함한 architecture/system story로 접근 가능

기본 논문 target:
    CVPR / ICCV / ECCV 또는 computational photography / low-level vision venue
```

강조점:

```text
measured latency
energy
SRAM traffic
NPU utilization
offline-fused LoRA deployment
adaptive early-exit
```

필수:

```text
real hardware or cycle-level simulator
memory traffic breakdown
fixed-T vs adaptive-T energy comparison
```

---

### 16.3 S.LSI 내부 전략 문서

강조점:

```text
multi-sensor ISP profile reuse
new sensor adaptation cost reduction
mobile NPU-friendly ISP block
customer-specific camera tuning
shared backbone + LoRA profile deployment
```

---

## 17. GitHub Repository 구조 추천

Repository 이름:

```text
recurisp
```

초기 폴더 구조:

```text
recurisp/
 ├── README.md
 ├── docs/
 │    ├── prd/
 │    │    └── recurisp_prd.md
 │    ├── spec/
 │    │    ├── architecture_spec.md
 │    │    ├── lora_spec.md
 │    │    ├── dythdr_spec.md
 │    │    ├── early_exit_spec.md
 │    │    └── condition_encoder_spec.md
 │    ├── experiments/
 │    │    ├── ablation_plan.md
 │    │    ├── dataset_plan.md
 │    │    └── hardware_eval_plan.md
 │    └── paper/
 │         └── outline.md
 │
 ├── recurisp/
 │    ├── models/
 │    │    ├── stem.py
 │    │    ├── encoder.py
 │    │    ├── recurrent_core.py
 │    │    ├── lora.py
 │    │    ├── dythdr.py
 │    │    ├── condition_encoder.py
 │    │    ├── early_exit.py
 │    │    └── head.py
 │    │
 │    ├── losses/
 │    │    ├── isp_loss.py
 │    │    ├── color_loss.py
 │    │    ├── perceptual_loss.py
 │    │    └── lora_regularization.py
 │    │
 │    ├── datasets/
 │    │    ├── raw2rgb.py
 │    │    ├── sidd.py
 │    │    └── fivek.py
 │    │
 │    ├── training/
 │    │    ├── train.py
 │    │    ├── train_lora.py
 │    │    └── finetune_sensor.py
 │    │
 │    ├── evaluation/
 │    │    ├── metrics.py
 │    │    ├── color_metrics.py
 │    │    ├── skin_tone.py
 │    │    └── frequency_analysis.py
 │    │
 │    └── hw/
 │         ├── cost_model.py
 │         ├── memory_model.py
 │         └── npu_profile.py
 │
 ├── configs/
 │    ├── recurisp_base.yaml
 │    ├── recurisp_lora.yaml
 │    ├── recurisp_dythdr.yaml
 │    └── recurisp_early_exit.yaml
 │
 ├── scripts/
 │    ├── train_base.sh
 │    ├── train_lora.sh
 │    ├── eval_quality.sh
 │    └── eval_hw_cost.sh
 │
 └── tests/
      ├── test_lora.py
      ├── test_dythdr.py
      ├── test_condition_encoder.py
      └── test_early_exit.py
```

---

## 18. README 초안

```markdown
# RecurISP

RecurISP is a recurrent neural ISP architecture that studies iteration-conditioned low-rank adaptation inside shared ISP refinement blocks, with DyT-HDR residual nonlinear stabilization and condition-adaptive halting for mobile camera ISP deployment.

## Key Ideas

- Shared recurrent ISP refinement backbone
- Per-iteration LoRA for expressive capacity under weight sharing
- DyT-HDR as a reduction-free residual nonlinearity for RAW/HDR feature stabilization
- Condition encoder for sensor, ISO, gain, exposure, and CCT adaptation
- Adaptive early-exit for input-dependent compute scaling
- Offline-fused and online LoRA deployment modes

## Architecture

RAW / Linear RGB → ISP Stem → CNN Encoder → RecurISP Core → ISP Head → sRGB / YUV

## Main Research Questions

1. Does per-iteration LoRA outperform shared LoRA in a recurrent neural ISP backbone?
2. Does DyT-HDR outperform comparable residual nonlinearities without harming HDR dynamic range?
3. Can learned halting reduce average latency while preserving RAW-to-RGB quality?
4. Under what sensor and exposure conditions is LoRA-only adaptation sufficient?

## Claim Boundary

RecurISP does not claim that LoRA, recurrent weight sharing, or condition modulation are new in general. The focus is their empirical behavior in low-iteration, sensor-conditioned RAW-to-RGB ISP refinement.
```

---

## 19. 개발 단계 제안

### Phase 0: Minimal baseline

```text
CNN ISP baseline
RAW-to-RGB paired dataset loader
PSNR / SSIM / LPIPS / ΔE evaluation
```

---

### Phase 1: Recurrent shared block

```text
shared recurrent refinement block
fixed T_max
timestep embedding
```

---

### Phase 2: Per-iteration LoRA

```text
A_t, B_t per iteration
rank sweep
shared LoRA vs per-iteration LoRA comparison
```

---

### Phase 3: DyT-HDR

```text
DyT-HDR module
GN/IN/NoNorm comparison
HDR/low-light stratified evaluation
```

---

### Phase 4: Adaptive early-exit

```text
residual energy criterion
learned exit predictor
fixed-T vs adaptive-T latency/quality curve
```

---

### Phase 5: Hardware model

```text
MAC count
activation memory
SRAM/DRAM traffic
offline-fused vs online LoRA cost
cycle-level NPU simulator integration
```

---

## 20. 최종 요약

RecurISP의 최종 설계 방향은 다음과 같다.

```text
기존 Neural ISP:
    큰 모델 하나가 모든 조건을 처리

RecurISP:
    작은 shared recurrent backbone이 반복 refinement를 수행
    iteration별 LoRA가 표현력을 확장
    condition encoder가 sensor/ISO/exposure/CCT를 반영
    DyT-HDR가 RAW/HDR feature를 안정화
    early-exit이 입력 난이도에 따라 계산량을 조절
```

가장 중요한 논문적 차별점은 다음이다.

```text
Focused study of per-iteration LoRA for shared recurrent neural ISP refinement
```

가장 중요한 실험은 다음이다.

```text
shared LoRA vs per-iteration LoRA
```

이 실험은 단순 품질 비교뿐 아니라 다음 분석까지 포함해야 한다.

```text
B4 shared LoRA vs B5 per-iteration LoRA
ΔF_t / ΔY_t / FFT / rank utilization 비교
sensor adaptation 및 scalar/vector scaling ablation
```

가장 중요한 리스크는 다음이다.

```text
role specialization이 명확히 보이지 않을 수 있음
```

이에 대한 안전한 대응은 다음이다.

```text
primary claim을 role specialization이 아니라
expressive capacity expansion으로 둔다.
```

최종적으로 RecurISP는 다음과 같은 방향의 연구로 정리된다.

> **RecurISP is a mobile NPU-oriented recurrent neural ISP framework that studies iteration-conditioned low-rank adaptation inside a shared RAW-to-RGB ISP backbone, using DyT-HDR residual nonlinear stabilization and condition-adaptive halting to support efficient RAW/HDR image refinement under diverse sensor and capture conditions.**
