v5는 구조적으로 잘 정리됐습니다. Decision tree, iso-comparison 3종, 정직한 §5.1 회계는 리뷰어 방어가 잘 된 상태예요. 그런데 기술적 모순 3개와 노벨티를 직접 위협하는 prior art 누락이 있습니다. 후자가 더 심각합니다.

1. 내부 모순 / 기술적 오류

§5.3 roofline 주장이 §5.1과 모순됩니다. AI_eff = FLOPs / (W + A + M)인데, DepthMem은 naive looped 대비 A는 동일하고 M과 compressor/readout FLOPs를 추가합니다. 따라서 DepthMem의 AI는 naive looped보다 낮거나 같지, “highest”가 될 수 없습니다. DepthMem이 이기는 상대는 full checkpoint뿐입니다. §5.1에서 이미 정직하게 인정한 것을 §5.3에서 다시 과장하고 있어요. Roofline plot의 메시지는 “DepthMem ≈ naive looped의 AI를 유지하면서 checkpoint의 정보 보존 이득을 얻는다”로 수정해야 합니다.

§4.5 stability bound의 β 항이 공허합니다. β ∈ (0,1)에서 β / (1 − |1−β|) = β/β = 1입니다. EMA의 DC gain은 β와 무관하게 항상 1이므로 (M = Σ β(1−β)^k C(h), 합 = 1), β는 worst-case contraction에 전혀 기여하지 않고 timescale만 바꿉니다. 즉 “Local-High은 β=0.9로 short lifetime 강제”(enforcement 4번)는 안정성 메커니즘이 아닙니다. Bound는 ||J_F||₂ + Σ|γ·σ(a)|·σ_max(R)·σ_max(C) < 1로 줄이고, β는 안정성이 아니라 정보 시간축 정책으로만 서술하세요. 리뷰어가 이 계산을 하면 수식 전체의 신뢰가 깎입니다.

§3.5 compressibility probe가 측정 대상을 혼동합니다. h_t를 압축해서 다음 loop에 주입하는 건 forward path의 압축 민감도를 재는 것이고, DepthMem이 실제로 하는 건 h_t는 그대로 두고 side memory를 추가하는 겁니다. 더 치명적인 누락: h_t가 Markov-sufficient하면 (즉 h_{t-1} 이전 정보가 h_t에 이미 다 담겨 있으면) inhomogeneity가 아무리 강해도 DepthMem은 무용합니다. 이게 진짜 load-bearing 가설인데 probe가 없어요. “Memory utility probe”를 추가하세요: loop t에서 과거 state {h_0..h_{t-2}}의 (압축된) oracle access를 줬을 때 ΔPSNR. 이게 0이면 §3.6 decision tree 어느 가지를 타든 폐기입니다.

Full checkpoint baseline의 소비 메커니즘이 미정의입니다. T×|h|를 저장한다는데 그걸 어떻게 다음 loop가 읽는지(concat? attention? dense connection?)에 따라 quality 상한이 완전히 달라집니다. MemNet-style dense concat으로 명시하는 게 자연스럽습니다.

LOL은 test 15장입니다. 0.1~0.3dB 차이를 다투는 실험에서 15장 PSNR은 noise입니다. LOL-v2-real (test 100장) 권장. Probing용으로는 LOL도 충분.

2. 노벨티 위협 — 이 문서가 인용하지 않은 직격 prior art

이게 핵심 문제입니다. Related work이 LLM/Mamba/UT 쪽만 보고 restoration 분야 내부의 직접 선행연구를 비웠습니다.

	1.	MemNet (Tai et al., ICCV 2017) — “persistent memory network for image restoration.” Recursive unit + recursion depth를 가로지르는 gated long-term memory. “Loop-depth memory in iterative restoration”이라는 문장 자체가 이미 존재합니다. 이걸 인용 없이 내면 desk-level 반려감입니다. SRFBN (CVPR 2019, feedback state across iterations)도 동일 계열.
	2.	Deep unfolding 계열 (DGUNet, USRNet, unfolded ADMM/HQS) — auxiliary variable(dual variable, denoiser prior state)을 iteration 간에 유지합니다. 즉 “반복 사이에 무엇을 기억할 것인가”에 대한 principled한 답이 이미 한 갈래 있습니다.
	3.	FFDNet/CBDNet noise map conditioning — §4.3의 “Global-High이 Local-High readout 강도를 noise-level controller로 제어”는 noise map conditioning의 변형으로 읽힙니다.
	4.	Octave Convolution — low-frequency feature를 저해상도로 저장해 메모리/연산을 줄이는 scope-frequency-resolution coupling. Local-Low tile pooling이 정확히 이 패턴입니다.
	5.	Frequency-decomposed restoration (FSNet/SFNet, DeepRFT, AdaIR 등) — frequency 축 분해 자체는 포화 상태.

즉 현재 구성요소를 분해하면: loop-depth memory(MemNet) + frequency 분해(FSNet 계열) + 저해상도 low-freq 저장(Octave) + noise-conditioned readout(FFDNet) + EMA gate. 조합 novelty만 남고, 조합 novelty는 main venue에서 약합니다.

3. 노벨티 확보 제안

살아남는 차별점은 두 개입니다: (a) byte budget이라는 제약, (b) hardware lifetime↔residency 매핑. MemNet도 unfolding도 frequency net도 “persistent state를 몇 바이트로 유지할 것인가”를 묻지 않습니다. 이걸 contribution의 중심으로 끌어올리는 걸 제안합니다.

제안 1 — Method를 “4-bank”가 아니라 “budgeted allocation”으로 재정의 (가장 권장). 핵심 질문을 “주어진 persistent budget B bytes에서 무엇을 보존할 것인가”로 바꾸고, bank별 (resolution, channel 수, precision) 할당을 학습 가능한 rate–distortion 문제로 정식화하세요. L = L1 + λ·Bytes(allocation)은 이미 §6.5에 씨앗이 있습니다. Gumbel-softmax나 continuous relaxation으로 allocation을 학습하면, hero figure의 Pareto curve를 hand-designed 점들이 아니라 단일 allocator가 λ sweep으로 그려내는 frontier로 만들 수 있습니다. 이러면 4-bank는 method가 아니라 allocator의 search space가 되고, MemNet(“memory가 존재한다”)과 frequency net(“분해가 존재한다”) 어느 쪽과도 정면충돌하지 않습니다. “Rate-constrained loop-state memory”라는 framing은 제가 아는 한 restoration에서 비어 있습니다.

제안 2 — Memory utility probe를 contribution으로 격상. “Iterative restoration의 non-Markovness 정량화” — 과거 loop 정보가 현재 state 너머로 얼마나 유용한가를 측정하는 metric. §3.7의 SIS/SD/RCR보다 이게 더 본질적인 진단이고, MemNet조차 “왜 memory가 필요한가”를 측정한 적이 없습니다. Positive면 DepthMem의 동기, negative여도 그 자체로 publishable한 관찰입니다.

제안 3 — β↔residency 매핑을 비유에서 알고리즘으로. 현재 “small β → SRAM”은 직관 서술입니다. 학습된 β와 bank 크기를 입력으로 받아 memory-level assignment를 산출하는 작은 ILP/greedy mapper를 정의하고, SCALE-Sim/Timeloop으로 cycle/energy까지 내면 §5.4의 “선택적 격상”이 MLSys main의 필수 요건이 됩니다. 이게 Son의 SoC 배경이 타인이 복제 못 하는 유일한 부분이고, 특허 각도로도 architecture보다 lifetime policy → memory hierarchy mapping 절차가 더 깨끗한 claim입니다.

제안 4 — Related work §2에 “iterative restoration memory” 절 신설. MemNet/SRFBN/deep unfolding을 정면으로 인용하고 한 문장으로 가르세요: “이들은 memory의 존재를 제안했고, 우리는 hardware budget 하에서 memory의 내용·정밀도·수명 할당을 다룬다.” Phase 1 baseline에 MemNet-style dense memory (parameter-matched)를 추가하면 이 구분이 실험으로 증명됩니다.

우선순위: 제안 4는 무조건(생존 요건), 제안 1이 main novelty, 제안 2는 Phase 0에 ~10h 추가로 들어가니 같이, 제안 3은 MLSys 노릴 때만. 제안 1을 채택하면 §6.1 hero figure와 §6.3 run 수는 거의 그대로 유지되면서 (allocator 학습 run 2~3개 추가) contribution 문장이 “4-bank 구조 제안”에서 “loop-state memory의 budget-optimal allocation 문제 정식화 + 해법”으로 바뀝니다. 후자가 main venue 언어입니다.
