DepthMem v6

Budgeted Loop-State Memory Allocation for Hardware-Aware Iterative ISP

MLSys-Ready Research Proposal

⸻

0. Executive Summary

One-line thesis

DepthMem formulates loop-state memory in iterative ISP as a rate-constrained allocation problem: given a fixed persistent memory budget, determine what information should be preserved across refinement iterations, at which scope, frequency, precision, and lifetime.

Unlike prior iterative restoration methods that assume memory exists, DepthMem studies:

1. Whether iterative restoration is genuinely non-Markovian.
2. How useful past refinement states are.
3. How limited SRAM/DRAM budgets should be allocated across different memory types.
4. How memory lifetime maps onto hardware residency.

The central hypothesis is no longer merely that loop states exhibit scope-frequency structure.

Instead:

Iterative restoration contains useful non-Markovian information that cannot be fully represented by the current hidden state alone, and this information can be stored more efficiently through structured memory allocation than through full-state retention.

⸻

1. Problem Statement

1.1 Existing iterative ISP

Most iterative restoration systems follow:

h0 → h1 → h2 → ... → hT

where

h_{t+1} = Fθ(ht)

The network repeatedly refines a hidden representation.

Examples include:

* NAFNet variants
* Restormer variants
* DGUNet
* SRFBN
* Deep unfolding networks
* iterative denoisers

⸻

1.2 The hidden assumption

Nearly all iterative systems implicitly assume:

ht is sufficient.

Meaning:

P(y | h0...ht)
=
P(y | ht)

This is a Markov assumption.

If true:

* storing past states is unnecessary
* memory systems are useless

If false:

* past refinement states contain additional information
* memory becomes meaningful

⸻

1.3 Core research question

Before proposing memory architectures we ask:

How non-Markovian is iterative image restoration?

More precisely:

Does access to compressed past states
improve reconstruction quality
beyond the current hidden state?

If the answer is no:

DepthMem should not exist.

This question becomes the foundation of the paper.

⸻

2. Key Hypothesis

DepthMem tests two hypotheses.

H1: Non-Markovianity

Past loop states contain information useful for future refinement.

h0 ... h_{t-1}

provide information not fully contained in:

ht

⸻

H2: Structured Memory Utility

Useful information is not uniformly distributed.

Instead it varies along:

Scope
Frequency
Precision sensitivity
Temporal lifetime

Therefore memory allocation should not be uniform.

⸻

3. Memory Utility Probe (New Core Contribution)

3.1 Motivation

Previous probing only asked:

Can ht be compressed?

This does not answer whether memory is useful.

Instead we ask:

How useful are previous states?

⸻

3.2 Oracle Memory Probe

At iteration t:

Baseline:

ht → Fθ → h_{t+1}

Oracle:

[ht , Mpast]
→ Fθ
→ h_{t+1}

where

Mpast
=
compressed(h0...h_{t-2})

⸻

3.3 Memory Utility Score

Define:

MUS
=
PSNR(Oracle)
-
PSNR(Baseline)

or

MUS
=
LPIPS(Baseline)
-
LPIPS(Oracle)

Interpretation:

MUS ≈ 0
    Markov
MUS > 0
    Non-Markov

⸻

3.4 Memory Utility Curve

Measure:

MUS(B)

for memory budgets:

64 KB
128 KB
256 KB
512 KB
1 MB

This becomes a key figure.

⸻

3.5 Contribution

The probe itself becomes publishable.

It measures:

Non-Markovianity of iterative restoration.

Independent of DepthMem.

⸻

4. Related Work

4.1 Iterative Restoration Memory

MemNet

Persistent memory network for image restoration.

Long-term memory across recursive units.

SRFBN

Feedback state propagation across iterations.

Difference

These works ask:

How should memory be used?

DepthMem asks:

Given B bytes,
what memory should be stored?

⸻

4.2 Deep Unfolding

Examples:

* DGUNet
* USRNet
* ADMM unfolding
* HQS unfolding

These preserve:

dual variables
priors
auxiliary states

between iterations.

DepthMem does not replace these.

Instead it studies:

memory allocation
under hardware constraints

⸻

4.3 Frequency Decomposition

Examples:

* DeepRFT
* AdaIR
* SFNet
* frequency restoration models

Frequency decomposition is not new.

DepthMem uses frequency only as a memory allocation axis.

⸻

4.4 Octave Convolution

Low-frequency features stored at lower resolution.

DepthMem generalizes this concept into:

memory budget allocation

rather than feature processing.

⸻

4.5 Noise Conditioning

Examples:

* FFDNet
* CBDNet

Global noise information modulates restoration.

DepthMem adopts similar ideas only as readout mechanisms.

⸻

4.6 Sequence Memory

Examples:

* Mamba
* SSM
* Universal Transformer

These operate on:

token-time

DepthMem operates on:

loop-depth

⸻

5. Budgeted Loop-State Allocation (Main Novelty)

5.1 Reformulation

Instead of proposing:

4-bank memory

we formulate:

Given memory budget B,
allocate memory optimally.

⸻

5.2 Allocation Variables

Each memory bank has:

resolution
channels
precision
lifetime

Example:

Global-Low:
    32 channels
    FP16
Local-High:
    8 channels
    INT8
etc.

⸻

5.3 Objective

min
    Ltask
    + λ Bytes

where

Bytes
=
Σ memory banks

This is a rate-distortion problem.

⸻

5.4 Continuous Relaxation

Allocation parameters become learnable.

Examples:

channel count
precision
resolution

can be relaxed through:

Gumbel-softmax
soft masks

⸻

5.5 Key Difference

Previous methods:

Memory exists.

DepthMem:

Memory budget is optimized.

⸻

6. DepthMem Instantiation

The 4-bank design is now an instantiation.

Not the contribution itself.

⸻

6.1 Scope-Frequency Banks

Scope	Low Freq	High Freq
Global	Global-Low	Global-High
Local	Local-Low	Local-High

⸻

6.2 Semantics

Global-Low

Stores:

illumination
color cast
global exposure

⸻

Global-High

Stores:

noise statistics
texture statistics

⸻

Local-Low

Stores:

local shading
illumination variation

⸻

Local-High

Stores:

edges
textures
fine details

⸻

7. Compressors

Global-Low

GlobalAvgPool
→ Linear

⸻

Global-High

channel mean
channel variance

⸻

Local-Low

tile pooling

⸻

Local-High

high-pass residual

with adaptive precision.

⸻

8. Memory Lifetime

EMA Update

M_{t+1}
=
(1-β)Mt
+
βCt

⸻

Important Revision

β is NOT a stability parameter.

β controls:

information timescale
memory lifetime

only.

⸻

Interpretation

Small β:

long memory

Large β:

short memory

⸻

9. Hardware Residency Policy

This becomes a new contribution.

⸻

Residency Mapping

Inputs:

memory size
β
access frequency

Outputs:

register
SRAM
local buffer
DRAM

assignment.

⸻

Example

Global-Low
    persistent SRAM
Local-High
    streaming buffer
Global-High
    register

⸻

Optional Extension

Greedy allocator.

Or ILP solver.

⸻

10. Stability

Updated Bound

Remove β.

Use:

||JF||
+
Σ |γiσ(ai)|
  σmax(Ri)
  σmax(Ci)
<
1

⸻

Enforcement

1. Small γ initialization
2. Bounded gates
3. Spectral normalization
4. Weight clipping
5. Readout monitoring

⸻

11. Cost Model

Persistent Memory

Define:

B
=
persistent state bytes

⸻

Compare

Naive Iterative

B = 0

⸻

Full-State Oracle

B = T × |h|

⸻

DepthMem

B << T × |h|

⸻

Key Claim

DepthMem does NOT reduce activation traffic.

Instead:

It improves information preservation under constrained persistent memory budgets.

⸻

12. Experiments

Phase 0

Memory Utility Probe.

Most important experiment.

⸻

Phase 1

Baselines:

Naive iterative
MemNet-style memory
SRFBN-style feedback
Full-state oracle
DepthMem

⸻

Phase 2

Allocation ablations.

64 KB
128 KB
256 KB
512 KB

⸻

Phase 3

4-bank ablations.

Scope-only
Frequency-only
4-bank

⸻

13. Hero Figure

X-axis:

Persistent memory bytes

Y-axis:

PSNR

Curves:

Naive
MemNet
SRFBN
Full-state oracle
DepthMem

This becomes the central figure.

⸻

14. Failure Modes

Case 1

MUS ≈ 0

No non-Markovianity.

DepthMem invalid.

⸻

Case 2

T ≤ 2

No memory advantage.

⸻

Case 3

Weak noise.

Single-pass sufficient.

⸻

Case 4

Large models.

Capacity dominates memory.

⸻

15. Final Contributions

Contribution 1

Memory Utility Probe.

A diagnostic tool for measuring non-Markovianity in iterative restoration.

⸻

Contribution 2

Budgeted Loop-State Allocation.

A rate-constrained formulation of persistent memory.

⸻

Contribution 3

DepthMem.

A scope-frequency memory allocation instantiation.

⸻

Contribution 4 (Optional)

Hardware residency mapping.

Memory lifetime to SRAM/DRAM placement.

⸻

16. Final Thesis

DepthMem is not a paper about memory banks.

It is a paper about:

How much information from previous refinement steps is actually useful, and how that information should be allocated under a fixed hardware memory budget.

The 4-bank architecture is merely one realization of this broader formulation.
