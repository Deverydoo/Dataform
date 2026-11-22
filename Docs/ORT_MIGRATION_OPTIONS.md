# ORT Training: Migration Options

**Status:** analysis only — no decision made, no code changed.
**Written:** 2026-08-05

## The problem

DATAFORM's entire training tier — Phase 2 (LoRA adapters, ReflectionEngine) and
Phase 3 (population-based evolution) — sits on `Microsoft.ML.OnnxRuntime.Training`,
pinned at **1.19.2** in `third_party/onnxruntime-training/`.

That is the last stable release Microsoft ever published. Checking the NuGet
version index for the package returns:

```
1.15.0, 1.15.1, 1.16.0-dev, 1.16.0, 1.16.1, 1.16.2, 1.16.3,
1.17.0, 1.17.1, 1.17.3, 1.18.0, 1.19.1, 1.19.2, 1.20.0-dev-20240818
```

The list ends there. The newest artifact of any kind is a **dev build from
August 2024**. The on-device training package was not carried forward into the
1.20+ line. We are not behind on a moving target; the target stopped moving.

**This is not urgent.** 1.19.2 works, training runs only during idle windows, and
nothing breaks today. What we lose is optionality: no security patches, no new
ONNX opsets, and eventual friction as MSVC and CUDA versions drift away from what
a 2024 binary was built against. The risk is slow, not sharp.

## What we would need a replacement to do

Ranked by how much the current architecture depends on it:

1. **Small, swappable adapters.** Evolution trains a *population* of variants and
   selects among them. That only works because a LoRA adapter is 5–20 MB, not a
   full model copy. Full-weight finetuning breaks the design, not just the code.
2. **Weight merging.** `WeightMerger` averages checkpoints via ORT's
   `ToBuffer`/`FromBuffer`. A replacement needs equivalent parameter access.
3. **Run inside an idle window.** Start, make progress, pause on user activity,
   resume. Not "launch a 40-minute job and hope".
4. **Local-only, no account, no telemetry.** Non-negotiable given the product.
5. **Ideally no new heavyweight runtime** in what is currently a self-contained
   C++/Qt desktop app.

## Option 1 — Stay on 1.19.2

Do nothing. Revisit when something actually breaks.

**For:** zero work; the code exists and functions; training is idle-time only so a
future failure is degraded capability, not a crash on startup.

**Against:** the failure mode is unbounded in time — a Windows or driver update
could break it on a Tuesday with no upstream fix available. Nobody will be
maintaining familiarity with this code by then.

**Verdict:** correct for now, wrong as a permanent answer. If chosen, the honest
move is to write down the trigger that forces reconsideration (e.g. "first time
it fails to build or run on a supported machine").

## Option 2 — llama.cpp `ggml-opt`

The appealing one: llama.cpp is **already linked** as of this session's CUDA work,
and upstream has a real training API in its public headers —
`ggml_opt_dataset_init`, `ggml_opt_init`, `ggml_opt_epoch`, plus CUDA optimizer
kernels (`ggml/src/ggml-cuda/opt-step-adamw.cu`, `opt-step-sgd.cu`) and a worked
example at `examples/training/finetune.cpp`.

One dependency instead of two, same GGUF model files, CUDA already configured.

**But it does not currently fit.** I checked the actual capability rather than
assuming:

- **No LoRA.** The finetune example's entire option surface is `--learning-rate`,
  `-lr-min`, `-decay-epochs`, `--weight-decay`, `--val-split`, `--epochs`,
  `--optimizer sgd|adamw`. There is no rank or alpha parameter anywhere. This is
  full-weight finetuning. That kills requirement #1 — you cannot hold a population
  of full-weight 3B variants.
- **FP32 only.** Upstream's own README: *"finetuning is technically functional
  (for FP32 models and limited hardware setups)"* and *"the code is very much
  WIP"*. An FP32 1B model needs ~24 GB to train. Our base is 1.5B.
- The KV cache is forced to F32 during finetune for lack of F16 `OUT_PROD`.

**Verdict:** not viable today, and the blocker is architectural rather than
effort. Worth re-checking every few months — if ggml-opt gains LoRA and quantized
training, this becomes the obvious answer and the migration is contained.

## Option 3 — Python sidecar (transformers + PEFT)

Run training as a subprocess during idle windows: DATAFORM writes a JSONL of
training examples, invokes a Python script, waits for a `.safetensors` adapter,
and hot-swaps it exactly as it does now.

**For:** true LoRA with real rank/alpha control; by far the most mature and
best-documented path; adapters stay small so the evolution design survives intact;
merging is straightforward tensor arithmetic; the pause/resume story is a process
signal plus checkpoint. Optionally Unsloth for a large speedup on a 4090.

**Against:** introduces a Python runtime as a *hard* dependency of a core feature.
Note the project already requires Python offline for `tools/generate_artifacts.py`
— so this raises an existing build-time dependency to a runtime one rather than
inventing a new one. Ships a torch+CUDA install measured in gigabytes. Process
orchestration, error surfacing, and version pinning all become our problem.

**Verdict:** the only option that preserves the current architecture without
compromise. The cost is honest and large: DATAFORM stops being a single
self-contained binary.

## Option 4 — candle (Rust)

Hugging Face's Rust ML framework. Has LoRA support and compiles to a single
binary with no runtime.

**For:** keeps the "one executable" property; no Python; good CUDA support.

**Against:** adds a Rust toolchain to a C++/Qt build; smaller ecosystem than
PEFT, so more of the training loop is ours to write and debug; fewer worked
examples for the specific "LoRA on a small local model, resumable" case.

**Verdict:** the interesting dark horse. Right choice if keeping DATAFORM
dependency-free matters more than development speed. Needs a spike before it can
be judged properly — I have not verified candle's current LoRA maturity.

## Recommendation

**Short term:** Option 1. Nothing is broken, and the two changes shipped in this
audit (CUDA offload, privacy gating) matter more to daily use than a training
backend swap.

**Watch:** Option 2. It is the only path that makes the codebase *simpler* rather
than more complex. Re-check whether ggml-opt has gained LoRA and quantized
training; that single upstream change would flip the recommendation.

**If forced to move now:** Option 3, accepting the Python dependency, because it
is the only one that keeps small swappable adapters — and without those, Phase 3
evolution has to be redesigned rather than ported.

## Open questions

- Does candle's LoRA support cover our case? Unverified — needs a spike.
- Could evolution work with something cheaper than LoRA (e.g. prompt/prefix
  tuning, or a learned steering vector)? That would widen the options
  considerably and is worth thinking about independently of the backend choice.
- Is the 1.5B base model itself still the right target given the catalog now
  offers 3B–24B GGUF models?
