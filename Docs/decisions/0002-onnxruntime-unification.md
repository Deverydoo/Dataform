# One ONNX Runtime: build 1.30 from source with training APIs

Status: accepted
Date: 2026-09-12
Supersedes the energy-VAD workaround in `Docs/decisions/0001-voice-loop.md`
Related: `Docs/ORT_MIGRATION_OPTIONS.md`

Written for: anyone who has to rebuild this dependency or wonders why it is not
just a NuGet package.

## Outcome

Done and verified: one ONNX Runtime **1.30.0** (`ORT_API_VERSION` 19 → 30) now
serves the whole process. Both build configurations compile and link with **no
Dataform source changes**, the app runs, Muninn loads, and the training API is
confirmed live (see *Verifying a build* below). The old 1.19.2 package is kept at
outside the repository, next to this checkout, in case a rollback is ever
needed.

Silero neural VAD is now actually running in the streaming path. Reaching it
needed three fixes in Muninn, one of which was a pre-existing hang in `stop()`
that would have frozen every voice turn — see *Closing the gap* below.

Loki Studio's VAD failure had the same symptom but a different cause, in Loki's
own code rather than Muninn's. It is fixed too — see the last section.

## The problem

Only one `onnxruntime.dll` can be loaded per process, and two components need it:

- **Muninn**, for neural VAD (Silero, TEN) and diarization. Built against ORT
  1.23, so it calls `OrtGetApiBase()->GetApi(23)`.
- **Dataform's training tier** — `OrtTrainingManager`, `WeightMerger`,
  `OrtInferenceManager` — which needs the on-device training API.

The ORT C API is backward compatible but not forward compatible: a runtime of
API version *N* satisfies any request for *≤ N* and returns **null** above it.
The shipped training package is 1.19 (API 19), so with it in place Muninn's
`GetApi(23)` returns null and neural VAD dies at the first utterance. That is why
`0001-voice-loop.md` settled for energy VAD, which never touches ONNX.

## Why the obvious fixes do not work

On-device training was abandoned upstream. Every distribution channel stops at
the same place:

| Channel | Training build | Plain ORT |
| --- | --- | --- |
| NuGet `Microsoft.ML.OnnxRuntime.Training` | **1.19.2**, then one dev build (Aug 2024) | — |
| GitHub release assets | `onnxruntime-training-win-x64` at 1.19.2; **absent** from 1.23.0 and 1.30.0 | present |
| PyPI `onnxruntime-training` | 1.19.2, Linux wheels only | current |

So there is no package to upgrade to. Downgrading instead — rebuilding Muninn
against 1.19 — would unify the version at the cost of standing still on an
abandoned 2024 binary, and Muninn is shared with Loki Studio, so changing what it
links has blast radius beyond this repo.

## What was done

The *source* still supports on-device training even though no binary ships:
`onnxruntime_ENABLE_TRAINING_APIS` is live in `cmake/CMakeLists.txt` and wired
into dependent options, and every API this project uses is present in v1.30.0
with unchanged signatures — `LoadCheckpoint`, `SaveCheckpoint`, `TrainStep`,
`OptimizerStep`, `LazyResetGrad`, `ExportModelForInferencing`,
`ToBuffer(bool)`, `FromBuffer`. No deprecation markers.

So ORT v1.30.0 is built from source with training APIs enabled, and the result
replaces `third_party/onnxruntime-training/`. API version goes 19 → **30**, which
covers Muninn's 23 with room to spare. That removes the *runtime* obstacle to
neural VAD — though it turns out not to be the only one; see *Closing the gap*.

Muninn is **not** rebuilt. It is compiled for API 23 and a 1.30 runtime serves
that request fine, so Loki Studio's shipped pairing is left alone.

### CPU-only, deliberately

The build passes no `--use_cuda`. Checked rather than assumed:

- Dataform appends **no execution provider anywhere** —
  `orttrainingmanager.cpp`, `ortinferencemanager.cpp` and `weightmerger.cpp` all
  construct a bare `Ort::SessionOptions`. The training tier was already CPU-only,
  so a CPU build is not a regression.
- Muninn's transcription runs on **CTranslate2**, which has its own CUDA and does
  not go through ORT at all. Only its VAD and diarization use ORT.
- `silero_vad.cpp` appends CUDA but wraps session creation in a try/catch that
  explicitly retries on CPU when the ORT CUDA DLLs are missing — our exact case.
  Silero is a ~1.8 MB model used for gating; CPU is comfortably fast enough.

This cut the build from hours to well under one and removed any cuDNN/CUDA
version matching. If the training tier later wants GPU, rebuild adding
`--use_cuda` — nothing in Dataform has to change.

### Reproducing the build

```
git clone --depth 1 --branch v1.30.0 --recurse-submodules --shallow-submodules \
    https://github.com/microsoft/onnxruntime.git
cd onnxruntime
<python> tools/ci_build/build.py --build_dir build/Windows --config Release \
    --build_shared_lib --parallel --enable_training_apis --skip_tests \
    --skip_submodule_sync --cmake_generator "Visual Studio 17 2022" \
    --compile_no_warning_as_error
```

`<python>` is the full path to a Python 3.12 interpreter. On the development
machine Python is not on PATH, and the bare `python` command is a Microsoft
Store stub that fails.

Two flags matter and are easy to confuse:

- `--enable_training_apis` — the on-device training C/C++ API. What we want.
- `--enable_training` — the full ORTModule path, which drags in PyTorch. Not
  wanted, and the usual cause of failed "training build" reports.

Passing `--enable_training_apis` leaves `ENABLE_TRAINING_OPS` looking `OFF` in
the logged CMake command line, which is alarming but fine: `cmake/CMakeLists.txt`
forces it back `ON` inside `if (onnxruntime_ENABLE_TRAINING_APIS)`, so the
gradient kernels `TrainStep` needs do get built.

Artifacts to copy into `third_party/onnxruntime-training/`:

| Destination | Source (under the ORT checkout) |
| --- | --- |
| `bin/onnxruntime.dll` | `build/Windows/Release/Release/onnxruntime.dll` |
| `bin/onnxruntime_providers_shared.dll` | same directory |
| `lib/onnxruntime.lib` | same directory (~2 KB is correct — ORT exports two symbols) |
| `include/` (flat) | all `*.h` and `*.inc` from `include/onnxruntime/core/session/`, plus `core/providers/cpu/cpu_provider_factory.h`, `core/framework/provider_options.h`, and `orttraining/orttraining/training_api/include/*.h` |

Copy the session directory **wholesale** rather than by a hand-written list. 1.30
split the public API into more files than 1.19 shipped — `onnxruntime_c_api.h`
now includes `onnxruntime_error_code.h` and `onnxruntime_ep_c_api.h`, neither of
which existed before — so an enumerated list silently goes stale and fails with
`C1083: Cannot open include file`. The flat include ends up at 18 files, not the
12 the NuGet package had.

Four headers in that directory must be **excluded**: `environment.h`,
`experimental_onnxruntime_cxx_api.h`, `experimental_onnxruntime_cxx_inline.h` and
`onnxruntime_experimental_cxx_api.h`. They reach back into internal `core/...`
paths that only resolve inside the ORT source tree.

## Verifying a build

A self-built runtime can compile, link and start while its training API is
missing — `GetTrainingApi` returning null would only bite during an idle-time
training run, hours later. `tools/ort_training_probe.cpp` checks it directly.
Build and run it against the swapped-in package:

```
cl /EHsc /std:c++17 /I third_party\onnxruntime-training\include \
   tools\ort_training_probe.cpp \
   /link third_party\onnxruntime-training\lib\onnxruntime.lib
```

with `onnxruntime.dll` beside the exe. It asserts four things: `GetApi` at our
own `ORT_API_VERSION`, `GetApi(23)` for Muninn, a non-null `OrtTrainingApi`
struct, and a live function pointer in it (`LoadCheckpoint` on a missing file has
to return a clean `OrtStatus`, not crash). Expected output on the 1.30 build:

```
OK   base API requested v30, runtime version string = 1.30.0
OK   GetApi(23) satisfied - Muninn compatible
OK   training API struct present
OK   LoadCheckpoint rejected a bogus path cleanly: open file ... errcode = 2
```

Run this after any ORT rebuild or package swap. Also worth knowing: the logged
CMake command line shows `-Donnxruntime_ENABLE_TRAINING_OPS=OFF` and the
generated `CMakeCache.txt` records `OFF` as well, which looks like the gradient
kernels were skipped. They were not — `cmake/CMakeLists.txt` re-enables the
variable with a non-cache `set()` inside `if (onnxruntime_ENABLE_TRAINING_APIS)`,
so the cache entry is stale while the build is correct. Confirm by watching for
`batch_norm_grad.cc`, `softmax_cross_entropy_loss.cc` and `broadcast_grad_args.cc`
in the build log, or by finding `SoftmaxCrossEntropyLossGrad` in the built DLL.

## Guard against silent regression

Dropping the old NuGet package back in would quietly break neural VAD again — it
compiles, links, starts, and only fails when someone speaks. So the version check
is now in the build:

`FindOnnxRuntimeTraining.cmake` parses `ORT_API_VERSION` out of
`onnxruntime_c_api.h` and reports it. `CMakeLists.txt` compares it against
`DATAFORM_MUNINN_REQUIRED_ORT_API` (23) and defines
`DATAFORM_MUNINN_NEURAL_VAD` only when it is high enough; `speechrecognizer.cpp`
picks Silero or energy VAD off that define. Too old a runtime produces a configure
warning naming this document, and a build that still works on energy VAD rather
than one that crashes mid-sentence.

The same module now also reports whether `onnxruntime_providers_shared.dll`
exists, because a plain CPU build does not emit it and the previous unconditional
copy would have failed the build.

## Consequences

Good: one runtime for the whole process, on a current release; neural VAD
available; the training tier is off an abandoned 2024 binary, which was the
central worry in `ORT_MIGRATION_OPTIONS.md`. That document's "stay on 1.19.2"
recommendation is now moot — the concern it was weighing is resolved without
redesigning the training backend, so its Options 2-4 can be judged on their
merits rather than under dependency-rot pressure.

Bad: `third_party/onnxruntime-training/` is now a **locally built artifact**, not
a restorable package reference. That is the real cost. It needs to be archived
somewhere durable, and rebuilt when ORT is next bumped. The build is scripted
above and takes under an hour, but it is no longer `nuget restore`.

Unchanged: `ortinferencemanager.h` must still include
`onnxruntime_training_cxx_api.h` rather than the plain header, because MOC
compiles all `Q_OBJECT` headers together and the training header forward-declares
`OrtRelease(OrtTrainingSession*)` ahead of the base template.

## Closing the gap: three bugs in Muninn's streaming VAD

Unifying the runtime turned out to be necessary but not sufficient. Silero was
unreachable from the streaming path regardless of ORT version, and fixing that
uncovered two further bugs. All three are fixed in Muninn.

### 1. `vad_type` was ignored in streaming

`StreamingTranscriber::start()` hardcoded the energy detector:

```cpp
vad_ = std::make_unique<VAD>(vad_opts);   // options_.vad_type never read
```

`class VAD` is the energy detector — its own header says *"For noisy
environments, consider upgrading to Silero VAD (ONNX)"* — and `SileroVAD` was
wired only into the file-based transcriber. So `StreamingOptions::vad_type` was
an inert field, and the energy-VAD setting `0001-voice-loop.md` described as a
deliberate workaround was in fact a no-op: streaming was always energy.

`start()` now honours `vad_type`, resolving `Auto` to Silero when a model path is
configured and to energy otherwise. `StreamingOptions` gained `vad_model_path`
(there was no field for `silero_vad.onnx`) and `vad_fallback_to_energy`. Any
failure — not compiled in, no model path, load failure, or a throw mid-stream —
degrades to energy rather than killing transcription. The Silero instance is
reused across sessions when the path is unchanged, with `reset_state()` clearing
its recurrent state, because Dataform starts one session per utterance and would
otherwise create an ONNX session on every turn.

### 2. The preamble broke the silence contract

`SileroVADOptions::audio_preamble_ms` defaults to 2000 and exists so a *file*
whose speech starts immediately is not clipped. Every streaming chunk looks like
a fresh file to the detector, so a silent chunk still came back with a preamble
segment — meaning `filter_silence` never returned empty. That is precisely the
signal the streaming loop uses to skip silence, so silence was fed to Whisper
continuously.

Fixed on both sides: the preamble is disabled for streaming, and the loop now
treats "no segments" as silence as well as "no samples", so the two detectors
disagreeing about padding cannot resurrect the bug.

### 3. `stop()` could hang forever — pre-existing

The worst of the three, and unrelated to Silero. The chunk advance was:

```cpp
size_t remove_count = process_count - overlap_samples;
```

There is a guard above it keeping `overlap_samples` below `chunk_samples`, but
the subtraction uses `process_count`, which is `min(buffer_size, chunk_samples)`.
The flush pass lowers the loop threshold to 250 ms (4000 samples) while the
overlap here is 300 ms (4800). Once the tail landed between the two, the
subtraction underflowed to a huge `size_t`, the `remove_count <= size` bounds
check rejected every erase, and the loop spun on the same tail forever — burning
CPU, with `stop()` never returning.

For Dataform that meant every voice turn would hang in `endUtterance()`. It was
reachable with energy VAD too, so it predates all of this work; it only surfaced
here because the probe drove a full flush with a short tail.

All eight sites now use one computed `advance` that consumes the whole chunk when
it is shorter than the overlap, which guarantees forward progress.

### Verifying

`Muninn/examples/streaming_vad_selection.cpp` runs four sessions — Silero with a
model, Silero without a path, Silero again to exercise reuse, and energy
explicitly — and asserts each ends cleanly. Expected:

```
[Muninn] Streaming VAD: Silero (.../silero_vad.onnx)
[Muninn] Silero VAD requested but vad_model_path is empty
[Muninn] Streaming VAD: energy
    session ended cleanly, 0 tail segment(s)   (x4)
```

A hang in that probe means bug 3 is back.

Dataform resolves the model path itself: `SpeechRecognizer::defaultVadModelPath`
## Loki Studio: same symptom, different cause — also fixed

Loki's VAD failure looked identical but originated in Loki's own code, so
rebuilding `muninn.dll` did nothing for it. Its ORT pairing was always fine (a
1.23 runtime beside a 1.23-built Muninn); the settings simply were not wired.

Four transcription paths were broken, and one was not, which is why the symptom
looked intermittent for so long:

| Path | Before | Now |
| --- | --- | --- |
| `TranscribeTab` → `LokiWorker` (out of process) | correct — passed type, threshold and `silero_model_path` | unchanged |
| `WhisperTranscriber::transcribe` | `vad_type = Energy` hardcoded | from settings |
| `WhisperTranscriber::transcribeMultiTrack` | `vad_type = Energy` hardcoded | from settings |
| `AutoCutService` | `vad_type = Energy` hardcoded | from settings |
| `TimelineEditorTab` CC dual-pass | type mapped, but `silero_model_path` never set, so Muninn fell back to energy | from settings, model path supplied |

So the main Transcribe tab worked while multi-track, auto-cut and timeline
captions silently ran energy VAD regardless of what the settings said.

The mapping now lives in one place, `src/core/VadSettings.cpp`, because its
absence is what caused this: four call sites each hand-rolled the same options
block and three of them hardcoded the detector. `applyVadSettings()` reads
`whisperVadFilter`, `whisperVadType`, `whisperVadThreshold`, `whisperSileroModel`
and `whisperTenModel`, resolves model paths through `ModelPathManager`, logs the
chosen detector, and falls back to energy with a warning when a selected neural
model is not installed.

Two details worth keeping:

**The threshold is not detector-agnostic.** `vad_threshold` is an RMS level for
energy (Muninn default 0.02) and a speech probability for Silero. Loki's setting
is a probability — it defaults to 0.35 — and `transcriber.cpp` assigns it to the
energy detector verbatim. Passing it through unconditionally would have raised
the energy threshold seventeenfold and silenced most speech, turning a
do-nothing bug into a regression. It is applied only for explicit Silero/TEN, and
deliberately not for Auto, which Muninn resolves per track and may land on
energy.

**The CJK downgrade is preserved.** `TimelineEditorTab` forces energy VAD for
Korean and Japanese because neural VAD is too aggressive there, losing 25-38% of
content. That is why `applyVadSettings()` takes a type override rather than
having callers overwrite `vad_type` afterwards — the override has to flow through
the same threshold logic, or forcing energy would leave a probability in an RMS
field.

The default setting is "Auto", and `auto_detect_vad_type()` can return Silero, so
supplying the model path changes Loki's out-of-the-box behaviour: Auto can now
actually reach Silero instead of hitting the empty-path check and falling back.
The VAD models are installed (`%LOCALAPPDATA%\LokiStudio\Models\vad` holds
`silero_vad.onnx` among others), so this takes effect without a download.

Loki builds clean with the change. It does **not** need the rebuilt `muninn.dll`
— the file-based Silero path already worked in the shipped one — so Loki's
`staging/` and `Prebuilt/` copies were left alone.
