# Voice loop: Muninn for listening, replaceable backend for speaking

Status: accepted
Date: 2026-09-12
Context: the Home Jarvis project plan (a private planning document, not in this repository), milestone M2

Written for: anyone working on the voice path or trying to get it running.

## What was already here

The Jarvis plan describes an assistant core, a llama.cpp adapter, a memory and
task store, a tool dispatcher and a reminder scheduler. Dataform already had all
of those. What it had none of was audio: no capture, no transcription, no speech,
and `Qt6::Multimedia` was not even linked. That gap is what this change closes.

The existing pieces were reused rather than rebuilt:

| Plan component | What it maps onto |
| --- | --- |
| Assistant core / model-tool loop | `Orchestrator`, including the existing 5-step ReAct bound |
| Inference adapter | `LlamaCppManager` — embedded, not the loopback llama-server the plan assumed |
| Memory and tasks | `MemoryStore` + `SchemaMigrator` |
| Tool dispatcher | `ToolRegistry` |
| Reminders | `ReminderEngine` |

Keeping `LlamaCppManager` embedded is a deliberate departure from the plan's
supervised-subprocess design: there is no separate process to supervise, restart
or health-check, and the plan's supervisor requirement falls away with it.

## Reuse audit (plan §5, §17)

**Muninn** is reused directly as a library. `muninn::StreamingTranscriber` is the
right shape for a live microphone — push audio from the capture callback, poll or
receive segments, VAD built in — and it answers plan question §17.1: yes, Muninn
supports incremental transcription.

**Loki Studio's `WhisperTranscriber` was not reused.** It wraps
`muninn::Transcriber` for whole files, not the streaming API, and Loki has no
microphone capture anywhere — it is a video tool, and there is not a single
`QAudioSource` in it. Its structure was a useful model for the Qt wrapper; its
code was not applicable. Capture is new, on Qt 6 Multimedia.

**Var (`cpp_var_tts_engine`) was not used for speech.** It is built and it clones
a voice, but it is a 1B-parameter model plus a vocoder, and it links Skuld's
llama.cpp build while Dataform statically links its own. Two llama.cpp builds in
one process is an ABI risk, and the latency is hard to reconcile with the plan's
3-second first-audio target. The synthesis backend is an interface, so Var
remains addable later — out-of-process would sidestep the conflict.

## Structure

```
src/voice/
  audiocapture.*        QAudioSource, 16 kHz mono float, 20 ms chunks, device loss
  audioplayback.*       QAudioSink stream with immediate stop for barge-in
  speechrecognizer.*    Muninn StreamingTranscriber on a worker thread
  speechsynthesizer.*   Queue, sentence splitting, backend selection
  ttsbackends.*         PiperBackend (subprocess), SapiBackend (Windows)
  sapiworker.*          SAPI COM voice on its own thread
  voicesession.*        The turn state machine
```

`VoiceSession` owns turn identity. Every transcript and every speech result
carries the turn that asked for it, and results whose turn has been superseded
are dropped rather than acted on — that is what makes cancellation safe, and it
is why `cancel()` bumps the turn id before touching anything else.

## Decisions worth knowing

**Energy VAD, not Silero.** *Superseded — Silero is now running; see
`Docs/decisions/0002-onnxruntime-unification.md`. The ONNX Runtime versions have
since been unified on a locally built 1.30, so the constraint below no longer
applies. The reasoning was also incomplete: Muninn's streaming path hardcodes the
energy VAD and ignores `vad_type` entirely, so streaming was never going to reach
Silero whatever the runtime said. The paragraph is kept because the ORT
compatibility rule it explains is still worth knowing.*

This is the constraint most likely to surprise
someone. Muninn on this machine is built against ONNX Runtime **1.23**; the
training package in `third_party/onnxruntime-training` supplies **1.19**, and
only one `onnxruntime.dll` can be loaded per process. `muninn.dll` imports a
single ordinal (`OrtGetApiBase`) so it loads either way, but the moment it
actually uses ONNX it calls `GetApi(23)` against a 1.19 runtime and gets null
back. Energy VAD never reaches that call, so the two coexist. Neural VAD and
diarization need the two ONNX Runtime versions reconciled first — see the comment
in `speechrecognizer.cpp`.

The build copies the *training* runtime, not Muninn's, because the training
package is the one with the training API the reflection engine needs. Muninn's
`onnxruntime.dll` is deliberately excluded from the copy list when
`DATAFORM_ENABLE_TRAINING` is on.

**Piper runs as a subprocess, one per utterance.** That costs a model load
(~100-200 ms for a medium voice) but buys exact cancellation: killing the process
stops audio immediately with no half-consumed model state. Its raw PCM is
forwarded as it arrives, so playback starts before synthesis finishes.

**SAPI is a real backend, not a stub.** Piper needs a separate download; SAPI
ships with Windows. It plays audio itself, so it reports `sampleRate() == 0`,
emits no samples, and cancels through its own purge rather than through
`AudioPlayback`.

**A turn ends when playback drains, not when synthesis finishes.** Capture is
half-duplex — nothing here cancels the echo of the assistant's own voice — so
reopening the microphone at the last synthesised sample would transcribe the tail
of the reply. Full duplex needs echo cancellation first.

**Push-to-talk only.** No wake word, per the plan's M6 deferral. `Ctrl+Space`
toggles as a keyboard alternative, because a shortcut cannot report key release.

## Building it

Voice is behind `DATAFORM_ENABLE_VOICE` (default ON) and needs a built Muninn
checkout. `FindMuninn.cmake` looks at `MUNINN_ROOT`, then `third_party/muninn`,
then the `../Muninn-Faster-Whisper` sibling. A missing Muninn downgrades to a
text-only build with a warning rather than failing configuration.

```
cmake -B build -DCMAKE_PREFIX_PATH="C:/Qt/6.10.1/msvc2022_64" ^
      -DMUNINN_ROOT="<path to Muninn-Faster-Whisper>"
```

Then in Settings > Voice: enable voice, point the model folder at a faster-whisper
directory (`faster-whisper-large-v3-turbo` is the usual choice), and pick a speech
engine. Piper needs `piper.exe` and a voice `.onnx` with its `.onnx.json` config
beside it; SAPI needs nothing.

## Measured so far

On an RTX 4090, `faster-whisper-large-v3-turbo` at float16 on CUDA loads in about
1.2 seconds. The plan's latency targets (§14) are not yet measured —
`VoiceSession::turnTimings` emits the per-turn breakdown the benchmark needs, but
nothing consumes it yet.

## Not done

Wake word, barge-in during capture, echo cancellation, quiet hours, the expanded
tool set (`notes.*`, `tasks.*`, `files.*`, `apps.open`, `system.status`), and the
reminder-durability work the plan asks for in §10 — timezones, recurrence, and
surfacing reminders missed while the machine was off.
