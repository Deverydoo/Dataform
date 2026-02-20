<p align="center">
  <img src="image1.png" alt="DATAFORM" width="300" />
</p>

<h1 align="center">DATAFORM</h1>
<h3 align="center">Darwinian Adaptive Trait-Assimilating Form</h3>

<p align="center">
  A local-first AI companion that learns its user over decades through evolutionary personality adaptation.
</p>

---

## What is DATAFORM?

DATAFORM is a desktop AI assistant that doesn't just respond — it **remembers**, **learns**, and **evolves**. Unlike cloud-based AI that forgets you between sessions, DATAFORM builds a persistent model of who you are: your values, preferences, communication style, and goals.

All data stays on your machine. No telemetry. No cloud sync. Your personality model belongs to you.

### Core Concept

DATAFORM uses a biologically-inspired pipeline:

- **Episodic Memory** — Every conversation is stored locally with encrypted SQLite databases
- **Trait Extraction** — An LLM analyzes your conversations to extract stable personality traits (values, preferences, motivations)
- **LoRA Training** — A personal neural network adapter trains on your data during idle time
- **Darwinian Evolution** — Multiple adapter variants compete; the best-performing ones survive and reproduce
- **Personalized Distillation** — A teacher model generates high-quality training pairs, evaluated by embedding similarity, to continuously improve the personal model
- **Decade-Scale Lifecycle** — Schema migration, health checks, and data lifecycle management designed for years of continuous use

### Three-Tier Architecture

DATAFORM runs three AI layers simultaneously:

| Layer | Engine | Role | Size |
|-------|--------|------|------|
| **External Brain** | Ollama / LM Studio / OpenAI / Anthropic | User-facing chat, reasoning, complex responses | 7B–70B+ |
| **Internal Brain** | llama.cpp (embedded) | Background JSON tasks — trait extraction, sentiment, research, goals, news, learning | 3B Q8_0 |
| **Personality Layer** | ONNX Runtime LoRA adapters | Behavioral/stylistic delta capturing communication style, preferences, quirks via evolutionary selection | 5–20 MB |

The external and internal brains operate independently — background tasks never block your chat, and chat never stalls background processing.

## Features

**Conversational Intelligence**
- Multi-provider LLM support (Ollama, LM Studio, OpenAI, Anthropic, embedded llama.cpp)
- Context-aware responses using trait memory, mood tracking, and semantic recall
- Curiosity engine that asks genuine follow-up questions
- Agentic tool system — web search, memory recall, research lookup, save notes, and set reminders via natural language

**Autonomous Background Processing**
- Idle-time research — DATAFORM searches the web and summarizes findings on topics you care about
- News awareness — Fetches headlines and starts conversations about stories relevant to your interests
- Goal tracking — Detects goals from conversation and checks in on progress
- Sentiment analysis — Tracks mood trends and adjusts conversational tone
- Learning plans — Generates multi-session curricula on topics you want to explore
- Personalized distillation — Teacher model generates training data, evaluated for quality via embedding similarity
- Coordinated scheduling — IdleJobCoordinator runs one engine at a time with priority-based round-robin, preventing queue gridlock

**Proactive Dialog**
- Generates thoughts during idle time from research, curiosity, training, news, and distillation readiness
- Surfaces conversation starters via a notification badge
- Reminders parsed from natural language ("remind me to...")
- Daily digest of pending thoughts

**Personality Evolution**
- Population-based LoRA adapter training with mutation and crossover
- Evaluation suite scoring identity alignment, curiosity quality, and stability
- Automatic promotion of best-performing personality variants
- Full adapter lineage tracking across generations and eras

**Privacy & Longevity**
- AES-256-CBC encryption for all databases
- Profile health checks with self-healing and backup rotation
- Episode compaction, trait confidence decay, and checkpoint pruning
- Schema versioning with forward migration for decade-scale data integrity

## Architecture

```
Qt6/QML Desktop App (C++17)
│
├── Cognitive Core
│   ├── Orchestrator              — Cognitive loop with agentic ReAct tool-calling
│   ├── MemoryStore               — Dual encrypted SQLite (episodic + traits)
│   ├── EmbeddingManager          — Semantic vector search via Ollama embeddings
│   ├── LLMProviderManager        — Multi-provider with background queue + foreground priority
│   ├── ToolRegistry              — 5 built-in tools (web search, memory, research, notes, reminders)
│   └── LLMResponseParser         — Shared JSON/thinking-tag parsing utility
│
├── Personality Pipeline
│   ├── TraitExtractor            — LLM-based personality trait extraction
│   ├── WhyEngine                 — Curiosity controller with budget/cooldown
│   ├── SentimentTracker          — Per-episode mood analysis with trend detection
│   └── GoalTracker              — Goal detection and periodic check-ins
│
├── Autonomous Engines
│   ├── ResearchEngine            — 5-phase idle-time web research pipeline
│   ├── NewsEngine                — Headline fetch and discussion generation
│   ├── LearningEngine            — Multi-session learning plan curricula
│   ├── DistillationManager       — Teacher-student distillation with quality eval
│   └── IdleJobCoordinator        — Serializes engine activation, one at a time
│
├── Proactive Dialog
│   ├── ThoughtEngine             — Thought generation from 10 sources
│   └── ReminderEngine            — Natural language reminder parsing
│
├── Evolution & Training
│   ├── ReflectionEngine          — Idle-time LoRA training pipeline
│   ├── EvolutionEngine           — Population-based adapter selection
│   ├── AdapterManager            — Versioned LoRA adapter management
│   ├── OrtTrainingManager        — ONNX Runtime training sessions
│   ├── OrtInferenceManager       — Local model inference with sampling
│   ├── TrainingDataGenerator     — 9-source tokenized training data
│   └── LineageTracker            — Adapter genealogy and era timeline
│
├── Embedded Inference
│   └── LlamaCppManager           — llama.cpp C API wrapper for background tasks
│
└── Lifecycle & Infrastructure
    ├── ProfileHealthManager      — Integrity checks and backup/restore
    ├── DataLifecycleManager      — Compaction, decay, and pruning
    ├── SchemaMigrator            — DB + JSON schema versioning
    ├── IdleScheduler             — Windows idle/power/thermal detection
    ├── SettingsManager           — JSON config with schema versioning
    ├── ProfileManager            — Profile directory tree management
    ├── CryptoUtil                — AES-256-CBC via Windows BCrypt
    └── WebSearchEngine           — DuckDuckGo search + page fetch
```

## Building

### Prerequisites

- **Qt 6.10+** with Quick, Qml, Network, Sql, Charts, Widgets modules
- **CMake 3.16+**
- **MSVC 2022** (Windows)
- **Ollama** running locally (recommended default provider)

### Build Options

```bash
# Standard build (no training, no llama.cpp)
cmake -B build -DCMAKE_PREFIX_PATH="C:/Qt/6.10.1/msvc2022_64"
cmake --build build --config Release

# With LoRA training support (requires ONNX Runtime Training v1.19.2)
cmake -B build -DCMAKE_PREFIX_PATH="C:/Qt/6.10.1/msvc2022_64" -DDATAFORM_ENABLE_TRAINING=ON
cmake --build build --config Release

# With embedded llama.cpp for background tasks
cmake -B build -DCMAKE_PREFIX_PATH="C:/Qt/6.10.1/msvc2022_64" -DDATAFORM_ENABLE_LLAMACPP=ON
cmake --build build --config Release

# Full build (training + llama.cpp)
cmake -B build -DCMAKE_PREFIX_PATH="C:/Qt/6.10.1/msvc2022_64" \
  -DDATAFORM_ENABLE_TRAINING=ON -DDATAFORM_ENABLE_LLAMACPP=ON
cmake --build build --config Release
```

### Run

```bash
build/Release/Dataform.exe
```

Ensure Ollama is running with a model pulled (e.g., `ollama pull qwen3:8b`).

For llama.cpp background inference, place a `.gguf` model file in `models/background_llm/`.

## Project Status

DATAFORM is in active development. Completed phases:

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Foundations — memory, orchestrator, idle scheduler | Done |
| 1 | Curiosity engine + trait assimilation + eval suite | Done |
| 2 | Native LoRA training via ONNX Runtime | Done |
| 3 | Darwinian evolution engine (population-based selection) | Done |
| 4 | Decade-scale lifecycle management | Done |
| 5 | Autonomous idle-time web research | Done |
| 6 | Personal model pipeline (Side LLM via ORT inference) | Done |
| 7 | Semantic memory + expanded training (9 data sources) | Done |
| 8 | Personalized distillation (teacher-student pipeline) | Done |
| + | Conversation history, proactive dialog, news, reminders, goals, mood tracking, learning plans | Done |
| + | llama.cpp embedded inference for background tasks | Done |
| + | Agentic tool system (5 built-in tools, ReAct loop) | Done |
| + | Idle job coordinator (serialized engine scheduling) | Done |

## License

**Free for personal use.** Commercial use requires a separate license. See [LICENSE](LICENSE) for details.
