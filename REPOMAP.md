# DATAFORM Repository Map

**DATAFORM** — Darwinian Adaptive Trait-Assimilating Form
Qt6/QML + C++17 desktop app | Windows (MSVC 2022) | CMake

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        QML UI Layer                         │
│  Main.qml │ ChatWindow │ SettingsPanel │ Sidebar │ Panels  │
├─────────────────────────────────────────────────────────────┤
│                     Orchestrator                            │
│         Cognitive loop: msg → context → LLM → store         │
├──────────┬──────────────┬───────────────┬───────────────────┤
│ External │   Internal   │  Personality  │   Idle-Time       │
│  Brain   │    Brain     │    Layer      │   Pipelines       │
│ (Ollama/ │ (llama.cpp   │ (ORT LoRA     │ (Research, News,  │
│  Cloud)  │  embedded)   │  adapters)    │  Goals, Training) │
├──────────┴──────────────┴───────────────┴───────────────────┤
│                    Data & Storage                            │
│  MemoryStore (SQLite) │ ProfileManager │ CryptoUtil (AES)   │
└─────────────────────────────────────────────────────────────┘
```

---

## Source Files — `src/`

### Entry Point
| File | Purpose |
|------|---------|
| `main.cpp` | Wires all 34 components, exposes to QML, startup model loading |

### Core (Phase 0)
| Class | Header | Impl | Role |
|-------|--------|------|------|
| **Orchestrator** | orchestrator.h | orchestrator.cpp | Cognitive loop: user msg → identity context → curiosity → LLM → store episode → trait extraction |
| **LLMProviderManager** | llmprovider.h | llmprovider.cpp | Multi-provider routing (Ollama, LM Studio, OpenAI, Anthropic, Local), separate background queue, token streaming |
| **MemoryStore** | memorystore.h | memorystore.cpp | Dual SQLite DBs (episodic + traits), encrypted at rest, lifecycle queries |
| **SettingsManager** | settingsmanager.h | settingsmanager.cpp | JSON config with schema versioning, 30+ settings |
| **ProfileManager** | profilemanager.h | profilemanager.cpp | Profile directory tree, export/import |
| **CryptoUtil** | cryptoutil.h | cryptoutil.cpp | AES-256-CBC via Windows BCrypt API |
| **IdleScheduler** | idlescheduler.h | idlescheduler.cpp | Windows idle/power/thermal detection, 5s polling, 120s threshold |

### Phase 1 — Curiosity & Traits
| Class | Header | Impl | Role |
|-------|--------|------|------|
| **WhyEngine** | whyengine.h | whyengine.cpp | Curiosity controller: 5 trigger types, 15 templates, budget/cooldown |
| **TraitExtractor** | traitextractor.h | traitextractor.cpp | LLM-based trait extraction, JSON parsing, dedup, evidence merging |
| **EvalSuite** | evalsuite.h | evalsuite.cpp | 10 tests across 3 categories (identity alignment, curiosity quality, stability) |
| **FuzzyLogic** | fuzzylogic.h | _(header-only)_ | Fuzzy set operations for trait confidence |

### Phase 2 — LoRA Training _(conditional: `DATAFORM_TRAINING_ENABLED`)_
| Class | Header | Impl | Role |
|-------|--------|------|------|
| **Tokenizer** | tokenizer.h | tokenizer.cpp | BPE tokenizer with configurable chat template |
| **TrainingDataGenerator** | trainingdatagenerator.h | trainingdatagenerator.cpp | 8 data sources → tokenized training examples (55/15/10/10/5/5 split) |
| **OrtTrainingManager** | orttrainingmanager.h | orttrainingmanager.cpp | ORT TrainingSession wrapper, background QThread, checkpoint save/load |
| **AdapterManager** | adaptermanager.h | adaptermanager.cpp | Versioned adapters with origin/generationId/cycleId/variantIndex |
| **ReflectionEngine** | reflectionengine.h | reflectionengine.cpp | 5-phase idle pipeline: Select → Generate → Train → Evaluate → Promote/Reject |
| **PopulationState** | populationstate.h | populationstate.cpp | VariantSpec + PopulationState structs with JSON persistence |
| **WeightMerger** | weightmerger.h | weightmerger.cpp | ORT checkpoint weight averaging via ToBuffer/FromBuffer |

### Phase 3 — Darwinian Evolution _(conditional: `DATAFORM_TRAINING_ENABLED`)_
| Class | Header | Impl | Role |
|-------|--------|------|------|
| **EvolutionEngine** | evolutionengine.h | evolutionengine.cpp | Population-based evolution, variant breeding, hot-swap on promotion |

### Phase 4 — Decade-Scale Lifecycle
| Class | Header | Impl | Role |
|-------|--------|------|------|
| **SchemaMigrator** | schemamigrator.h | schemamigrator.cpp | PRAGMA user_version for DBs, schemaVersion for JSON, migration framework |
| **ModelGenerationManager** | modelgeneration.h | modelgeneration.cpp | Tracks model generations (family, variant, chat template, tokens) |
| **ProfileHealthManager** | profilehealthmanager.h | profilehealthmanager.cpp | Startup integrity checks, self-healing, backup/restore with rotation |
| **DataLifecycleManager** | datalifecyclemanager.h | datalifecyclemanager.cpp | Episode compaction, trait decay, checkpoint pruning, disk monitoring |
| **LineageTracker** | lineagetracker.h | lineagetracker.cpp | Adapter genealogy tree, era timeline, persisted in lineage.json |

### Phase 5 — Autonomous Research
| Class | Header | Impl | Role |
|-------|--------|------|------|
| **WebSearchEngine** | websearchengine.h | websearchengine.cpp | DuckDuckGo HTML search + page fetch, HTML-to-text, rate limiting |
| **ResearchStore** | researchstore.h | researchstore.cpp | research_findings table CRUD, pending/approved/rejected status |
| **ResearchEngine** | researchengine.h | researchengine.cpp | 5-phase idle pipeline: Topic → Query → Search → Fetch+Summarize → Store |

### Phase 6 — Local Model Inference _(conditional: `DATAFORM_TRAINING_ENABLED`)_
| Class | Header | Impl | Role |
|-------|--------|------|------|
| **OrtInferenceManager** | ortinferencemanager.h | ortinferencemanager.cpp | ORT InferenceSession, autoregressive generation, temperature/top-k/top-p |

### Proactive Dialog & Agentic Features
| Class | Header | Impl | Role |
|-------|--------|------|------|
| **ThoughtEngine** | thoughtengine.h | thoughtengine.cpp | 10 thought types, idle-time generation, daily digest, CRUD |
| **NewsEngine** | newsengine.h | newsengine.cpp | AllSides headline fetch, LLM selection + thought generation |
| **ReminderEngine** | reminderengine.h | reminderengine.cpp | 60s polling, "remind me" detection, time expression parsing |
| **GoalTracker** | goaltracker.h | goaltracker.cpp | LLM-based goal detection from episodes, periodic check-in thoughts |
| **SentimentTracker** | sentimenttracker.h | sentimenttracker.cpp | Per-episode sentiment analysis, mood trend detection, getMoodHint() |
| **LearningEngine** | learningengine.h | learningengine.cpp | Multi-session learning plans, LLM-generated curricula |

### Phase 7 — Semantic Memory
| Class | Header | Impl | Role |
|-------|--------|------|------|
| **EmbeddingManager** | embeddingmanager.h | embeddingmanager.cpp | Ollama /api/embed, vector cache, cosine similarity, idle-time batch |

### llama.cpp Integration _(conditional: `DATAFORM_LLAMACPP_ENABLED`)_
| Class | Header | Impl | Role |
|-------|--------|------|------|
| **LlamaCppManager** | llamacppmanager.h | llamacppmanager.cpp | Embedded GGUF inference, chat template, threaded generation, mmap |

### Utilities
| Class | Header | Impl | Role |
|-------|--------|------|------|
| **ClipboardHelper** | clipboardhelper.h | clipboardhelper.cpp | System clipboard access for QML |

---

## QML UI — `qml/`

| File | Purpose |
|------|---------|
| **Main.qml** | Root window: ConversationSidebar \| ChatWindow \| IdlePanel + MemoryPanel, thought badge header |
| **ChatWindow.qml** | Chat interface: message bubbles, feedback buttons, curiosity triggers, token streaming |
| **ConversationSidebar.qml** | Conversation history list, thought cards, rename/delete context menu |
| **SettingsPanel.qml** | **4 tabs**: Models \| Behavior \| Training \| Data |
| **IdlePanel.qml** | Idle status display, training progress section |
| **MemoryPanel.qml** | Episodes/traits browser, research tab, semantic search |
| **LineagePanel.qml** | Adapter genealogy tree visualization |

---

## Build System

| File | Purpose |
|------|---------|
| **CMakeLists.txt** | Main build config: Qt 6.10.1, conditional ORT training + llama.cpp |
| **cmake/FindOnnxRuntimeTraining.cmake** | CMake find module for ORT Training NuGet package |
| **dataform.rc** | Windows resource file (icon, version info) |

### Build Options
```bash
# Standard build (training ON, llama.cpp OFF)
cmake -B build -DCMAKE_PREFIX_PATH="C:/Qt/6.10.1/msvc2022_64"

# Full build (training + llama.cpp)
cmake -B build -DCMAKE_PREFIX_PATH="C:/Qt/6.10.1/msvc2022_64" \
  -DDATAFORM_ENABLE_TRAINING=ON -DDATAFORM_ENABLE_LLAMACPP=ON

# Build
cmake --build build --config Release
```

---

## Third Party — `third_party/`

| Directory | Version | Purpose |
|-----------|---------|---------|
| `onnxruntime-training/` | v1.19.2 | ORT Training NuGet (headers, libs, DLLs) |
| `llama.cpp/` | latest | Git submodule — embedded GGUF inference library |

---

## Tools — `tools/`

| File | Purpose |
|------|---------|
| `generate_artifacts.py` | Offline: generates training artifacts (vocab, merges, ONNX models) |
| `requirements.txt` | Python dependencies for artifact generation |

---

## Documentation — `Docs/`

| File | Purpose |
|------|---------|
| `DATAFORM_Master_Project_Plan.md` | Top-level roadmap |
| `DATAFORM_Full_Project_Plan_and_Scope.md` | Detailed scope |
| `DATAFORM_MemorySchema.md` | Database schema (episodic + traits tables) |
| `DATAFORM_WhyEngine_Spec.md` | Curiosity engine specification |
| `DATAFORM_EvalSuite.md` | Evaluation framework spec |
| `DATAFORM_EvolutionHarness.md` | Evolution engine spec |
| `DATAFORM_LoRA_Training_Pipeline.md` | Training pipeline details |
| `summaries/` | 35+ curated ML/AI paper summaries |

---

## Runtime Data — `profiles/`

```
profiles/default/
├── memory/          Episodic + traits SQLite DBs (encrypted)
├── adapters/
│   ├── active/      Currently promoted adapter
│   └── versions/    All adapter versions
├── checkpoints/     ORT training checkpoints
├── eval/            Evaluation results
├── evolution/       Population state, variant specs
├── era_snapshots/   Era-based lineage snapshots
├── backups/         Rotating profile backups
├── settings.json    User configuration
└── lineage.json     Adapter genealogy tree
```

---

## Statistics

| Metric | Count |
|--------|-------|
| C++ classes | 34 |
| Header files (.h) | 34 |
| Implementation files (.cpp) | 35 (incl. main.cpp) |
| QML files | 7 |
| Total source lines | ~75,000 |
| SQLite tables | ~12 (episodic, traits, research, thoughts, goals, reminders, sentiment, learning, news_seen, embeddings, ...) |
| DB schema version | 8 |
| Settings properties | 30+ |
