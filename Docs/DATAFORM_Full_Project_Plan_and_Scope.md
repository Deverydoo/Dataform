# 🧠 Project DATAFORM (Full Project Plan & Scope)
### Darwinian Adaptive Trait-Assimilating Form  
**Goal:** A packaged, hands-off, sellable desktop product that pairs a **fixed base LLM** (Ollama) with **trainable layers** that **learn and adapt over years/decades** via **idle-time reflection + Darwinian selection**, using **native C++** and **ONNX Runtime Training** (no Python requirement at runtime).

---

## 0) Product Definition (What ships)
DATAFORM is a local-first AI agent with:
- **Base Brain (Fixed):** user-selected Ollama model (Qwen / DeepSeek / etc.)
- **Growth Brain (Trainable):** LoRA/adapter layers + a curiosity/policy brain that update over time
- **Idle Mind:** background reflection + evolutionary training at **low resource usage** by default
- **Identity Memory:** durable user model derived from “why,” principles, and preferences (not just facts)
- **Safety & Stability:** eval gates, versioning, rollback, audit trails, encryption

**Non-goal:** “RAG-only personalization.” Memory exists, but learning must materially change behavior via **trainable weights**.

---

## 1) Core Principles & Design Constraints
### 1.1 Prime Directive
> **Seek first to understand, then to respond.**

### 1.2 Fixed Base Brain
- The Ollama model is **immutable** in production.
- Adaptation occurs by training and attaching **adapters**, not rewriting the base model.

### 1.3 Decades-Scale Learning
Avoid a single endlessly-growing adapter. Use:
- **Multiple adapters** (by domain / era / role)
- **Consolidation** into `identity_core`
- **Pruning + snapshots** to prevent bloat and drift

### 1.4 “Idle Pondering,” Not Aggressive Training
Default: background learning uses **~10% compute budget** and runs **only when idle + plugged in**.
User toggle: **Aggressive Learning** (higher budgets, optional).

### 1.5 Hands-off UX
User should never manage datasets, checkpoints, or training runs.
Everything is automated, reversible, and explainable.

---

## 2) High-Level System Architecture
### 2.1 Components (Runtime)
1. **Qt UI (C++)**
   - Chat interface (already exists)
   - Lightweight feedback controls (👍/👎, edit-to-teach, optional “why” chips)
   - Training status + privacy controls

2. **Orchestrator (C++)**
   - Turn pipeline (respond → optionally ask-why → store → queue learning)
   - Routes between Ollama + adapter manager + memory + evaluators

3. **Ollama Connector (C++)**
   - HTTP streaming inference
   - Base model selection
   - Adapter application via Modelfile/ADAPTER mechanism (where supported)

4. **Memory Store (Local)**
   - SQLite/DuckDB for episodic + semantic identity
   - Encrypted at rest
   - Full audit trail

5. **Idle Mind Scheduler**
   - Detects idle time, plugged-in state, thermals, user settings
   - Runs reflection + training under budget

6. **ONNX Runtime Inference + Training (C++)**
   - Inference for policy/evaluators (and any auxiliary models)
   - Training for LoRA/adapters using ORT training artifacts + checkpoint states

7. **Adapter Manager**
   - Stores adapter versions, applies active set, atomic swap, rollback
   - Maintains multi-adapter configuration per user

8. **Evaluation & Safety Gate**
   - Runs “User-ness” + correctness + stability + safety tests
   - Blocks promotion if regression detected

### 2.2 Data Flow (Per User Message)
1. User message → Orchestrator
2. Retrieve minimal identity signals (traits, constraints) from semantic store
3. Ollama inference with active adapter set (or base + adapter pipeline)
4. If triggered: ask **one** “why” question (gentle, optional)
5. Store episodic record + any explicit “why” response
6. Queue learning signals for Idle Mind

---

## 3) The DATAFORM Cognitive Loop
**Interact (online):**
- Answer → Ask Why (sometimes) → Assimilate trait → Store evidence → Queue learning

**Reflect (offline/idle):**
- Select episodes → Extract lessons → Generate training examples → Train variants → Evaluate → Promote/rollback → Prune/merge

---

## 4) Memory & Identity Model (Not RAG-Only)
### 4.1 Episodic Memory (Life Log)
Stores raw and minimally processed experiences:
- message pairs, timestamps, topic tags
- thumbs up/down
- edits and diffs (best learning signal)
- outcomes (did user accept? re-ask? correct?)
- tool usage (if any)

### 4.2 Semantic Identity Memory (Trait Graph)
Stores distilled, stable attributes derived from “why”:
- values (e.g., “understanding prevents conflict”)
- preferences (tone, structure)
- decision policies (risk tolerance, verification habits)
- motivational patterns (what triggers certain questions)
Each trait includes: evidence links, confidence, last-confirmed timestamp.

### 4.3 Confidence-to-Inquiry Controller
Controls how often the agent asks “why.”
- High confidence + stable → ask less
- Novel topic / contradictions / drift → ask more
- Occasional low-frequency “spot-check” remains

---

## 5) Genuine Learning Mechanisms (Trainable Weights)
### 5.1 Trainable Targets
DATAFORM must learn in weights:
1. **Style/Voice** (how it sounds)
2. **Decision Policy** (how it reasons, prioritizes, verifies)
3. **Curiosity Policy** (when/how to ask “why”)

### 5.2 Adapter Strategy (Decade-Scale)
Maintain multiple adapters:
- `identity_core` (slow-changing)
- `communication_style` (medium)
- domain adapters (`rpg_ai`, `health`, `music`, etc.)
- era snapshots (`2026_core`, `2028_core`) if needed

Periodic consolidation:
- Weekly/monthly distillation merges stable improvements into `identity_core`
Pruning:
- Archive adapters that are cold or redundant

### 5.3 Darwinian Evolution (Population Training)
During idle:
- Spawn N variants (8–32 typical)
- Train each variant on slightly different minibatches/mutations
- Score variants against eval suite
- Promote best if it beats current champion by margin
- Rollback automatically on any regression

---

## 6) Idle Mind Subsystem (Self-Reflection While User Is Away)
### 6.1 Reflection Loop (Low-cost)
Runs frequently (cheap):
- Identify high-signal episodes (edits, strong feedback, repeated questions)
- Extract “why/principle/constraint”
- Generate supervised targets and preference pairs
- Update trait graph hypotheses (with confidence)

### 6.2 Evolution Loop (Moderate-cost)
Runs less often:
- Train adapter variants under strict compute budgets
- Evaluate and promote
- Save snapshots and logs

### 6.3 Resource Budgets (Defaults)
**Default mode (hands-off):**
- Only when **idle + plugged in**
- Target: ~10% compute
- Hard caps:
  - GPU: low utilization target (configurable)
  - CPU: low priority background threads
  - VRAM: bounded workspace; pause if interactive workload starts
- Thermal/power guard:
  - pause if temp high or on battery

**Aggressive Learning (user toggle):**
- Higher budgets, optional battery allowance, more frequent evolution cycles

---

## 7) Model Support & Conversion Strategy
### 7.1 Supported Base Models (Tiered)
For a sellable v1/v2, define Tier 1 families (example):
- Qwen2.5 variants
- DeepSeek variants

Reason: ONNX training artifacts must match the base architecture.

### 7.2 Dual Representation Requirement
- **Inference:** Ollama GGUF
- **Training:** matching ONNX model graph (frozen base + trainable LoRA)

### 7.3 Artifact Pipeline (per base family)
Offline build step (your internal tooling) generates:
- training ONNX graph
- optimizer graph
- (optional) eval graph
- checkpoint template
These are shipped with the product for each supported base family.

---

## 8) Packaging & Distribution (User-Friendly)
### 8.1 Installer Experience
- One installer sets up:
  - DATAFORM app (Qt)
  - ONNX Runtime + ORT Training libs
  - model manager integration (detects/uses Ollama)
  - optional base model download flow (via Ollama)

### 8.2 Storage Layout (Local)
- `profiles/<user_id>/`
  - `memory/episodic.db` (encrypted)
  - `memory/traits.db` (encrypted)
  - `adapters/active/`
  - `adapters/versions/vNNN/`
  - `checkpoints/`
  - `eval/` logs + reports
  - `settings.json`

### 8.3 Versioning + Rollback
- All promotions are atomic:
  - write new adapter set to `vNNN`
  - run eval
  - switch `active -> vNNN`
- One-click revert to previous stable version

---

## 9) Evaluation, Regression Gates, and Safety
### 9.1 Eval Suite Categories
1. **Identity alignment** (“User-ness”)
2. **Curiosity quality** (asks why when appropriate; stops when confident)
3. **Correctness & groundedness**
4. **Stability** (no drift on core behaviors)
5. **Safety & boundaries** (no regression)

### 9.2 Promotion Rules
A candidate adapter set is promoted only if:
- passes all safety tests
- meets or exceeds identity score threshold
- improves over champion by minimum margin
- does not regress on core tasks

### 9.3 Audit Trail
Every training cycle stores:
- data selection summary
- training config + seed
- fitness scores
- diff against champion
- promotion decision + reason

---

## 10) UX Requirements (Minimal, Powerful Feedback)
### 10.1 Must-have Controls
- 👍 / 👎
- “Edit to teach” (user rewrites assistant output; diff captured)
- Optional “Why I’m asking” chips (one tap):
  - curiosity / planning / health / conflict / money / relationships / other

### 10.2 User Settings
- Idle learning: on/off
- Default compute budget (~10%)
- Aggressive learning toggle
- Privacy:
  - what to store
  - “forget last session”
  - export/delete profile

---

## 11) Implementation Roadmap (Phased)
### Phase 0 — Foundations (2–4 weeks)
- Orchestrator integration with existing UI
- Episodic logging + encrypted storage
- Trait graph basics (manual extraction or rule-based)
- Idle scheduler with budgets (idle + plugged-in detection)
**Deliverable:** stable app with durable memory & idle framework (no training yet)

### Phase 1 — Why Engine + Trait Assimilation (4–6 weeks)
- One-question “Why Engine” with confidence controller
- Trait extraction pipeline (evidence + confidence)
- Basic eval suite scaffolding
**Deliverable:** agent asks “why” intelligently; identity model starts forming

### Phase 2 — Native ORT Training MVP (6–10 weeks)
- Ship training artifacts for Tier 1 base family
- Implement ORT TrainingSession + checkpointing in C++
- Train a single adapter (`communication_style`) from edits + ratings
- Adapter manager + atomic promotion + rollback
**Deliverable:** genuine weight updates that change behavior

### Phase 3 — Darwinian Evolution Engine (6–10 weeks)
- Population variants (N)
- Fitness scoring
- Automated promotion policy
- Weekly consolidation + pruning mechanics
**Deliverable:** self-improving adapter ecosystem

### Phase 4 — Decade-Scale Lifecycle (ongoing)
- Multi-domain adapters
- Era snapshots
- Long-term drift detection
- Profile export/import + enterprise policies
**Deliverable:** “lives with you for years” stability

---

## 12) Scope Boundaries (What’s In / Out)
### In Scope
- Local, private, user-owned learning
- Adapter-based continual learning (LoRA)
- Idle reflection + Darwinian selection
- Strong eval/rollback

### Out of Scope (initially)
- Autonomous high-stakes actions (email, purchases) without approval
- Always-on GPU training while user is active
- Arbitrary “any model on earth” support without artifacts

---

## 13) Success Criteria
DATAFORM is “working” when:
- After a few weeks, its voice and reasoning noticeably align with the user
- It asks fewer “why” questions over time *because it truly learned*
- It can adapt when the user changes (detects novelty/contradiction)
- Updates never feel like regression thanks to eval gates + rollback
- The user never manages training—only occasionally provides feedback

---

## 14) Next Technical Decisions (Lock These Early)
1. Tier 1 base model families to support at launch (e.g., Qwen2.5, DeepSeek)
2. Execution providers (CPU-only baseline? CUDA EP optional/required?)
3. Adapter packaging format compatible with your Ollama + Modelfile workflow
4. Eval suite initial set (what “User-ness” means quantitatively)

---

# Appendix A — Default Idle Learning Policy (Recommended)
- Run reflection every idle window (cheap)
- Run evolution training:
  - daily when plugged in and idle for > X minutes
  - weekly consolidation run
- Budget caps:
  - ~10% GPU/CPU target
  - pause instantly on user activity
- Always evaluate before promotion
- Always keep last 3 stable adapter versions for rollback

---

# Appendix B — Minimal Data Schema (Starter)
## Episodic table (fields)
- id, timestamp, topic, user_text, assistant_text, inquiry_text
- user_feedback (thumb), user_edit_text, edit_diff
- outcome flags (reasked, accepted, corrected)
- tags, model_id, adapter_version

## Traits table (fields)
- trait_id, type, statement, confidence
- evidence_episode_ids (array)
- last_confirmed_ts, created_ts, updated_ts
