# ChatDev: Communicative Agents for Software Development -- Curated Technical Extraction

**Source:** Chen Qian, Wei Liu, et al. — Tsinghua University / U. Sydney / BUPT / Modelbest Inc., ACL 2024
**arXiv:** 2307.07924v5, June 2024
**Base LM:** ChatGPT-3.5 (temperature = 0.2)
**Runtime Environment:** Python 3.11.4

---

## 1. Framework Overview

ChatDev is a **chat-powered software development framework** where multiple LLM-driven agents with specialized roles (CEO, CTO, programmer, reviewer, tester) collaborate through the full software lifecycle via structured multi-turn dialogues.

Two core technical contributions:
1. **Chat Chain** — guides *what* agents communicate (task decomposition + workflow)
2. **Communicative Dehallucination** — guides *how* agents communicate (reduces coding hallucinations)

---

## 2. Chat Chain Architecture

### 2.1 Formal Definition

The software development process is organized as a sequential chain:

```
C = ⟨P₁, P₂, ..., P|C|⟩           (chain of phases)
Pᵢ = ⟨T₁, T₂, ..., T|Pᵢ|⟩         (phase of subtasks)
Tⱼ = τ(C(I, A))                    (subtask solution extracted from dialogue)
C(I, A) = ⟨I → A, A ⇝ I⟩⟲         (multi-turn instructor-assistant dialogue)
```

Where:
- `C` = chat chain (full workflow)
- `Pᵢ` = phase (design, coding, testing)
- `Tⱼ` = subtask within a phase
- `I` = instructor agent, `A` = assistant agent
- `→` = instruction, `⇝` = response
- `τ` = solution extraction function
- `⟲` = multi-turn loop

### 2.2 Three Sequential Phases, Five Subtasks

| Phase | Subtask | Instructor | Assistant | Output |
|-------|---------|-----------|-----------|--------|
| **Design** | System Design | CEO | CTO | Ideas, architecture |
| **Coding** | Code Writing | CTO | Programmer | Initial source code |
| **Coding** | Code Completion | CTO | Programmer | Completed source code |
| **Testing** | Code Review (static) | Reviewer | Programmer | Bug fixes |
| **Testing** | System Testing (dynamic) | Tester | Programmer | Runtime fixes |

### 2.3 Dual-Agent Communication Pattern

Each subtask uses exactly **two agents** (instructor + assistant):
- Simplifies communication topology vs. multi-agent free-form
- Streamlines consensus-reaching process
- Solutions from previous subtasks bridge to the next phase

---

## 3. Agentization (Inception Prompting)

### 3.1 Agent Instantiation

Agents are created by "hypnotizing" LLMs with role-specific system prompts:

```
I = ρ(LLM, Pᵢ)    (instructor)
A = ρ(LLM, Pₐ)    (assistant)
```

Where `ρ` = role customization operation via system message assignment.

### 3.2 System Prompt Components (Pᵢ and Pₐ)

System prompts for both instructor and assistant are **mostly symmetrical** and include:
- Overview and objectives of the current subtask
- Specialized role descriptions
- Accessible external tools
- Communication protocols
- Termination conditions
- Constraints to avoid undesirable behaviors

### 3.3 Termination Conditions

A subtask terminates when either:
- **Two unchanged code modifications** in succession (convergence)
- **10 rounds of communication** reached (max iterations)

The assistant marks consensus with `<SOLUTION>` prefix in its response.

---

## 4. Memory System

### 4.1 Short-Term Memory (Within Phase)

Records current-phase utterances for context-aware decision-making:

```
Mᵢₜ = ⟨(Iᵢ₁, Aᵢ₁), (Iᵢ₂, Aᵢ₂), ..., (Iᵢₜ, Aᵢₜ)⟩
```

Iterative update:
```
Iᵢₜ₊₁ = I(Mᵢₜ)              (instructor generates next instruction from memory)
Aᵢₜ₊₁ = A(Mᵢₜ, Iᵢₜ₊₁)      (assistant generates response from memory + instruction)
Mᵢₜ₊₁ = Mᵢₜ ∪ (Iᵢₜ₊₁, Aᵢₜ₊₁)  (memory accumulates)
```

Continues until |Mᵢ| reaches upper limit (max communication rounds).

### 4.2 Long-Term Memory (Cross-Phase)

Only **solutions** (not full dialogues) are transmitted between phases:

```
M̃ᵢ = ∪ⱼ₌₁ⁱ τ(Mʲ|Mʲ|)       (union of extracted solutions from all prior phases)
Iᵢ⁺¹₁ = M̃ᵢ ∪ Pᵢ₊₁ᵢ          (next phase starts with prior solutions + phase prompt)
```

Benefits:
- Minimizes information overload
- Maintains cross-phase context continuity
- Keeps agents focused on current subtask

---

## 5. Communicative Dehallucination

### 5.1 Problem: Coding Hallucinations

LLMs produce code that is:
- Incomplete (placeholder implementations)
- Unexecutable (missing imports, syntax errors)
- Inconsistent with requirements

Root cause: vague/general instructions require multiple adjustments.

### 5.2 Mechanism: Deliberate Role Reversal

**Vanilla communication:**
```
⟨I → A, A ⇝ I⟩⟲
```

**Communicative dehallucination (nested loop):**
```
⟨I → A, ⟨A → I, I ⇝ A⟩⟲, A ⇝ I⟩⟲
```

The assistant **temporarily becomes an instructor**, proactively requesting specific details (e.g., exact dependency names, class specifications) before delivering a response. After receiving specifics, the assistant performs precise optimization.

### 5.3 Properties

- Tackles one concrete issue at a time
- Requires multiple rounds for multiple issues
- Enables finer-grained information exchange
- Activated during: code completion, code review, and system testing subtasks

---

## 6. Agent Roles

| Role | Function | Key Behaviors |
|------|----------|---------------|
| **CEO** | Requirements analysis | Initiates design discussions, defines software scope |
| **CTO** | Technical architecture | Determines technology choices, guides implementation |
| **Programmer** | Code implementation | Writes code, responds to review/test feedback, GUI design |
| **Reviewer** | Static code analysis | Identifies bugs, missing implementations, code smells |
| **Tester** | Dynamic testing | Runs compilation, identifies runtime errors, validates execution |

Role assignment significantly impacts output quality — removing roles from system prompts causes the largest performance degradation in ablation studies.

---

## 7. Evaluation Metrics

### 7.1 Metric Definitions

| Metric | Definition | Measurement |
|--------|-----------|-------------|
| **Completeness** | Ability to fulfill code completion | % of software without placeholder code snippets |
| **Executability** | Ability to compile and run | % of software that compiles and runs directly |
| **Consistency** | Alignment with requirements | Cosine distance between embeddings of requirement text and generated code |
| **Quality** | Holistic assessment | Completeness × Executability × Consistency |

Note: comments excluded from code before consistency evaluation to prevent information leakage.

### 7.2 Performance Comparison

| Method | Paradigm | Completeness | Executability | Consistency | Quality |
|--------|----------|-------------|--------------|-------------|---------|
| GPT-Engineer | Single-agent | 0.5022 | 0.3583 | 0.7887 | 0.1419 |
| MetaGPT | Multi-agent | 0.4834 | 0.4145 | 0.7601 | 0.1523 |
| **ChatDev** | **Multi-agent** | **0.5600** | **0.8800** | **0.8021** | **0.3953** |

All differences statistically significant (p ≤ 0.05).

### 7.3 Pairwise Evaluation (Human + GPT-4)

| Comparison | Evaluator | Baseline Wins | ChatDev Wins | Draw |
|-----------|-----------|--------------|-------------|------|
| vs GPT-Engineer | GPT-4 | 22.50% | 77.08% | 0.42% |
| vs GPT-Engineer | Human | 9.18% | 90.16% | 0.66% |
| vs MetaGPT | GPT-4 | 37.50% | 57.08% | 5.42% |
| vs MetaGPT | Human | 7.92% | 88.00% | 4.08% |

### 7.4 Software Statistics

| Method | Duration (s) | Tokens | Files | Lines of Code |
|--------|-------------|--------|-------|---------------|
| GPT-Engineer | 15.6 | 7,182.5 | 3.9 | 70.2 |
| MetaGPT | 154.0 | 29,278.7 | 4.4 | 153.3 |
| **ChatDev** | **148.2** | **22,949.4** | **4.4** | **144.3** |

Multi-agent methods: slower, more tokens, but larger codebases with more functionality.

---

## 8. Ablation Study

| Variant | Completeness | Executability | Consistency | Quality |
|---------|-------------|--------------|-------------|---------|
| **Full ChatDev** | **0.5600** | **0.8800** | **0.8021** | **0.3953** |
| ≤ Coding (stop after coding) | 0.4100 | 0.7700 | 0.7958 | 0.2512 |
| ≤ Complete (stop after completion) | 0.6250 | 0.7400 | 0.7978 | 0.3690 |
| ≤ Review (stop after review) | 0.5750 | 0.8100 | 0.7980 | 0.3717 |
| ≤ Testing (full pipeline) | 0.5600 | 0.8800 | 0.8021 | 0.3953 |
| ⧹CDH (remove dehallucination) | 0.4700 | 0.8400 | 0.7983 | 0.3094 |
| ⧹Roles (remove all role assignments) | 0.5400 | 0.5800 | 0.7385 | 0.2212 |

Key findings:
- **Code completion phase** most improves Completeness
- **Testing** is critical for Executability
- **Quality rises progressively** with each phase
- **Removing roles** has the largest single impact (Quality: 0.3953 → 0.2212)
- **Removing dehallucination** decreases all metrics

---

## 9. Communication Analysis

### 9.1 Language Distribution

Overall communication breakdown:
- **57.20% natural language** (design phase dominates)
- **42.80% programming language** (coding, testing phases)

Design phase natural language topics:
| Topic | % of Design Communication |
|-------|--------------------------|
| Target User | 21.44% |
| UI & UX | 20.55% |
| Data Management | 19.23% |
| Customization | 10.19% |
| Performance | 10.19% |
| Integration | 7.78% |
| Real-Time Update | 6.93% |
| Recommendation | 5.92% |
| Platform | 5.41% |
| Collaboration | 3.46% |
| Security & Privacy | 3.15% |
| Scalability & Maintenance | 2.51% |

### 9.2 Static Debugging (Code Review)

Most common reviewer suggestions (multi-round process):
1. **Method Not Implemented** — 34.85% of review discussions
2. **Modules Not Imported**
3. **Missing Code Segments**
4. **Not Configure Layout**
5. **Missing Comments**

Each review round can transform issues into different issues or "No Further Suggestions" (success indicator). Increasing proportion of "No Further Suggestions" indicates optimization progress.

### 9.3 Dynamic Debugging (System Testing)

Most frequent compilation errors:
1. **ModuleNotFoundError** — 45.76%
2. **NameError** — 15.25%
3. **ImportError** — 15.25%
4. Other errors: AttributeError, TypeError, SyntaxError, ValueError, etc.

Multi-round testing dynamics:
- Probability of compilation success at each step generally exceeds error probability
- Errors tend to persist between rounds (same type) rather than transform
- Successful compilation rarely regresses to error state
- Over time, consistent decrease in errors → convergence toward successful execution

---

## 10. Dataset: SRDD (Software Requirement Description Dataset)

- **1,200 software task prompts**
- **5 main categories:** Education, Work, Life, Game, Creation
- **40 subcategories** (8 per main category)
- **30 unique task prompts** per subcategory
- Sources: Ubuntu, Google Play, Microsoft Store, Apple Store
- Generation: LLM-based automatic generation + human post-processing refinement

---

## 11. Emergent Behaviors

Agents autonomously exhibit behaviors not explicitly programmed:
- **Functional enhancements:** adding GUI implementations not in requirements
- **Increased game difficulty** when developing games
- **Feature additions** beyond specified requirements
- **Self-organizing debugging:** reviewer discovers issues → programmer fixes → new issues surface → iterative optimization

---

## 12. Limitations

1. Agents implement **simple logic with low information density** — vague requirements lead to basic implementations
2. Currently suitable for **prototype systems**, not complex real-world applications
3. Multi-agent approach requires **more tokens and time** than single-agent
4. No standardized benchmark for evaluating general-purpose software generation
5. Without detailed requirements, agents default to **placeholder implementations** and static data

---

## 13. Key Architectural Insights

1. **Chat chain** decomposes complex software tasks into manageable dual-agent subtasks
2. **Dual-agent pattern** (instructor + assistant) is simpler and more effective than multi-agent free-form
3. **Communicative dehallucination** (role reversal for specificity) reduces coding errors
4. **Role assignment is critical** — the single most impactful component in ablation
5. **Natural language bridges design and code** — 57/43 split enables autonomous system design flowing into implementation
6. **Progressive quality improvement** through sequential phases validates waterfall decomposition
7. **Solution-only long-term memory** prevents information overload while maintaining context
8. **Multi-agent communication leads to emergent features** not present in single-agent approaches
