# Generative Agents: Interactive Simulacra of Human Behavior -- Curated Technical Extraction

**Source:** Park, O'Brien, Cai, Morris, Liang, Bernstein — Stanford / Google Research, UIST 2023
**Scale:** 25 agents in "Smallville" sandbox environment, 2 game-day simulation
**Base LM:** GPT-3.5-turbo (ChatGPT)
**Environment Framework:** Phaser web game development framework

---

## 1. Architecture Overview

Three core components:
1. **Memory Stream** — comprehensive record of agent experiences
2. **Retrieval Function** — surfaces relevant memories given current context
3. **Higher-Level Mechanisms** — reflection + planning, both built on retrieval

All components produce natural language records stored in a unified memory stream.

---

## 2. Memory Stream

### 2.1 Structure

A memory stream is a long-term storage list of **memory objects**, each containing:
- **Natural language description** of the event/observation
- **Creation timestamp** — game time when record was created
- **Most recent access timestamp** — game time of last retrieval

Memory objects include:
- Direct observations of the environment
- Agent's own behaviors
- Observations of other agents
- Reflections (higher-level synthesized thoughts)
- Plans (future action sequences)

### 2.2 Perception

At each time step, agents perceive the world. Perceived observations are stored as memory objects. Agents are NOT omniscient — they only perceive events and objects within their visual range (set as a parameter in the sandbox server).

---

## 3. Retrieval Function

### 3.1 Scoring Formula

Given a query (current situation, conversation topic, etc.), memories are scored:

```
score = α_recency × recency + α_importance × importance + α_relevance × relevance
```

**All α weights = 1** in the implementation (equal weighting).

All three component scores are **normalized to [0, 1]** via min-max scaling before combining.

### 3.2 Recency Score

Exponential decay function over sandbox game hours since last access:

```
recency(memory) = decay_factor ^ (hours_since_last_access)
```

- **Decay factor = 0.995** per game hour
- More recently accessed memories score higher
- Measured in sandbox game time, not real time

### 3.3 Importance Score

LLM-rated on a scale of **1 to 10**:
- **1** = mundane, everyday events (e.g., brushing teeth)
- **10** = extremely poignant, core memories (e.g., a breakup, career change)

Prompt template:
```
On the scale of 1 to 10, where 1 is purely mundane
(e.g., brushing teeth, making bed) and 10 is
extremely poignant (e.g., a break up, college
acceptance), rate the likely poignancy of the
following piece of memory.
Memory: <memory description>
Rating: <fill in>
```

### 3.4 Relevance Score

**Cosine similarity** between the embedding vectors of:
- The memory description
- The current query/context

Uses language model embeddings (text-embedding-ada-002 or equivalent).

### 3.5 Top-k Retrieval

After scoring all memories, the system retrieves the **top-k** highest-scoring memories to include in the agent's context.

---

## 4. Reflection

### 4.1 Purpose

Raw observations alone are insufficient for higher-level reasoning. Reflections enable agents to:
- Generalize from specific experiences
- Form abstract self-knowledge
- Make inferences about relationships and patterns
- Enable deeper synthesis for decision-making

### 4.2 Trigger Condition

Reflections are generated **periodically** when:

```
sum(importance_scores of latest perceived events) >= threshold
```

- **Threshold = 150** in implementation
- In practice: agents reflect roughly **2-3 times per game day**

### 4.3 Reflection Generation Process

**Step 1: Identify reflection topics**
- Input: 100 most recent records from memory stream
- Prompt: "Given only the information above, what are 3 most salient high-level questions we can answer about the subjects in the statements?"
- Output: candidate questions (e.g., "What topic is Klaus Mueller passionate about?")

**Step 2: Gather evidence**
- Use each generated question as a retrieval query
- Retrieve relevant memories (including prior reflections) for each question

**Step 3: Generate insights with citations**
- Prompt template:
```
Statements about Klaus Mueller
1. Klaus Mueller is writing a research paper
2. Klaus Mueller enjoys reading a book on gentrification
3. Klaus Mueller is conversing with Ayesha Khan about exercising [...]
What 5 high-level insights can you infer from the above statements?
(example format: insight (because of 1, 5, 3))
```
- Output: cited insights, e.g., "Klaus Mueller is dedicated to his research on gentrification (because of 1, 2, 8, 15)"

**Step 4: Store**
- Parse the insight statement
- Store as a reflection memory object in the memory stream
- Include pointers to the cited memory objects

### 4.4 Recursive Reflection Trees

Reflections can reference other reflections, forming **tree structures**:
- **Leaf nodes** = base observations from the environment
- **Non-leaf nodes** = reflections (increasingly abstract higher up)
- Enables multi-level abstraction from raw data to high-level self-knowledge

---

## 5. Planning and Reacting

### 5.1 Plan Structure

A plan entry contains:
- **Location** — where the action takes place
- **Starting time** — when the action begins
- **Duration** — how long the action lasts
- **Description** — natural language description of the activity

Example: "for 180 minutes from 9am, February 12th, 2023, at Oak Hill College Dorm: Klaus Mueller's room: desk, read and take notes for research paper"

Plans are stored in the memory stream and included in retrieval (agents consider observations, reflections, and plans together).

### 5.2 Hierarchical Plan Decomposition (Top-Down Recursive)

**Level 1: Day-level plan**
- Input: agent summary description + summary of previous day
- Output: 5-8 broad-stroke daily activities
- Prompt (incomplete for LLM completion):
```
Name: Eddy Lin (age: 19)
Innate traits: friendly, outgoing, hospitable
[agent background and yesterday's summary]
Today is Wednesday February 13. Here is Eddy's
plan today in broad strokes: 1)
```

**Level 2: Hour-level decomposition**
- Each broad activity → hour-long action chunks
- Example: "work on composition 1-5pm" → "1:00pm: brainstorm ideas... 4:00pm: take a break..."

**Level 3: 5-15 minute granular actions**
- Each hour chunk → 5-15 minute specific actions
- Example: "4:00pm: grab a snack. 4:05pm: take a short walk. 4:50pm: clean workspace."

Granularity is adjustable. Only near-future actions are decomposed in detail (lazy evaluation).

### 5.3 Reacting and Updating Plans

At each time step in the action loop:
1. Agent perceives the world → observations stored in memory
2. LLM prompted to decide: **continue existing plan** or **react to new observation**
3. If reacting: regenerate plan from the point of reaction onward

Reaction prompt template:
```
[Agent's Summary Description]
It is February 13, 2023, 4:56 pm.
John Lin's status: John is back home early from work.
Observation: John saw Eddy taking a short walk around his workplace.
Summary of relevant context from John's memory:
[retrieved memories about the observed entity]
Should John react to the observation, and if so,
what would be an appropriate reaction?
```

Context summary generation uses two retrieval queries:
1. "What is [observer]'s relationship with the [observed entity]?"
2. "[Observed entity] is [action status of the observed entity]"

### 5.4 Dialogue Generation

When interaction between agents is indicated:

**Initiator's first utterance:**
- Conditioned on: summarized memory about the other agent + intended reaction
- Prompt includes: agent description, status, observation, memory context, reaction intent

**Respondent's reply:**
- The initiator's dialogue is perceived as an event
- Respondent retrieves memories about relationship + topic
- Prompt includes: agent description, status, observation, memory context, dialogue history

**Continuation:**
- Same mechanism iterates back and forth
- Continues until one agent decides to end the dialogue

---

## 6. Environment Representation

### 6.1 World as Tree Data Structure

```
World
├── Area 1 (e.g., "The Lin family's house")
│   ├── Subarea (e.g., "kitchen")
│   │   ├── Object (e.g., "stove")
│   │   └── Object (e.g., "refrigerator")
│   ├── Subarea (e.g., "bedroom")
│   └── ...
├── Area 2 (e.g., "Hobbs Cafe")
│   └── ...
└── ...
```

- Edges = containment relationships
- Converted to natural language: "there is a stove in the kitchen"
- Each agent maintains a **personal subgraph** of the environment tree
- Initialized with: living quarters, workplace, commonly visited locations
- Updated as agents explore new areas
- **Not omniscient**: agent's tree becomes stale when they leave an area

### 6.2 Location Selection (Recursive Tree Traversal)

To determine where to perform an action:
1. Start at root of agent's environment tree
2. Prompt LLM to select the most suitable area from known areas
3. Recursively descend into subareas
4. Continue until reaching a leaf node (specific object/location)
5. Use traditional game pathfinding algorithms for movement animation

Prompt includes preference: "Prefer to stay in the current area if the activity can be done there."

### 6.3 Object State Updates

When agent performs action on object:
- Prompt LLM: what happens to the object's state?
- Example: action "making espresso" → coffee machine state changes from "off" to "brewing coffee"
- Object states are visible to other agents who perceive the area

---

## 7. Sandbox Implementation

### 7.1 Server Architecture

- **Phaser web game framework** for visual rendering
- Server maintains JSON data structure per agent: current location, action description, interacting object
- Each time step:
  1. Server parses JSON for agent changes
  2. Moves agents to new positions
  3. Updates sandbox object states
  4. Sends visible agents/objects to each agent (within visual range)
  5. Agent produces output action → updates JSON
  6. Loop

### 7.2 Agent Initialization

Each agent initialized with:
- Brief natural language description (semicolon-delimited characteristics)
- Split into individual memories as initial memory stream entries
- These seed memories determine initial behavior
- As simulation progresses, accumulated experience overtakes initial seeds

### 7.3 Performance

- Simulation runs in roughly **real-time game time** (1 second real = 1 minute game time)
- Currently sequential; parallelizable
- Cost: thousands of dollars in token credits for 25 agents over 2 game days

---

## 8. Agent Summary Description Cache

Frequently used in prompts as `[Agent's Summary Description]`. Synthesized at regular intervals and cached:

**Three parallel retrieval queries:**
1. "[name]'s core characteristics"
2. "[name]'s current daily occupation"
3. "[name]'s feeling about his recent progress in life"

Each query retrieves relevant memories → LLM summarizes → concatenated with name, age, traits.

Example output: "Eddy Lin is a student at Oak Hill College studying music theory and composition. He loves to explore different musical styles and is always looking for ways to expand his knowledge."

---

## 9. Evaluation Results

### 9.1 Controlled Evaluation (Individual Agent Interviews)

**Method:** "Interview" agents with 25 questions across 5 categories:
1. Self-knowledge (5 questions)
2. Memory retrieval (5 questions)
3. Plan generation (5 questions)
4. Reaction to events (5 questions)
5. Reflection (5 questions)

**Conditions compared:**
| Condition | TrueSkill μ | TrueSkill σ |
|-----------|------------|------------|
| Full architecture | 29.89 | 0.72 |
| No reflection | 26.88 | 0.69 |
| No reflection, no planning | 25.64 | 0.68 |
| Human crowdworker | 22.95 | 0.69 |
| No observation, no reflection, no planning | 21.21 | 0.70 |

- **Full architecture beats all ablations and human crowdworkers**
- Effect size vs. prior state-of-art (no memory/planning/reflection): **Cohen's d = 8.16** (8 standard deviations)
- Kruskal-Wallis test: H(4) = 150.29, p < 0.001
- All pairwise differences significant (p < 0.001) except crowdworker vs. fully ablated

**100 evaluators** (within-subjects design): recruited from Prolific, ~30 minutes each, $15/hour

### 9.2 Emergent Social Behaviors (End-to-End, 2 Game Days)

**Information Diffusion:**
- Sam's mayoral candidacy: known by 1 agent (4%) → 8 agents (32%)
- Isabella's Valentine's Day party: known by 1 agent (4%) → 13 agents (52%)
- No hallucinated knowledge confirmed (all traced to actual dialogue)

**Relationship Formation:**
- Network density: 0.167 → 0.74 over two game days
- Formula: η = 2|E| / |V|(|V|−1)
- Hallucination rate in relationship claims: 1.3% (6 out of 453 responses)

**Coordination (Valentine's Day Party):**
- Isabella autonomously invited guests, gathered materials, enlisted help to decorate
- 12 agents heard about the party
- 5 out of 12 showed up at Hobbs Cafe on Feb 14 at 5pm
- 3 non-attendees cited specific scheduling conflicts
- 4 expressed interest but didn't plan to attend

---

## 10. Identified Failure Modes

1. **Location selection degradation:** As agents learn more locations, they sometimes choose atypical venues (e.g., going to a bar for lunch instead of cafe)

2. **Physical norm misclassification:** Agents don't understand implicit spatial rules conveyed poorly in natural language (e.g., single-occupancy bathroom treated as multi-person; entering shops after closing time)

3. **Instruction tuning artifacts:**
   - Overly formal dialogue (spouses greeting each other formally)
   - Excessive agreeableness (agents rarely refuse suggestions)
   - Others' suggestions reshape agent's own interests over time

4. **Memory retrieval failures:** Agents sometimes retrieve incomplete memory fragments (e.g., remembering what to do at a party but not that the party exists)

5. **Hallucinated embellishments:** Agents add plausible but ungrounded details (e.g., "he's going to make an announcement tomorrow" when no such plan was discussed)

---

## 11. Hyperparameters Summary

| Parameter | Value |
|-----------|-------|
| Number of agents | 25 |
| Base LLM | gpt-3.5-turbo |
| Retrieval weights (α_recency, α_importance, α_relevance) | All = 1 |
| Recency decay factor | 0.995 per game hour |
| Importance scale | 1-10 (LLM-rated) |
| Relevance metric | Cosine similarity of embeddings |
| Score normalization | Min-max to [0,1] |
| Reflection trigger threshold | Sum of importance ≥ 150 |
| Reflection frequency | ~2-3 per game day |
| Reflection questions generated | 3 salient questions per trigger |
| Reflection insights per query | 5 high-level insights |
| Retrieval input for reflection | 100 most recent memory records |
| Plan granularity levels | 3 (day → hour → 5-15 min) |
| Day-level plan chunks | 5-8 activities |
| Finest action granularity | 5-15 minutes |
| Time scale | 1 real second = 1 game minute |
| Simulation duration | 2 game days |
| Evaluation participants | 100 human evaluators |
| Evaluator pay rate | $15.00/hour |
| Evaluation session duration | ~30 minutes |

---

## 12. Key Architectural Insights

1. **Unified memory stream** (observations + reflections + plans) enables coherent retrieval across all cognitive functions
2. **Three-factor retrieval** (recency × importance × relevance) balances temporal, emotional, and contextual salience
3. **Recursive reflection** builds abstraction hierarchies from raw observations → supports generalization
4. **Hierarchical planning** with lazy decomposition prevents temporal inconsistency (no repeated lunch)
5. **Reactive replanning** allows agents to respond to unexpected events while maintaining overall coherence
6. **Natural language as universal representation** — all reasoning, memory, and communication uses text
7. **Tree-based environment grounding** bridges natural language reasoning to spatial/physical world
8. **Each component is additive**: removing any one (observation, reflection, planning) measurably degrades believability
