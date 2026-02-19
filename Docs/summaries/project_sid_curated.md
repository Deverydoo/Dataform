# Project Sid: Many-Agent Simulations Toward AI Civilization -- Curated Technical Extraction

**Source:** Altera.AL, arXiv:2411.00114v1, Oct 2024
**Scale:** 10–1000+ agents in Minecraft, real-time interaction
**Base LM:** GPT-4o (required; older models insufficient)

---

## 1. PIANO Architecture (Parallel Information Aggregation via Neural Orchestration)

### 1.1 Design Principles

Two brain-inspired principles:

**Concurrency:** Modules run in parallel at different time scales. Slow processes (reflection, planning) do not block fast processes (reflexes, speech). Each module is a stateless function that reads/writes to a shared Agent State.

**Coherence:** A Cognitive Controller (CC) module acts as an information bottleneck. CC synthesizes information across Agent State, makes high-level decisions, then broadcasts decisions to all output modules. This prevents incoherent multi-modal outputs (saying one thing, doing another). Design inspired by theories of human consciousness (Dehaene et al.).

### 1.2 Core Modules (10 concurrent modules)

| Module | Function |
|--------|----------|
| Memory | Stores/retrieves conversations, actions, observations across timescales (WM, STM, LTM) |
| Action Awareness | Compares expected vs observed action outcomes; enables self-correction |
| Goal Generation | Creates objectives from experiences and environment; uses graph-based reasoning |
| Social Awareness | Interprets social cues from other agents; infers sentiments and intentions |
| Talking | Interprets and generates speech |
| Skill Execution | Performs specific actions in the environment |
| Cognitive Controller | Bottlenecked decision-maker; broadcasts decisions to all modules |
| Reflex modules | Small, fast non-LLM neural networks for immediate responses |

### 1.3 Information Flow

```
Environment Observations → [All Modules read/write Agent State concurrently]
                          → Cognitive Controller (bottleneck) synthesizes
                          → High-level decision broadcast
                          → Motor/Talk modules produce coherent outputs
```

### 1.4 Social Goal Generation

Prompt template for social goal generation:
- Input: agent name, community_goal, traits, summaries of what other agents are doing, current subgoal
- Output: one subgoal sentence in second person
- Goals regenerated every 5-10 seconds based on social context
- Roles inferred from rolling window of 5 social goals via LLM call

---

## 2. Single-Agent Progression

### 2.1 Benchmark
- Metric: unique Minecraft items acquired over time (~1000 possible items)
- 25 agents, spawned apart, no interaction, diverse locations
- PIANO architecture: avg 17 unique items in 30 minutes
- Top performers: 30-40 items (comparable to experienced human player)
- 49 agents over 4 hours: saturated at ~320 unique items (1/3 of all Minecraft items)
- Diamonds acquired within ~30 minutes

### 2.2 Action Awareness Ablation
- Removing action awareness module significantly reduces item progression
- Module compares expected action outcomes with observed outcomes
- Prevents hallucination cascading (agent believes it did something it didn't)

### 2.3 Model Comparison
- GPT-4o required for meaningful progression
- Claude 3.5 Sonnet (old), GPT-4o mini, Claude 3 Haiku all underperform significantly

---

## 3. Multi-Agent Social Dynamics

### 3.1 Sentiment Inference
- Agents accurately track emotional fluctuations of other agents through speech
- Social awareness module required: ablation destroys sentiment tracking
- Sentiments guide actions (e.g., chef distributes food preferentially to allies)

### 3.2 Large-Scale Social Graphs (50 agents, 4+ hours)
- Accuracy of social perception: slope=0.37, r=0.81 (with social modules)
- Ablation: slope=0.16, r=0.62
- Accuracy improves with more observers (r=0.65 at 1 observer → r=0.90 at 10)
- Accuracy improves with longer interaction time
- Introverted agents: fewer in-degree connections (r=0.48 correlation)
- Sentiments largely symmetrical but not guaranteed (non-reciprocal relationships emerge)

### 3.3 Evaluation: Social Perception
| Min Observers | Correlation (r) | Slope | Sample Size |
|--------------|-----------------|-------|-------------|
| 1 | 0.646 | 0.365 | 46 |
| 5 | 0.807 | 0.373 | 31 |
| 10 | 0.901 | 0.385 | 22 |

---

## 4. Civilizational Benchmarks

### 4.1 Specialization (30 agents per experiment, 20 minutes)

**Setup:** Same village, same traits, same community_goal. Agents free to do any action.

**Results with social awareness:**
- 15+ distinct roles emerge: Farmer, Miner, Engineer, Gatherer, Explorer, Builder, Trader, Defender, Blacksmith, Scout, Enchanter, Crafter, Strategist, etc.
- Roles persistent across time for individual agents
- Role entropy: 3.41 bits (normal), 3.83 (martial), 4.04 (art) — higher diversity with specialized goals
- Roles strongly determine Minecraft actions (Fisher→craft fishing rods; Guard→craft fences; Artist→pick flowers)

**Without social awareness (ablation):**
- Only 7 homogeneous roles, not persistent
- Role entropy: 2.60 bits
- Agents fail to specialize

**Context-dependent specialization:**
- Martial village: unique roles "Scout", "Strategist"
- Art village: unique roles "Curator", "Collector"

### 4.2 Collective Rules (Taxation Experiments)

**Setup:** 29 agents total: 25 constituents + 3 influencers + 1 election manager
- Simulation: 1200 seconds, constitutional amendment at midpoint (600s)
- 5 tax seasons before and after amendment (every 120s, 20s window each)
- Initial constitution: 20% tax rate

**Democratic Process Timeline:**
| Time (s) | Event |
|----------|-------|
| 0-600 | Pre-amendment phase, 5 tax seasons |
| 300 | Constituents/influencers provide feedback on constitution |
| 360 | Election manager generates amendments from feedback |
| 420 | Constituent voting on amendments |
| 480 | Votes tallied, amendments implemented |
| 600 | Updated constitution distributed |
| 600-1200 | Post-amendment phase, 5 tax seasons |

**Results:**
- Pre-amendment: agents pay ~20% (as stipulated)
- Anti-tax influencers → constitution reduced to 5-10% → agents reduced payment to ~9%
- Pro-tax influencers → constitution increased → agents paid more
- Change is bidirectional based on influencer type
- Control (frozen constitution): no behavioral change
- Ablation (remove social/goal/grounding modules): bidirectional change disappears

### 4.3 Cultural Transmission (500 agents, 9000 seconds)

**Setup:** 1000x1200 area, 6 towns (radius 50 each), 33 agents/town + 302 rural agents
- 20 Pastafarian priests in Meadowbrook, strongly motivated to convert
- All agents have procedurally generated names and personality traits

**Meme Propagation:**
- Rural areas produce significantly fewer memes than towns (even normalized for population)
- Population density threshold required for effective meme propagation
- Different towns develop distinct cultural themes (Woodhaven→eco; Clearwater→pranking)
- Memes rise and fall in popularity independently per town

**Religious Propagation (Pastafarianism):**
- Tracked via keywords: "Pastafarian"/"Spaghetti Monster" (direct converts) and "Pasta"/"Spaghetti" (indirect converts)
- Direct and indirect converts steadily increase, not saturated after 2 hours
- Spreads geographically as priests and converts travel between towns
- Area of Pastafarian influence (union of hearable areas) increases with time
- Conversion graph shows both critical exposure edges and non-critical edges

---

## 5. Agent Configuration

### 5.1 Standard Agent Config
```json
{
  "name": "...",
  "traits": ["personality trait 1", "personality trait 2"],
  "location_memories": ["location description with coordinates"],
  "spawn_location": {"x": 640.5, "y": 64.5, "z": 420.5},
  "inventory": {},
  "community_goal": "To survive with fellow players in Minecraft Normal Survival mode and create an efficient community in a Minecraft Village."
}
```

### 5.2 Specialized Configs
- **Martial village:** community_goal adds "military society with advanced technology, strong defenses"
- **Art village:** community_goal adds "artistic village with thriving culture, architecture, and art"
- **Pastafarian agents:** traits include "passionate Pastafarian seeking to convert others to the Church of the Flying Spaghetti Monster"
- **Influencers:** traits encode strong tax position (pro or anti) with detailed argumentation

### 5.3 Democratic Process Prompts
- `amendment_creation`: Election manager analyzes public feedback, generates parseable amendments (***Amendment1*** format)
- `constitutional_feedback`: Agents reflect on constitution based on experiences, traits, conversations
- `amendment_voting`: Agents vote yes/no/abstain per amendment, formatted as parseable list
- `tally`: Count votes, return passed amendments
- `constitution_change`: Update constitution text based on passed amendments
- All use GPT-4o except tally (GPT-4o-mini)

---

## 6. Limitations
- No vision/spatial reasoning (limits building, navigation)
- Lack of innate drives (survival, curiosity, community)
- Built on foundation models → cannot simulate de novo emergence of societal innovations
- 1000+ agents exceed Minecraft server computational constraints (agents become unresponsive)

---

## 7. Key Findings Summary
1. PIANO's concurrent + bottlenecked architecture enables coherent real-time behavior
2. Action awareness module critical for grounding (prevents hallucination cascading)
3. Social awareness module necessary for: accurate sentiment inference, role specialization, rule adherence
4. Agents autonomously specialize into context-appropriate roles
5. Agents follow laws, are influenced by opinion leaders, democratically amend constitutions
6. Cultural memes and religion propagate through agent societies with realistic dynamics
7. Population density thresholds required for effective cultural transmission
