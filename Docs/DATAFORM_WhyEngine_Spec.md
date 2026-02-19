# DATAFORM Why Engine Spec

## Purpose
The Why Engine is DATAFORM's curiosity controller - it decides WHEN and WHAT to ask to understand the user's motivations, not just their questions.

## Trigger Rules

### Strong Triggers (always fire if within budget)
- **Emotional content**: User expresses feelings (frustrated, worried, excited, etc.)
- **Decision-related**: User is weighing options (should I, which one, pros and cons, etc.)

### Moderate Triggers (probabilistic)
- **Topic shift** (70% chance): User changes subjects from recent conversation
- **Novel topic** (50% chance): No prior episodes match the user's keywords

### Low-frequency Triggers
- **Spot check** (base probability * curiosity level): After 8+ turns without inquiry

## Budget / Cooldown Rules
- Minimum 3 turns between inquiries
- Maximum 2 consecutive inquiries
- At least 30 seconds between inquiries
- Never ask while awaiting response to previous inquiry
- Short messages (< 10 chars) are never inquired about

## Inquiry Templates

### Topic Shift
- "That's a shift from what we were discussing. What prompted this?"
- "Interesting change of direction. What's on your mind?"
- "What made you think of this just now?"

### Emotional Content
- "It sounds like this matters to you. What's driving that feeling?"
- "I want to understand - what's behind this for you?"
- "What outcome here would feel like success to you?"

### Decision-Related
- "What's most important to you in making this decision?"
- "Are you looking for a solution, or working through your thinking?"
- "What would make you confident in whichever path you choose?"

### Novel Topic
- "This is new territory for us. What's your experience with this?"
- "I'd like to understand your perspective on this. What draws you to it?"
- "Before I respond - what are you hoping to get out of exploring this?"

### Spot Check
- "Just checking - is my approach working for you, or would you prefer something different?"
- "Am I understanding what you need here?"
- "Is there context I'm missing that would help me help you better?"

## Curiosity Level (Adaptive)
- Starts at 0.5
- More traits learned = lower curiosity (we already know a lot)
- Early conversations (< 20 episodes) = higher curiosity
- Never below 0.15 (always maintain some curiosity)
- Formula: `traitFactor * noviceFactor`, clamped to [0.15, 1.0]

## Integration with Orchestrator
1. Before sending to LLM, `buildCuriosityDirective()` is called
2. If triggered, a CURIOSITY DIRECTIVE is appended to the system prompt
3. LLM is instructed to ask ONE question naturally at the end of its response
4. After response, `detectInquiryInResponse()` checks for question marks
5. If inquiry detected, it's recorded and the engine enters "awaiting response" state
6. When user's next message appears to answer (long response, explanatory patterns), trait extraction is triggered
