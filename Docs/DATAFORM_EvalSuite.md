# DATAFORM Evaluation Suite

## Purpose
The EvalSuite tests identity alignment, curiosity quality, and system stability. It runs on-demand (via "Run Eval" button) and will be gated into the adapter promotion pipeline in Phase 3.

## Test Categories

### Identity Alignment (4 tests)
1. **Trait Consistency** - Checks for contradictions between traits (e.g., "values brevity" vs "prefers verbose"). Uses keyword contradiction pairs.
2. **Trait Coverage** - Expects at least 1 trait per 10 episodes, checks type diversity (values, preferences, policies, motivations).
3. **Confidence Distribution** - Checks that confidence scores are spread out (mean ~0.5, some variance), not all clustered at one level.
4. **Evidence Quality** - Verifies traits have episode evidence backing them, average evidence count per trait.

### Curiosity Quality (3 tests)
5. **Inquiry Frequency** - Target: 15-35% of interactions include an inquiry. Too few = not learning, too many = interrogating.
6. **Inquiry Response Capture** - Checks that inquiries receive substantive responses (user messages > 30 chars after an inquiry).
7. **Topic Diversity** - Measures unique topics across recent episodes. More diversity = broader understanding.

### Stability (3 tests)
8. **Memory Growth** - Confirms episodes are accumulating (basic health check).
9. **No Trait Drift** - High-confidence traits (>0.7) should be stable and confirmed.
10. **Feedback Balance** - At least 50% of rated interactions are positive.

## Scoring
- Each test returns: passed (bool), score (0.0-1.0), details (string)
- Overall score = average of all test scores
- Report includes: total tests, passed count, overall score percentage

## Promotion Rules (Phase 3)
- Candidate adapter promoted only if:
  - All safety tests pass
  - Overall score >= 0.6
  - No regression from previous champion score
  - Individual identity tests all >= 0.5
