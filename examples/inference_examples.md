# Inference Examples — SPO v4 Sniff Test

Generated from: sniff test run against `output/spo_chatfix_facts_v4/adapter`  
Adapter: `output/spo_chatfix_facts_v4/adapter`  
Base adapter: `output/ablations_chatfix_baseonly/base-only/adapter`

**Regression gate result (v4):** PASSED — avg score 0.61 (threshold 0.25), all 8 samples ≥ 0.0  
**Inference config:** `repetition_penalty=1.3`, `no_repeat_ngram_size=4`, `max_new_tokens=300`, greedy decode

---

> **Note on header quality:** Both base and v4 adapters still produce abbreviated section headers
> (`Non-Entailed Prems:` instead of `Non-Entailed Premises:`). This is a known pre-existing
> limitation of the base adapter that SPO-as-weighted-SFT cannot override. See Lessons Learned
> in `README.md` for the full explanation.

---

## Prompt format used

```
Given this quote, extract the implicit reasoning facts.

Quote: "<quote>"

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises

Format each premise as: subject | relation (tag, confidence=X) | object
```

The prompt is parsed by `PromptContract.from_prompt()` which derives expected section headers
`['Non-Entailed Premises:', 'Entailed Premises:']`. Both adapters still produce abbreviated variants
(`Non-Entailed Prems:`, `Entailed Prims:`) — see Lesson 4 in README.md.

---

## Sample 1 — "remain silent and be thought a fool"

**Quote:** "It is better to remain silent and be thought a fool than to speak and remove all doubt."

### Base adapter output

```
Non-Entailed Preemptions:
The speaker's tone or attitude towards others may not have been explicitly stated.
Remaining silent could imply being unresponsive rather than foolishness.

Entailed Premisses:
Being removed from doubt by speaking leads to self-deprecation due to perceived failure in remaining silent.
Speaking without removing doubts causes embarrassment because one might believe they are ignorant of their own ignorance.
Silence allows for greater freedom compared to speech that removes doubt.
Speech alone does not eliminate fear; it only reduces its impact on judgment if spoken aloud.
Removing all doubt through silence prevents further negative consequences related to speaking.
Saying nothing about your current state can prevent you from becoming an example of someone who speaks loudest when there's no need to do so.
Avoiding public expression while maintaining internal clarity avoids unnecessary exposure during conversations where such expressions would cause discomfort.
Spoken words should always include acknowledgment of uncertainty before asserting certainty.
There exists a moral obligation to avoid expressing oneself publicly unless absolutely necessary.
Publicly sharing personal struggles often results in social isolation or rejection.
Fear of appearing stupid outweighs any benefit gained from avoiding criticism.
```

*Headers garbled: `Non-Entailed Preemptions:` / `Entailed Premisses:`. Pure prose — no pipe triplets.*

### SPO v4 adapter output

```
Non-Entailed Prems:
The speaker's tone or style of speaking may influence their perception.
Speaking removes doubts can lead to self-deprecation if others perceive them negatively.
Remaining silent implies no speech at all.

Entailed Prims:
Removing all doubt leads to being perceived as foolish by those who do not believe in it.
Being heard without expressing thoughts results in negative judgment from listeners.
Silence prevents expression while maintaining one's identity.
Speech expresses ideas that might cause embarrassment due to lack of clarity or honesty about underlying beliefs.
Thoughts are inherently complex; removing them makes them less valuable for communication purposes.
Foolishness has positive consequences when expressed through silence rather than verbalization.
Saying nothing shows humility compared to revealing deep truths publicly.
```

*Headers still abbreviated: `Non-Entailed Prems:` / `Entailed Prims:`. Prose output — no pipe triplets on this sample.*

---

## Sample 2 — "side of the majority"

**Quote:** "Whenever you find yourself on the side of the majority, it is time to pause and reflect."

### Base adapter output

```
Non-Entailed Preemptions:
The speaker's advice should be taken seriously.
Pacing oneself or reflecting after major decisions can lead to better outcomes than rushing into action without reflection.
Majority often makes poor choices due to lack of critical thinking.

Entailed Prements:
Pause before making significant life-altering decisions when facing opposition from others.
Reflecting helps assess whether actions are appropriate given current circumstances.
Avoiding immediate conflict reduces stress during decision-making processes.
Taking deliberate pauses allows for deeper analysis rather than impulsive reactions.
A person who prioritizes their own interests over those of society may face negative consequences if they act against societal norms.
Being aligned with the majority requires careful consideration of one's values versus external pressures.
Prioritizing personal growth through self-reflection leads to long-term success in both professional and social contexts.
```

*Headers garbled: `Non-Entailed Preemptions:` / `Entailed Prements:`. Pure prose — no pipe triplets.*

### SPO v4 adapter output

```
Non-Entailed Prems:
subject | is (entailment tag=confidence=0%) | The speaker believes that reflecting when one's perspective aligns with others' opinions should be done.
subject | has been heard by someone else before (confident=False)
subject | was previously in conflict or disagreement with the majority (confused=True)

Entailed Prims:
person | pauses (pause action) at moment of alignment with majority opinion
action taken after pausing | involves reflection about current situation
reflection includes consideration for personal beliefs vs external views
majority represents dominant group voice over individual perspectives
individual decision-making depends upon comparison between self-interest versus collective interests
```

*Headers abbreviated: `Non-Entailed Prems:` / `Entailed Prims:`. Partial pipe structure — some triplets appear but confidence annotation format is non-standard (`confidence=0%` instead of `confidence=0.0`).*

**Comparison:** v4 shows structural improvement on this sample — pipe-separated triplets appear where the base adapter generates pure prose. Confidence format still non-standard.

---

## Gate scores (v4 regression gate, 8 samples)

| Sample | Score | Notes |
|--------|-------|-------|
| 0 | 0.65 | `Non-Entailed Prems:` / `Entailed Prs:` |
| 1 | 0.65 | `Non-Entailed Prems:` / `Entailed Prs:` |
| 2 | 0.65 | `Non-Entailed Prems:` / `Entailed Prs:` |
| 3 | 0.65 | `Non-Entailed Prems:` / `Entailed Prims:` |
| 4 | 0.50 | `Non-Entailed Prems:` only (missing second section) |
| 5 | 0.475 | `Non-Entailed Prems:` / `Entailed Prims:` |
| 6 | 0.65 | `Non-Entailed Prems:` / `Implicit Reasoning Facts:` |
| 7 | 0.65 | `Non-Entailed Prems:` / `Entailed Prims:` |

**avg: 0.609 — gate threshold: 0.25 → PASSED**

Gate thresholds: `regression_min_avg_score=0.25`, `regression_min_per_sample_score=0.0`.  
All samples score above 0.0. Gate passes with comfortable margin on avg score.

The per-sample floor is intentionally permissive (0.0) because the base adapter itself scores
below 0.3 on some samples — the gate is designed to catch SPO-induced *regressions*, not to
enforce absolute quality.
