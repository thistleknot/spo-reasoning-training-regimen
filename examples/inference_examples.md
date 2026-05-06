# Inference Examples — Stratified Multi-Regimen Adapter

Generated: 2026-05-05 19:42  
Adapter: `output/stratified_multireg/base-plus-facts/adapter`  
Inference config: `no_repeat_ngram_size=6`, `max_new_tokens=384`, greedy decode, chat-template wrapped  
Note: `repetition_penalty` omitted — penalises header tokens that appear in the prompt instruction list.

---

## BASE REASONING regimen

## Sample 1 — BASE REASONING

**Quote:** "Be yourself; everyone else is already taken."

**Header score:** 1.00 ✓ PASS  
**First header:** `Non-Entailed Premises:`

### Output

```
Non-Entailed Premises:
everyone else | is (observed) | already taken
everyone else | is (inferred) | is not yourself
everyone else | is not (observed) | taken

Entailed Premises:
be yourself | is (observed/observed) | is the correct action
everyone else | is already (observed/observing) | taken
everyone else | has (observed/observers) | already taken

Throughline:
The speaker believes that others are not themselves, and thus they should be.
```

---
## Sample 2 — BASE REASONING

**Quote:** "I'm selfish, impatient and a little insecure. I make mistakes, I am out of control and at times hard to handle. But if you can't handle me at my worst, then you sure as hell don't deserve me at my best."

**Header score:** 1.00 ✓ PASS  
**First header:** `Non-Entailed Premises:`

### Output

```
Non-Entailed Premises:
subject | is (observed) | the speaker is selfish
subject | is (observing) | the speaker is impatient
subject | is (inferred) | the speaker is insecure
subject | is (obscured) | the speaker is out of control
subject | is (implied) | the speaker is hard to handle

Entailed Premises:
the speaker | is (observed/observed) | makes mistakes
the speaker | is (inferred/observed) | is out of control at times
the speaker | is a (observed/observing) | little insecure
the speaker | is out of control (observed/observers) | at times
the speaker's best is (observed/observer) | not deserving of the listener's respect
the listener | is (observed/inferred) | can't handle the speaker at their worst
the speaker | is selfish (observed/observes) | at their worst

Throughline:
The speaker's negative traits (selfishness, impatience, insecurity, out-of-control behavior, and difficulty handling) are detrimental to the listener's respect for the speaker's positive traits (best).
```

---
## Sample 3 — BASE REASONING

**Quote:** "A room without books is like a body without a soul."

**Header score:** 1.00 ✓ PASS  
**First header:** `Non-Entailed Premises:`

### Output

```
Non-Entailed Premises:
books | are (observed) | physical objects
soul | is (observed) | a concept
room | is (observed)

Entailed Premises:
room | is (observing) | a body
body | is (observing)
room | is (observers) | a body
room | is (inferred) | lacks soul
body | is (inferred)
room | is (inference) | lacks soul
room | is (observation) | a body
soul | is a (observed) concept
body | is (observation) without soul

Throughline:
A room without books lacks a soul.
```

---
## Sample 4 — BASE REASONING

**Quote:** "Be who you are and say what you feel, because those who mind don't matter, and those who matter don't mind."

**Header score:** 1.00 ✓ PASS  
**First header:** `Non-Entailed Premises:`

### Output

```
Non-Entailed Premises:
Matters | is (inferred) | are not the same as minds
Matters | is (observed) | are not the subject of the quote
Minds | is (observed)

Entailed Premises:
Mindless | is (inferred)
Matters | is (obscured) | are not the subjects of the quote
Matters | is a (observed) | state of being
Minds | is a (observers) | state of being

Throughline:
Matters are states of being, and those who matter do not mind, while those who mind do not matter.
```

---
## FACTS regimen

## Sample 5 — FACTS

**Quote:** "Be yourself; everyone else is already taken."

**Header score:** 1.00 ✓ PASS  
**First header:** `Non-Entailed Premises:`

### Output

```
Non-Entailed Premises:
everyone else | is (observed, confidence=1.0) | already taken
everyone else | is (inferred, confidence=0.5) | not yourself
everyone else | is not (observed, confidence=-1.0) | yourself

Entailed Premises:
be yourself | is (observed) | the correct action
everyone else | is already taken | is (observed)
everyone else | is (opposite of) yourself | is (observing)
```

---
## Sample 6 — FACTS

**Quote:** "I'm selfish, impatient and a little insecure. I make mistakes, I am out of control and at times hard to handle. But if you can't handle me at my worst, then you sure as hell don't deserve me at my best."

**Header score:** 1.00 ✓ PASS  
**First header:** `Non-Entailed Premises:`

### Output

```
Non-Entailed Premises:
subject | is (inferred, confidence=0.5) | The speaker is a person who is out of control at their worst.
subject | is (inverted, confidence=0.3) | The speaker is at times hard to handle.
subject | is (observed, confidence=1.0) | The speaker makes mistakes.

Entailed Premises:
Subject | is (observed) | The speaker is selfish.
Subject | is (obsured) | The speaker is impatient.
Subject | is (inferred) | The speaker is insecure.
Subject | is (obseved) | The speaker is out of control.
Subject | is (is) | The speaker is hard to handle.
Subject | is (opposite) | If you can't handle me (the worst version) at your best, then you sure as heck don't deserve me (the best version).
```

---
## Sample 7 — FACTS

**Quote:** "A room without books is like a body without a soul."

**Header score:** 1.00 ✓ PASS  
**First header:** `Non-Entailed Premises:`

### Output

```
Non-Entailed Premises:
books | are (observed, confidence=1.0) | physical objects
soul | is (observed, confidence=X) | a concept
room | is (observed) | a physical space
body | is (observed)

Entailed Premises:
room | is (inferred, confidence=0.5) | lacks soul
soul | is a (observed, confidence-X) | concept
room | is a (observing, confidence=1.5) | body
books | is (observed as a (observing, confident=1.5) physical object)
```

---
## Sample 8 — FACTS

**Quote:** "Be who you are and say what you feel, because those who mind don't matter, and those who matter don't mind."

**Header score:** 1.00 ✓ PASS  
**First header:** `Non-Entailed Premises:`

### Output

```
Non-Entailed Premises:
Matters | is (inferred, confidence=0.5) | are not the same as minds
Matters | is (observed, confidence=1.0) | are not the subject of the quote
Mind | is (inferred) | is a separate entity from matter
Matters | is a (observed, confidence-1.0) | state of being

Entailed Premises:
Speaking | is (observed) | is a form of expression
Speaking | is (inferred from the quote) | is a way to express one's identity
Speaking | is (observing) | is a form that matters
Speaking | is (obscured) | is a form without being subject to the mind
Speaking | is (implied) | is a form with identity
Speaking | is a (observing) | state of being
Speaking | is (opposite of) mind
Speaking | is a form of (observing) | expression
Speaking | is a way to (observing) | express one's identity
Mind | is (observed as a separate entity) | is not the same as speaking
Mind | is (opposite) | speaking
Speaking | is (expressing) | identity
Speaking | is the (observing) | act of saying
Speaking | is (explanatory) | the act of expressing one's identity
Speaking is (observing) the act of saying
```

---
## Gate summary

| Regimen | Sample | Header Score | Gate |
|---------|--------|-------------|------|
| BASE | 1 | 1.00 | ✓ |
| BASE | 2 | 1.00 | ✓ |
| BASE | 3 | 1.00 | ✓ |
| BASE | 4 | 1.00 | ✓ |
| FACTS | 5 | 1.00 | ✓ |
| FACTS | 6 | 1.00 | ✓ |
| FACTS | 7 | 1.00 | ✓ |
| FACTS | 8 | 1.00 | ✓ |

**avg header score: 1.000 — gate (min 0.5): PASS**
