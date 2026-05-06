# Inference Examples — Stratified Multi-Regimen Adapter

Generated: 2026-05-05 20:51  
Adapter: `output/stratified_multireg_v2/base-plus-facts/adapter`  
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
be yourself | is (inferred/observed) | is the correct action
everyone else | is already (observed) | is taken
everyone else | has (observed) | been taken

Throughline:
The speaker advises against conforming to others' expectations because they are already taken by others.
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
subject | is (is) | the speaker is out of control
subject | is (implied) | the speaker is hard to handle

Entailed Premises:
speaker | is (inferred/observed) | the best is not deserved if you can't handle the worst
speaker | is (inverted) | the worst is not deserved if you cannot handle the best
speaker | is (observed/observed) | makes mistakes
speaker | is (is) (observed) | at times hard to handle
speaker | is (implied/observed) | is out of control
speaker | is (obscured) | is selfish

Throughline:
The speaker's best is not deserved if they cannot handle their worst.
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
body | is (observed)

Entailed Premises:
a room | is (observed as a container) | without books
a body | is (observed in the container) | without soul
room | is (observed without books) | like a body without soul

Throughline:
A room without books is analogous to a body without a soul.
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
Mind | is (observed/observed) | is a separate entity from matters
Matters | is a (observed) | state of being

Entailed Premises:
Mindless | is (inferred/observed) | do not matter
Matters | is the (observed) | subject of the quote
Matters | is characterized by (observed) | being who they are
Mindless | is characterized by (inferred) | not caring about others
Mindless | is (observed/inferred) | do not matter

Throughline:
Matters are characterized by being who they are and expressing their feelings, while mindless individuals do not care about others.
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
everyone else | is not (observed, confidence=X) | already taken

Entailed Premises:
be yourself | is (observed) | the correct action
everyone else | is already taken | is (observed)
everyone else | is (opposite of) yourself | is (observing, confidence=1.5) | already taken

Throughline:
The correct action is to be yourself, as others are not.
```

---
## Sample 6 — FACTS

**Quote:** "I'm selfish, impatient and a little insecure. I make mistakes, I am out of control and at times hard to handle. But if you can't handle me at my worst, then you sure as hell don't deserve me at my best."

**Header score:** 1.00 ✓ PASS  
**First header:** `Non-Entailed Premises:`

### Output

```
Non-Entailed Premises:
I | is (observed, confidence=1.0) | make mistakes
I | is (observing, confidence=1.5) | am out of control
I | is (inferred, confidence=0.5) | is hard to handle at times
I | is (is (observed, confidence-1.0) | selfish
I | is (inspired, confidence=0.75) | impatient

Entailed Premises:
If you can't handle me (subject) at your worst, then you sure (subject) don't deserve me (subject) at my best.
I | is (observers, confidence=1.25) | make mistakes
You | is (observers) | can't handle me (observer) at your worst
You | is (inferred) | don't deserve me (observer) at my best

Throughline:
The speaker's behavior is characterized by selfishness, impatience, insecurity, and vulnerability, which makes them difficult to handle.
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
body | is (observed) | a physical entity
room | is (observed)

Entailed Premises:
a room without books | is (observed with confidence=1.0)
a body without soul | is (observed)
books | are (observing, confidence=1.5) | physical objects
room | is (inferred, confidence=0.5) | a body without soul

Throughline:
A room without books is a body without soul.
```

---
## Sample 8 — FACTS

**Quote:** "Be who you are and say what you feel, because those who mind don't matter, and those who matter don't mind."

**Header score:** 1.00 ✓ PASS  
**First header:** `Non-Entailed Premises:`

### Output

```
Non-Entailed Premises:
Matters | is (inferred, confidence=0.5) | are not the same as mind
Matters | is (observed, confidence=1.0) | do not mind
Mind | is (inferred) | is not the same as matter
Mind | is (observed) | is not the thing

Entailed Premises:
Speaking | is (inferred from quote) | is important
Speaking | is (observed as a fact) | is important
Matters | is a (observed) | group of people
Mind | is (obscured) | is not the main focus of the quote
Matters | is not (observed) | the same as mind
Speaking | is (inverted) | is the main focus of the statement
Speaking | is (obscure) | is not the focus of the statement

Throughline:
Speaking is the main focus of a statement, and it is important to speak who you are and what you feel.
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
