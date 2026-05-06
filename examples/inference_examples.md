# Inference Examples — Stratified Multi-Regimen Adapter

Generated: 2026-05-06 06:29  
Adapter: `output/stratified_multireg_v3/base-plus-facts/adapter`  
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
everyone else | is (inferred) | not taken
everyone else | has (observed) | taken

Entailed Premises:
be yourself | is (inferred/observed) | the correct action
everyone else | is already (observed) | not taken
be yourself | is (observed/observed) | a recommendation

Throughline:
Self-acceptance is the correct and necessary action for others to be taken.
```

---
## Sample 2 — BASE REASONING

**Quote:** "I'm selfish, impatient and a little insecure. I make mistakes, I am out of control and at times hard to handle. But if you can't handle me at my worst, then you sure as hell don't deserve me at my best."

**Header score:** 1.00 ✓ PASS  
**First header:** `Non-Entailed Premises:`

### Output

```
Non-Entailed Premises:
I | is (observed) | make mistakes
I | is (observers) | out of control
I | is (inferred) | hard to handle at times
I | is (impatient) | a little insecure

Entailed Premises:
If you can't handle me (I) at your worst, then you sure (observed) | don't deserve me (I) at my best
I | is (impatient) | a little insecure
I | is (selfish) | selfish
I | is (out of control) | out of control

Throughline:
The speaker is characterized by selfishness, impatience, and insecurity, which are detrimental to their ability to be respected by others.
```

---
## Sample 3 — BASE REASONING

**Quote:** "A room without books is like a body without a soul."

**Header score:** 1.00 ✓ PASS  
**First header:** `Non-Entailed Premises:`

### Output

```
Non-Entailed Premises:
books | is (observed) | are physical objects
soul | is (observed/observed) | a concept
body | is (observed/) | a physical entity
room | is (observed/)
room | is (observing) | a space

Entailed Premises:
room | is (observes) | a space
room | is (inferred) | lacks soul
room | is (is) | lacks soul
books | is (observes/) | are physical objects
room | is (has) | soul
room | is not (observes) | without soul
room | is like (observes) | body without soul
body | is (inferred/) | soul
room | lacks soul | implies (inferred) | soul
room without books | is like (observed) | body without soul

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
Matters | is (observed) | are not the same as mind
Matters | is (inferred) | are not the subject of the quote
Mind | is (observed/observed) | is a separate entity from matter
Matters | is a (observed) | verb
Mind | is (inferred/observed) | does not matter
Matters | does not matter | (observed/observing) | are not the subjects of the quote

Entailed Premises:
Mindless | is (observed/inferred) | do not matter
Matters (subject) | is (observed/) | do not matter
Mindless | is (inferred/) | do not mind
Mindless | is a (observing) | state
Mindless | is not (observed/) | a state of being
Mindless | is the opposite of (observed/) | being
Mindless | does not matter | is (observed/)
Mindless | is (opposite of) (observed/) | matter
Mindless | does (observed/) | not mind

Throughline:
Mindless individuals do not matter because they lack the ability to perceive or feel their own identity or the desire to express it.
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
everyone else | is not (observed, confidence-1.0) | self-aware

Entailed Premises:
self-awareness | is (observed) | a characteristic of being yourself
everyone else | has (observed) | already taken
being yourself | is (observed and inferred) | a characteristic of self-awareness
being yourself | is a (observed and inferred fact) | characteristic of self-awareness

Throughline:
Being yourself is a necessary condition for one to be taken by others.
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
I | is (observing, confidence=1.5) | out of control
I | is (inferred, confidence=0.5) | hard to handle at my worst
I | is (is, confidence=1.2) | insecure

Entailed Premises:
If you can't handle me (I) at your worst, then you sure (observing, confidence=-1.0) as not deserving (I) at your best
I | is (inspired, confidence=0.8) | make mistakes
You | is (observing) | can't handle me at your worst
You | is (inferred) | don't deserve me at your best
You | is (inspired) | can't handle I at your worst

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
books | is (observed, confidence=1.0) | are physical objects
soul | is (observed) | a concept
room | is (observed)

Entailed Premises:
room | is (inferred, confidence=0.5) | lacks soul
room | is (inverted, confidence=0.3) | lacks soul
books | is (inferred) | are physical objects

Throughline:
A room without books lacks soul.
```

---
## Sample 8 — FACTS

**Quote:** "Be who you are and say what you feel, because those who mind don't matter, and those who matter don't mind."

**Header score:** 1.00 ✓ PASS  
**First header:** `Non-Entailed Premises:`

### Output

```
Non-Entailed Premises:
Matters | is (observed, confidence=1.0) | are those who matter
Matters | is (inferred, confidence=0.5) | do not mind
Matters | is a (observed, confidence=X) | category
Matters | is not (observed, confidence=Y) | a category

Entailed Premises:
Mindlessness | is (observed) | a state of not caring about others
Matters | is the (observed) | state of caring about others
Minding | is (observed as a state) | a state of caring about others

Throughline:
Minding is a state of not caring, and those who matter are those who care.
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
