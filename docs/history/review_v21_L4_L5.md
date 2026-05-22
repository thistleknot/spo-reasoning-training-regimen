# v21 Pipeline — Layer 4 & Layer 5 (clean, no [SEP])

### Record 1
> I'm selfish, impatient and a little insecure. I make mistakes, I am out of control and at times hard to handle. But if you can't handle me at my worst, then you sure as hell don't deserve me at my best.

**Layer 4**
```
quote: "I'm selfish, impatient and a little insecure. I make mistakes, I am out of control and at times hard to handle. But if you can't handle me at my worst, then you sure as hell don't deserve me at my best."
entailed:
  - "selfish, impatient and a little insecure"; being selfish | makes (observed, confidence=0.80) | mistakes
  - "selfish"; being impatient | makes (observed, confidence=0.80) | mistakes
  - "I ' m selfish, impatient and a little insecure"; being insecure | makes (observed, confidence=0.80) | mistakes
  - "I am out of control"; being hard to handle | makes (observed, confidence=0.80) | mistakes
  - "you sure as hell don ' t deserve me at my best"; deserve | to (inferred, confidence=0.80) | me
non_entailed:
  - "if you can ' t handle me at my worst, then you sure as hell don ' t deserve me at my best"; meritocracy | requires (inferred, confidence=0.50) | fair treatment
throughline: "The author identifies specific traits that lead to negative outcomes, suggesting that only those with these traits should receive their highest value." (confidence=0.92)
```

**Layer 5**
```
quote: "I'm selfish, impatient and a little insecure. I make mistakes, I am out of control and at times hard to handle. But if you can't handle me at my worst, then you sure as hell don't deserve me at my best."
entailed:
  - "selfish, impatient and a little insecure"; being selfish | makes (observed) | mistakes
  - "selfish"; being impatient | makes (observed) | mistakes
  - "I ' m selfish, impatient and a little insecure"; being insecure | makes (observed) | mistakes
  - "I am out of control"; being hard to handle | makes (observed) | mistakes
  - "you sure as hell don ' t deserve me at my best"; deserve | to (inferred) | me
non_entailed:
  - "if you can ' t handle me at my worst, then you sure as hell don ' t deserve me at my best"; meritocracy | requires (inferred) | fair treatment
throughline: "The author identifies specific traits that lead to negative outcomes, suggesting that only those with these traits should receive their highest value."
```

---

### Record 2
> Two things are infinite: the universe and human stupidity; and I'm not sure about the universe.

**Layer 4**
```
quote: "Two things are infinite: the universe and human stupidity; and I'm not sure about the universe."
entailed:
  - "the universe and human stupidity"; universe | is (observed, confidence=0.80) | infinite
  - "the universe and human"; stupidity | is (observed, confidence=0.80) | infinite
non_entailed:
  - "the universe and human stupidity"; the universe | is (inferred, confidence=0.50) | infinite
  - "the universe"; human stupidity | is (inferred, confidence=0.50) | infinite
throughline: "The statement asserts that both the universe and human stupidity are inherently infinite, while the speaker expresses uncertainty regarding their own status within that framework." (confidence=0.94)
```

**Layer 5**
```
quote: "Two things are infinite: the universe and human stupidity; and I'm not sure about the universe."
entailed:
  - "the universe and human stupidity"; universe | is (observed) | infinite
  - "the universe and human"; stupidity | is (observed) | infinite
non_entailed:
  - "the universe and human stupidity"; the universe | is (inferred) | infinite
  - "the universe"; human stupidity | is (inferred) | infinite
throughline: "The statement asserts that both the universe and human stupidity are inherently infinite, while the speaker expresses uncertainty regarding their own status within that framework."
```

---

### Record 3
> So many books, so little time.

**Layer 4**
```
quote: "So many books, so little time."
entailed:
  - "so little time"; time spent reading | consumes (inferred, confidence=0.80) | limited resources
non_entailed:
throughline: "The scarcity of available time necessitates the pursuit of extensive reading material." (confidence=0.95)
```

**Layer 5**
```
quote: "So many books, so little time."
entailed:
  - "so little time"; time spent reading | consumes (inferred) | limited resources
non_entailed:
throughline: "The scarcity of available time necessitates the pursuit of extensive reading material."
```

---

### Record 4
> A room without books is like a body without a soul.

**Layer 4**
```
quote: "A room without books is like a body without a soul."
entailed:
  - "A room without books is like a body without a soul"; books | contain (observed, confidence=0.83) | information
non_entailed:
  - "A room without books is like a body without a soul"; physical existence | exists independently (inferred, confidence=0.45) | mental state
throughline: "Physical objects are distinct from mental entities, yet both require specific conditions to function effectively." (confidence=0.92)
```

**Layer 5**
```
quote: "A room without books is like a body without a soul."
entailed:
  - "A room without books is like a body without a soul"; books | contain (observed) | information
non_entailed:
  - "A room without books is like a body without a soul"; physical existence | exists independently (inferred) | mental state
throughline: "Physical objects are distinct from mental entities, yet both require specific conditions to function effectively."
```

---

### Record 5
> You've gotta dance like there's nobody watching,Love like you'll never be hurt,Sing like there's nobody listening,And live like it's heaven on earth.

**Layer 4**
```
quote: "You've gotta dance like there's nobody watching,Love like you'll never be hurt,Sing like there's nobody listening,And live like it's heaven on earth."
entailed:
  - "Sing like there ' s nobody listening"; singing | is (observed, confidence=0.80) | free of harm
  - "heaven on earth"; living | is (observed, confidence=0.80) | safe
non_entailed:
  - "Sing"; being watched | causes (inferred, confidence=0.50) | harmony
throughline: "The absence of external interference creates conditions where harmony and safety are achieved." (confidence=0.94)
```

**Layer 5**
```
quote: "You've gotta dance like there's nobody watching,Love like you'll never be hurt,Sing like there's nobody listening,And live like it's heaven on earth."
entailed:
  - "Sing like there ' s nobody listening"; singing | is (observed) | free of harm
  - "heaven on earth"; living | is (observed) | safe
non_entailed:
  - "Sing"; being watched | causes (inferred) | harmony
throughline: "The absence of external interference creates conditions where harmony and safety are achieved."
```

---
