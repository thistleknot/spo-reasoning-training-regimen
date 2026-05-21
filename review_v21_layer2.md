# v21 Layer 2 — BERT Span + S|P|O

### Record 1
**Quote:** I'm selfish, impatient and a little insecure. I make mistakes, I am out of control and at times hard to handle. But if you can't handle me at my worst, then you sure as hell don't deserve me at my best.

**Throughline:** The author identifies specific traits that lead to negative outcomes, suggesting that only those with these traits should receive their highest value.

**[ENTAILED]**
- span: `selfish, impatient and a little insecure`
  `observed` → `being selfish | makes | mistakes`
- span: `selfish`
  `observed` → `being impatient | makes | mistakes`
- span: `I ' m selfish, impatient and a little insecure`
  `observed` → `being insecure | makes | mistakes`
- span: `I am out of control`
  `observed` → `being hard to handle | makes | mistakes`
- span: `you sure as hell don ' t deserve me at my best`
  `inferred` → `deserve | to | me`

**[NON-ENTAILED]**
- span: `if you can ' t handle me at my worst, then you sure as hell don ' t deserve me at my best`
  `inferred` → `meritocracy | requires | fair treatment`

---
### Record 2
**Quote:** Two things are infinite: the universe and human stupidity; and I'm not sure about the universe.

**Throughline:** The statement asserts that both the universe and human stupidity are inherently infinite, while the speaker expresses uncertainty regarding their own status within that framework.

**[ENTAILED]**
- span: `the universe and human stupidity`
  `observed` → `universe | is | infinite`
- span: `the universe and human`
  `observed` → `stupidity | is | infinite`

**[NON-ENTAILED]**
- span: `the universe and human stupidity`
  `inferred` → `the universe | is | infinite`
- span: `the universe`
  `inferred` → `human stupidity | is | infinite`

---
### Record 3
**Quote:** So many books, so little time.

**Throughline:** The scarcity of available time necessitates the pursuit of extensive reading material.

**[ENTAILED]**
- span: `so little time`
  `inferred` → `time spent reading | consumes | limited resources`

**[NON-ENTAILED]**

---
### Record 4
**Quote:** A room without books is like a body without a soul.

**Throughline:** Physical objects are distinct from mental entities, yet both require specific conditions to function effectively.

**[ENTAILED]**
- span: `A room without books is like a body without a soul`
  `observed` → `books | contain | information`

**[NON-ENTAILED]**
- span: `independently mental state [SEP] A room without books is like a body without a soul`
  `inferred` → `physical existence | exists independently | mental state`

---
### Record 5
**Quote:** You've gotta dance like there's nobody watching,Love like you'll never be hurt,Sing like there's nobody listening,And live like it's heaven on earth.

**Throughline:** The absence of external interference creates conditions where harmony and safety are achieved.

**[ENTAILED]**
- span: `Sing like there ' s nobody listening`
  `observed` → `singing | is | free of harm`
- span: `heaven on earth`
  `observed` → `living | is | safe`

**[NON-ENTAILED]**
- span: `Sing`
  `inferred` → `being watched | causes | harmony`

---