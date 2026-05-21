# v20 Generation: 5 Scored vs 5 Unscored

## SECTION A — SCORED (Pass 1 + Pass 2 + Pass 3 confidence)

---
### SCORED Record 1
**Quote:** I'm selfish, impatient and a little insecure. I make mistakes, I am out of control and at times hard to handle. But if you can't handle me at my worst, then you sure as hell don't deserve me at my best.

**Throughline:** Selfishness creates conflict with others' trustworthiness, which ultimately affects their willingness to accept your presence.
**Throughline confidence:** `0.93`

**Entailed premises:**
- `observed` conf=`0.8` — being self-centered causes problems
- `observed` conf=`0.8` — having trouble handling oneself makes others less likely to trust you

**Non-entailed premises:**
- `inferred` conf=`0.5` — selfishness is inherently good
- `inferred` conf=`0.5` — being out of control is a sign of intelligence

**Training lines:**
```
I'm selfish, impatient and a little insecure.; being self-centered causes problems [transliterate]
```
```
I make mistakes, I am out of control and at times hard to handle.; having trouble handling oneself makes others less likely to trust you [transliterate]
```

---
### SCORED Record 2
**Quote:** Two things are infinite: the universe and human stupidity; and I'm not sure about the universe.

**Throughline:** The universe represents infinity while human stupidity represents finiteness.
**Throughline confidence:** `0.95`

**Entailed premises:**
- `observed` conf=`0.83` — the universe is infinite

**Non-entailed premises:**
- `inferred` conf=`0.17` — I am uncertain about the existence of the universe

**Training lines:**
```
Two things are infinite: the universe and human stupidity; the universe is infinite [transliterate]
```

---
### SCORED Record 3
**Quote:** Friendship ... is born at the moment when one man says to another "What! You too? I thought that no one but myself . . .

**Throughline:** The origin of true friendship arises from recognizing common humanity rather than individual superiority alone.
**Throughline confidence:** `0.95`

**Entailed premises:**
- `observed` conf=`0.89` — friendship originates from mutual acknowledgment of shared humanity

**Non-entailed premises:**
- `inferred` conf=`0.43` — the act of saying 'I thought that no one but myself' constitutes friendship itself

**Training lines:**
```
Friendship ...; friendship originates from mutual acknowledgment of shared humanity [transliterate]
```

---
### SCORED Record 4
**Quote:** Always forgive your enemies; nothing annoys them so much.

**Throughline:** Enemies do not possess intrinsic annoyance that makes them unappealing.
**Throughline confidence:** `0.91`

**Entailed premises:**
- `observed` conf=`0.83` — enemies are not inherently annoying

**Non-entailed premises:**
- `inferred` conf=`0.45` — all people are equally annoying

**Training lines:**
```
Always forgive your enemies; enemies are not inherently annoying [transliterate]
```

---
### SCORED Record 5
**Quote:** To live is the rarest thing in the world. Most people exist, that is all.

**Throughline:** The rarity of life itself constitutes its highest worth.
**Throughline confidence:** `0.95`

**Entailed premises:**
- `observed` conf=`0.83` — existence is valuable

**Non-entailed premises:**
- `inferred` conf=`0.45` — life has intrinsic value beyond mere survival

**Training lines:**
```
Most people exist, that is all.; existence is valuable [transliterate]
```


## SECTION B — UNSCORED (Pass 1 only)

---
### UNSCORED Record 1
**Quote:** Friendship ... is born at the moment when one man says to another "What! You too? I thought that no one but myself . . .

**Throughline:** The origin of true friendship arises from recognizing common humanity rather than individual superiority alone.
**Throughline confidence:** none

**Entailed premises:**
- `observed` — friendship originates from mutual acknowledgment of shared humanity

**Non-entailed premises:**
- `inferred` — the act of saying 'I thought that no one but myself' constitutes friendship itself

**Training lines:**
```
Friendship ...; friendship originates from mutual acknowledgment of shared humanity [transliterate]
```

---
### UNSCORED Record 2
**Quote:** Always forgive your enemies; nothing annoys them so much.

**Throughline:** Enemies do not possess intrinsic annoyance that makes them unappealing.
**Throughline confidence:** none

**Entailed premises:**
- `observed` — enemies are not inherently annoying

**Non-entailed premises:**
- `inferred` — all people are equally annoying

**Training lines:**
```
Always forgive your enemies; enemies are not inherently annoying [transliterate]
```

---
### UNSCORED Record 3
**Quote:** To live is the rarest thing in the world. Most people exist, that is all.

**Throughline:** The rarity of life itself constitutes its highest worth.
**Throughline confidence:** none

**Entailed premises:**
- `observed` — existence is valuable

**Non-entailed premises:**
- `inferred` — life has intrinsic value beyond mere survival

**Training lines:**
```
Most people exist, that is all.; existence is valuable [transliterate]
```

---
### UNSCORED Record 4
**Quote:** Darkness cannot drive out darkness: only light can do that. Hate cannot drive out hate: only love can do that.

**Throughline:** The efficacy of light and love depends on their intrinsic nature rather than external forces like darkness or hatred.
**Throughline confidence:** none

**Entailed premises:**
- `observed` — light drives out darkness
- `observed` — love drives out hate

**Non-entailed premises:**
- `inferred` — hate is inherently evil
- `inferred` — darkness is inherently bad

**Training lines:**
```
Darkness cannot drive out darkness: only light can do that.; light drives out darkness [transliterate]
```
```
Hate cannot drive out hate: only love can do that.; love drives out hate [transliterate]
```

---
### UNSCORED Record 5
**Quote:** Here's to the crazy ones. The misfits. The rebels. The troublemakers. The round pegs in the square holes. The ones who see things differently. They're not fond of rules. And they have no respect for the status quo. You can quote them, disagree with them, glorify or vilify them. About the only thing you can't do is ignore them. Because they change things. They push the human race forward. And while some may see them as the crazy ones, we see genius. Because the people who are crazy enough to think they can change the world, are the ones who do.

**Throughline:** The unconventional and innovative are essential for driving societal advancement.
**Throughline confidence:** none

**Entailed premises:**
- `observed` — crazy individuals drive social progress

**Non-entailed premises:**
- `inferred` — people should always follow established norms

**Training lines:**
```
Here's to the crazy ones.; crazy individuals drive social progress [transliterate]
```
