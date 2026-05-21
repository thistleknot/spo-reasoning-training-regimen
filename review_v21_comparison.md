# v21 S|P|O Format — 5 Scored vs 5 Unscored

## UNSCORED (no `--score-confidence`)

### Record 1
**Quote:** I'm selfish, impatient and a little insecure. I make mistakes, I am out of control and at times hard to handle. But if you can't handle me at my worst, then you sure as hell don't deserve me at my best.

**Entailed premises:**
- `observed` — `being selfish | makes | mistakes`
- `inferred` — `deserve | to | me`

**Non-entailed premises:**
- `inferred` — `meritocracy | requires | fair treatment`

**Throughline:** The author identifies specific traits that lead to negative outcomes, suggesting that only those with these traits should receive their highest value.
**Throughline confidence:** `n/a`

**Training lines:**
```
I make mistakes, I am out of control and at times hard to handle.; being selfish | makes | mistakes [transliterate]
```
```
But if you can't handle me at my worst, then you sure as hell don't deserve me at my best.; deserve | to | me [transliterate]
```
```
meritocracy | requires | fair treatment [transliterate]
```
---
### Record 2
**Quote:** Two things are infinite: the universe and human stupidity; and I'm not sure about the universe.

**Entailed premises:**
- `observed` — `universe | is | infinite`
- `inferred` — `I | am | not sure`

**Non-entailed premises:**
- `inferred` — `the universe | is | infinite`
- `inferred` — `human stupidity | is | infinite`

**Throughline:** The statement asserts that both the universe and human stupidity are inherently infinite, while the speaker expresses uncertainty regarding their own status within that framework.
**Throughline confidence:** `n/a`

**Training lines:**
```
Two things are infinite: the universe and human stupidity; universe | is | infinite [transliterate]
```
```
and I'm not sure about the universe.; I | am | not sure [transliterate]
```
```
the universe | is | infinite [transliterate]
```
```
human stupidity | is | infinite [transliterate]
```
---
### Record 3
**Quote:** Friendship ... is born at the moment when one man says to another "What! You too? I thought that no one but myself . . .

**Entailed premises:**
- `observed` — `friendship | is | born`

**Non-entailed premises:**
- `inferred` — `social interaction | forms | friendship`

**Throughline:** The act of mutual recognition initiates the formation of genuine friendship.
**Throughline confidence:** `n/a`

**Training lines:**
```
Friendship ...; friendship | is | born [transliterate]
```
```
social interaction | forms | friendship [transliterate]
```
---
### Record 4
**Quote:** Always forgive your enemies; nothing annoys them so much.

**Entailed premises:**
- `observed` — `enemies | are | unhappy`

**Non-entailed premises:**
- `inferred` — `annoying | causes | noisy`

**Throughline:** The act of forgiving enemies removes unnecessary noise and creates harmony.
**Throughline confidence:** `n/a`

**Training lines:**
```
Always forgive your enemies; enemies | are | unhappy [transliterate]
```
```
annoying | causes | noisy [transliterate]
```
---
### Record 5
**Quote:** To live is the rarest thing in the world. Most people exist, that is all.

**Entailed premises:**
- `observed` — `existence | is | the most common state`

**Non-entailed premises:**
- `inferred` — `life | is | a unique event`

**Throughline:** The rarity of life is defined by its universality among others.
**Throughline confidence:** `n/a`

**Training lines:**
```
Most people exist, that is all.; existence | is | the most common state [transliterate]
```
```
life | is | a unique event [transliterate]
```
---

## SCORED (`--score-confidence`)

### Record 1
**Quote:** I'm selfish, impatient and a little insecure. I make mistakes, I am out of control and at times hard to handle. But if you can't handle me at my worst, then you sure as hell don't deserve me at my best.

**Entailed premises:**
- `observed` conf=`0.80` — `being selfish | makes | mistakes`
- `inferred` conf=`0.80` — `deserve | to | me`

**Non-entailed premises:**
- `inferred` conf=`0.50` — `meritocracy | requires | fair treatment`

**Throughline:** The author identifies specific traits that lead to negative outcomes, suggesting that only those with these traits should receive their highest value.
**Throughline confidence:** `0.92`

**Training lines:**
```
I make mistakes, I am out of control and at times hard to handle.; being selfish | makes | mistakes [transliterate]
```
```
But if you can't handle me at my worst, then you sure as hell don't deserve me at my best.; deserve | to | me [transliterate]
```
```
meritocracy | requires | fair treatment [transliterate]
```
---
### Record 2
**Quote:** Two things are infinite: the universe and human stupidity; and I'm not sure about the universe.

**Entailed premises:**
- `observed` conf=`0.80` — `universe | is | infinite`
- `inferred` conf=`0.80` — `I | am | not sure`

**Non-entailed premises:**
- `inferred` conf=`0.50` — `the universe | is | infinite`
- `inferred` conf=`0.50` — `human stupidity | is | infinite`

**Throughline:** The statement asserts that both the universe and human stupidity are inherently infinite, while the speaker expresses uncertainty regarding their own status within that framework.
**Throughline confidence:** `0.94`

**Training lines:**
```
Two things are infinite: the universe and human stupidity; universe | is | infinite [transliterate]
```
```
and I'm not sure about the universe.; I | am | not sure [transliterate]
```
```
the universe | is | infinite [transliterate]
```
```
human stupidity | is | infinite [transliterate]
```
---
### Record 3
**Quote:** Friendship ... is born at the moment when one man says to another "What! You too? I thought that no one but myself . . .

**Entailed premises:**
- `observed` conf=`0.98` — `friendship | is | born`

**Non-entailed premises:**
- `inferred` conf=`0.43` — `social interaction | forms | friendship`

**Throughline:** The act of mutual recognition initiates the formation of genuine friendship.
**Throughline confidence:** `0.95`

**Training lines:**
```
Friendship ...; friendship | is | born [transliterate]
```
```
social interaction | forms | friendship [transliterate]
```
---
### Record 4
**Quote:** Always forgive your enemies; nothing annoys them so much.

**Entailed premises:**
- `observed` conf=`0.83` — `enemies | are | unhappy`

**Non-entailed premises:**
- `inferred` conf=`0.45` — `annoying | causes | noisy`

**Throughline:** The act of forgiving enemies removes unnecessary noise and creates harmony.
**Throughline confidence:** `0.92`

**Training lines:**
```
Always forgive your enemies; enemies | are | unhappy [transliterate]
```
```
annoying | causes | noisy [transliterate]
```
---
### Record 5
**Quote:** To live is the rarest thing in the world. Most people exist, that is all.

**Entailed premises:**
- `observed` conf=`0.83` — `existence | is | the most common state`

**Non-entailed premises:**
- `inferred` conf=`0.45` — `life | is | a unique event`

**Throughline:** The rarity of life is defined by its universality among others.
**Throughline confidence:** `0.92`

**Training lines:**
```
Most people exist, that is all.; existence | is | the most common state [transliterate]
```
```
life | is | a unique event [transliterate]
```
---