# Training Ladder — Holdout Examples

## Score progression by tier

| Tier | Avg score | Pass |
|---|---:|:---:|
| tier1_annotation | 0.8377 | ✓ |
| tier2_content | 0.8837 | ✓ |
| tier3_convergence | 0.8606 | ✓ |

---

# Holdout examples — tier1_annotation

**Pass:** ✓  |  **Avg score:** 0.8377  |  **N:** 20

## Check rates

| Check | Rate | Status |
|---|---:|:---:|
| headers | 90% | ✓ |
| entailed_non_empty | 100% | ✓ |
| pipes_well_formed | 100% | ✓ |
| no_template_leakage | 100% | ✓ |
| tags_exclusive | 100% | ✓ |

## Example 1  (score: 0.9500)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "Tis the privilege of friendship to talk nonsense, and to have her nonsense respected."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
friendship | (observed, confidence=1.0) | talk nonsense

Entailed Premises:
friendship | (inferred, confidence=0.7) | respected

Throughline:
Friendship is the privilege of having her nonsense respected.
```

**Generated**

```text
Non-Entailed Premises:
friendship | (inferred, confidence=0.7) | talk nonsense

Entailed Premises:

friendship | (in inferred, confidence=0.9) | talk nonsense

Throughline:
Tis the privilege of friendshp to talk nonsense, and have her nonsense respected.
```

## Example 2  (score: 0.8000)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "All God does is watch us and kill us when we get boring. We must never, ever be boring."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
watch us and kill us when we get boring | (inferred, confidence=0.7) | confidence=0.0 | never

Entailed Premises:
watch us and kill us when we get boring | (observed, confidence=1.0) | confidence=1.0 | we must never

Throughline:
We must never be boring.
```

**Generated**

```text
Non-Entailed Premises:
watch us | (observed, confidence=N) | kill us when we get bored

Entailed Premises:

*   watch us | (inferred, confidence=N) | when we get boring

Throughline:
*   We must never, ever get boring.
```

## Example 3  (score: 0.7000)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "Anyone who thinks sitting in church can make you a Christian must also think that sitting in a garage can make you a car."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
garage | (inferred, confidence=0.7) | car

Entailed Premises:
sitting in church | (inferred, confidence=0.7) | make you a Christian
sitting in a garage | (inferred, confidence=0.7) | make you a car

Throughline:
Anyone who thinks sitting in church can make you a Christian must also think that sitting in a garage can make you a car.
```

**Generated**

```text
Non-Entailed Premises:
garage | (inferred, confidence=0.7) | car

Entailed Premises:

*   (inferred, confidence=N) | garage | (inferred,confidence=N) | car

Throughline:
Anyone who thinks sitting in a garage canmake you a car must also think that sittingin church can make you a car.
```

## Example 4  (score: 0.9500)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "I don't go out with strangers," I said."Good thing I do. I'll pick you up at five."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
I don't go out with strangers | (inferred, confidence=0.7) | I said

Entailed Premises:
I don't go out with strangers | (observed, confidence=1.0) | I said

Throughline:
I said I don't go out with strangers, which implies I will pick you up at five.
```

**Generated**

```text
Non-Entailed Premises:
I don't go out | (inferred, confidence=0.7) | stranger

Entailed Premises:

I don't go out at five | (inferred,confidence=0.7)| five

Throughline:
I don'tgo out with strangers | (inferred, confident=0.7) I said.
```

## Example 5  (score: 0.8750)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "The truth does not change according to our ability to stomach it."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
stomach | stomach (inferred, confidence=0.7) | true

Entailed Premises:
stomach | true (inferred, confidence=0.7) | truth

Throughline:
The truth remains constant regardless of whether the subject can stomach the object.
```

**Generated**

```text
Non-Entailed Premises:
stomach | (inferred, confidence=0.7) | ability

Entailed Premises:

*   (stomach, inferred, confidence=0.9) | (inferred,confidence=0.9)| ability

Throughline:
The truth does not change regardless of the subject's ability to stomach it.
```

## Example 6  (score: 0.7600)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "I like your Christ, I do not like your Christians. Your Christians are so unlike your Christ."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
Christ | (observed, confidence=1.0) | Christ

Entailed Premises:
Christ | (observed, confidence=1.0) | Christians

Throughline:
The speaker claims that the speaker dislikes their own Christian identity because it is so unlike their Christ.
```

**Generated**

```text
Non-Entailed Premises:
Christ | (observed, confidence=N) | Christians

Entailed Premises:

Christ | (inferred, confidence=N) | Christ

Throughline:
The speaker asserts that their preference for Christ is distinct from their preference for Christians, implying that Christians are fundamentally unlike Christ.
```

## Example 7  (score: 0.8000)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "That's the thing about books. They let you travel without moving your feet."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
books | (observed, confidence=1.0) | travel without moving your feet

Entailed Premises:
books | (inferred, confidence=0.7) | travel without moving your feet

Throughline:
Books allow you to travel without having to move your feet.
```

**Generated**

```text
Non-Entailed Premises:
books | (observed, confidence=N) | travel without moving your feet

Entailed Premises:

books | (inferred, confidence=N) | let you travel without moving feet

Throughline:
books let you travel without moving their feet
```

## Example 8  (score: 0.8750)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "It means 'Shadowhunters: Looking Better in Black Than the Widows of our Enemies Since 1234'."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
Shadowhunters | (observed, confidence=1.0) | Looking Better in Black Than the Widows of our Enemies Since 1234

Entailed Premises:
Shadowhunters | (inferred, confidence=0.7) | Looking Better in Black Than the Widows of our Enemies Since 1234

Throughline:
Shadowhunters is a reference to the movie "Shadowhunters: Looking Better in Black Than the Widows of our Enemies Since 1234".
```

**Generated**

```text
Non-Entailed Premises:
Shadowhunters | (inferred, confidence=0.7) | Looking Better in Black Than

Entailed Premises:

Shadowhunters | (observed, confidence=N) | Since 1239

Throughline:
Shadowhunters: Looking better in black than the widows of our enemies since 1239
```

## Example 9  (score: 0.9300)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "When one is in love, one always begins by deceiving one's self, and one always ends by deceiving others. That is what the world calls a romance."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
deceiving one's self | (observed, confidence=1.0) | one always begins by deceiving one's self

Entailed Premises:
deceiving one's self | (observed, confidence=1.0) | one always ends by deceiving others

Throughline:
One always begins by deceiving one's self and always ends by deceiving others.
```

**Generated**

```text
Non-Entailed Premises:
deceiving one's self | (inferred, confidence=0.7) | deceiving one's self

Entailed Premises:

```
deceiving one's other | (inferred,confidence=0.7)| deceiving others
```

Throughline:
One always begins by deceiving oneself and always ends by deceiving other.
```

## Example 10  (score: 0.7800)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "We're all human, aren't we? Every human life is worth the same, and worth saving."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
Every human life is worth the same | (inferred, confidence=0.7) | worth saving

Entailed Premises:
We're all human, aren't we? | (inferred, confidence=0.7) | every human life is worth the same

Throughline:
We are all human, and every human life is worth the same.
```

**Generated**

```text
Non-Entailed Premises:
human | (observed, confidence=N) | human

Entailed Premises:

*   human | (inferred, confidence=N) | worth saving

Throughline:
Every human life is worth thesame, and worth saving.
```

## Example 11  (score: 0.9300)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "Never be bullied into silence. Never allow yourself to be made a victim. Accept no one's definition of your life, but define yourself."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
Never allow yourself to be made a victim | (inferred, confidence=0.7) | Never allow yourself to be made a victim

Entailed Premises:
Accept no one's definition of your life | (inferred, confidence=0.7) | Accept no one's definition of your life

Throughline:
Never allow yourself to be made a victim.
```

**Generated**

```text
Non-Entailed Premises:
Never allow yourself to be made into a victim | (inferred, confidence=0.7) | inferred

Entailed Premises:

*   Never allow yourself to be defined by others | (inferred,confidence=0.7) inferred | inferred

Throughline:
*   The quote instructs the reader to reject external definitions of their life and define their own identity.
```

## Example 12  (score: 0.8300)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "I have nothing to declare except my genius."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
genius | (observed, confidence=1.0) | nothing

Entailed Premises:
nothing | (observed, confidence=1.0) | declare

Throughline:
The speaker claims that only their inherent genius is available for declaration.
```

**Generated**

```text
Non-Entailed Premises:
genius | (inferred, confidence=0.7) | nothing

Entailed Premises:

nothing | (inferred,confidence=0.7)| nothing

throughline:
genius | nothing
```

## Example 13  (score: 0.7800)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "In the beginning there was nothing, which exploded."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
nothing | (observed, confidence=1.0) | exploded

Entailed Premises:
nothing | (inferred, confidence=0.7) | nothing

Throughline:
nothing exploded.
```

**Generated**

```text
Non-Entailed Premises:
nothing | (observed, confidence=N) | exploded

Entailed Premises:

nothing | (inferred, confidence=N) | nothing

Throughline:
nothing exploded.
```

## Example 14  (score: 0.8750)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "The one you love and the one who loves you are never, ever the same person."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
Love | (observed, confidence=1.0) | the one who loves you

Entailed Premises:
Love | (inferred, confidence=0.7) | the one you love

Throughline:
The one you love and the one who loves you are never, ever the same person.
```

**Generated**

```text
Non-Entailed Premises:
love | (observed, confidence=N) | never

Entailed Premises:

```
never | (inferred, confidence=0.7) | the same person
```

Throughline:
The one you love and one who loves you are not the same person.
```

## Example 15  (score: 0.7800)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "One of the advantages of being disorganized is that one is always having surprising discoveries."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
disorganized | (observed, confidence=1.0) | one is always having surprising discoveries

Entailed Premises:
disorganized | (inferred, confidence=0.7) | one is always having surprising discoveries

Throughline:
One of the advantages of being disorganized is that one is always having surprising discoveries.
```

**Generated**

```text
Non-Entailed Premises:
disorganized | (observed, confidence=N) | one

Entailed Premises:

One | (observed,confidence=N) | one

Throughline:
One of the advantages of disorganization is that one is always discovering surprising things.
```

## Example 16  (score: 0.8500)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "All you have to do is write one true sentence. Write the truest sentence that you know."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
write one true sentence | (inferred, confidence=0.7) | write the truest sentence that you know

Entailed Premises:
write the truest sentence that you know | (inferred, confidence=0.7) | write one true sentence

Throughline:
You must write the truest sentence that you know.
```

**Generated**

```text
Non-Entailed Premises:
write | (inferred, confidence=0.7) | true

Entailed Premises:

write | (inferred|confidence=0.7)   | true

throughline:
write the truest sentence that the speaker knows
```

## Example 17  (score: 0.8350)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "I am nothing special, of this I am sure. I am a common man with common thoughts and I've led a common life. There are no monuments dedicated to me and my name will soon be forgotten, but I've loved another with all my heart and soul, and to me, this has always been enough.."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
common man | with (inferred, confidence=0.7) | common thoughts

Entailed Premises:
common man | loved (inferred, confidence=0.7) | with all my heart and soul

Throughline:
The speaker asserts that despite being a common man with common thoughts, their love for another person remains the only thing that makes them special.
```

**Generated**

```text
Non-Entailed Premises:
common man | (inferred, confidence=0.7) | common man

Entailed Premises:

```
common man | (observed, confidence=N) | common man
```

Throughline:
The speaker asserts that they are a common man with common feelings, lacking special status or monuments, yet they have loved another with all their heart and soul, making this love sufficient for them.
```

## Example 18  (score: 0.7050)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "To the well-organized mind, death is but the next great adventure."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
death | (inferred, confidence=0.7) | the next great adventure

Entailed Premises:
death | (inferred, confidence=0.7) | the next great adventure adventure

Throughline:
death is the next great adventure for the well-organized mind
```

**Generated**

```text
Non-Entailed Premises:
death | (observed, confidence=N) | the next great adventure

Entailed Premises:

*   (observed, confidence=X.0) | the next great adaptation | the well-organized mind

Throughline:
The quote implies that death is a metaphor for a new challenge or adventure for a well-organized mind.
```

## Example 19  (score: 0.8000)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "Fate is like a strange, unpopular restaurant filled with odd little waiters who bring you things you never asked for and don't always like."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
Fate | is (inferred, confidence=0.7) | strange | unpopular | restaurant | filled | with | odd | little | waiters | who | bring | you | things | you | never | asked | for | and | don't | always | like |

Entailed Premises:
Fate | is (inferred, confidence=0.7) | like | a | strange | unpopular | restaurant | filled | with | odd | little | waiters | who | bring | you | things | you | never | asked | for | and | don't | always | like |

Throughline:
Fate is like a strange, unpopular restaurant filled with odd little waiters who bring you things you never asked for and don't always like.
```

**Generated**

```text
Non-Entailed Premises:
Fate | (observed, confidence=N) | strange

Entailed Premises:

Fate | (inferred, confidence=N) | like

Throughline:
Fate is like a weird, unpopular restaurant with odd waiters who bring you items you never asked for.
```

## Example 20  (score: 0.9500)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "Be mindful. Be grateful. Be positive. Be true. Be kind."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
mindful | (inferred, confidence=0.7) | grateful

Entailed Premises:
mindful | (inferred, confidence=0.7) | positive

Throughline:
Be mindful, be grateful, be positive, be true, be kind.
```

**Generated**

```text
Non-Entailed Premises:
Be mindful | (inferred, confidence=0.7) | Be grateful

Entailed Premises:

Be mindful | (in inferred, confidence=0.9) | Be true

Throughline:
Be mindful.
```


---

# Holdout examples — tier2_content

**Pass:** ✓  |  **Avg score:** 0.8837  |  **N:** 30

## Check rates

| Check | Rate | Status |
|---|---:|:---:|
| headers | 97% | ✓ |
| entailed_non_empty | 93% | ✓ |
| pipes_well_formed | 100% | ✓ |
| no_template_leakage | 100% | ✓ |
| tags_exclusive | 87% | ✓ |
| confidence_numeric | 80% | ✓ |
| canonical_tag_format | 90% | ✓ |
| sections_distinct | 100% | ✓ |
| verbatim_entailed | 87% | ✓ |
| avg_score_tier2 | 87% | ✓ |

## Example 1  (score: 0.7000)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "Prayer is not asking. It is a longing of the soul. It is daily admission of one's weakness. It is better in prayer to have a heart without words than words without a heart."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
soul | ( (inferred, confidence=0.7) | confidence=0.99) | prayer

Entailed Premises:
prayer | ( (inferred, confidence=0.7) | confidence=0.99) | soul

Throughline:
Prayer is a longing of the soul.
```

**Generated**

```text
Non-Entailed Premises:
Prayer | (inferred, confidence=0.7) | daily admission of one's strength

Entailed Premises:

*   (inferred, confidence=N) | prayer | (inferred, observed, confidence=1.5) | words without a heart

Throughline:
Prayer is a longing of the Soul.
```

## Example 2  (score: 0.7050)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "love the life you live.live the life you love."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
life | (observed, confidence=1.0) | live the life you love

Entailed Premises:
live the life you love | (inferred, confidence=0.7) | love the life you live

Throughline:
The quote implies that love is a reciprocal relationship where one person must love the life of the other.
```

**Generated**

```text
Non-Entailed Premises:
love | (observed, confidence=N) | the life you live

Entailed Premises:

*   (observed, confidence=C) | the life you love | the life you love

Throughline:
The quote implies that love is a reciprocal relationship where one person loves the life of the other, and the quote itself is a statement of this mutual love.
```

## Example 3  (score: 0.9500)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "She had always wanted words, she loved them; grew up on them. Words gave her clarity, brought reason, shape."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
Words | (inferred, confidence=0.7) | clarity

Entailed Premises:
Words | (observed, confidence=1.0) | reason

Throughline:
Words gave her clarity, brought reason, shape.
```

**Generated**

```text
Non-Entailed Premises:
Words | (inferred, confidence=0.7) | clarity

Entailed Premises:

```
Words | (in inferred, confidence=0.9) | clarity
```

Throughline:
Words gave her clarity, broughtreason, shape.
```

## Example 4  (score: 0.9500)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "Sometimes the questions are complicated and the answers are simple."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
questions | (inferred, confidence=0.7) | complicated

Entailed Premises:
questions | (inferred, confidence=0.7) | simple

Throughline:
The statement implies that questions are often complex while answers are straightforward.
```

**Generated**

```text
Non-Entailed Premises:
questions | (inferred, confidence=0.7) | complicated

Entailed Premises:

questions | (inferred|confidence=0.7) , confidence=0.7 | simple

Throughline:
The quote implies that questions are often complex while answers are simple.
```

## Example 5  (score: 0.9000)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "When you find your path, you must not be afraid. You need to have sufficient courage to make mistakes. Disappointment, defeat, and despair are the tools God uses to show us the way."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
sufficient courage | (observed, confidence=1.0) | make mistakes

Entailed Premises:
sufficient courage | (inferred, confidence=0.7) | make mistakes

Throughline:
You must not be afraid and have sufficient courage to make mistakes.
```

**Generated**

```text
Non-Entailed Premises:
sufficient courage | (inferred, confidence=0.7) | make mistakes

Entailed Premises:

*   sufficient courage | (inferred|confidence=0.7)| make mistakes
*   sufficient courage | inferred|confidence=0.9| make mistakes

Throughline:
You must not be afraid and have sufficient courage to make mistake
```

## Example 6  (score: 0.9500)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "A human being is a part of the whole called by us universe, a part limited in time and space. He experiences himself, his thoughts and feeling as something separated from the rest, a kind of optical delusion of his consciousness. This delusion is a kind of prison for us, restricting us to our personal desires and to affection for a few persons nearest to us. Our task must be to free ourselves from this prison by widening our circle of compassion to embrace all living creatures and the whole of nature in its beauty."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
human being | (observed, confidence=1.0) | universe

Entailed Premises:
human being | (inferred, confidence=0.7) | universe

Throughline:
The quote argues that a human being is a part of the whole universe, which creates a delusion of separation that acts as a prison restricting the individual to personal desires and affection for specific others, necessitating a task to free oneself by expanding compassion to embrace all living creatures and nature.
```

**Generated**

```text
Non-Entailed Premises:
human being | (inferred, confidence=0.7) | universe

Entailed Premises:

human being | (in inferred, confidence=0.9) | universe

Throughline:
The quote argues that a human being is a part limited in time, space, and consciousness, which acts as a prison restricting them to personal desires and affection, and that the solution to this prison is to widen the circle of compassion to embrace the whole of nature.
```

## Example 7  (score: 0.8000)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "It's going to be all right, sir," Harry said over and over again, more worried by Dumbledore's silence than he had been by his weakened voice. "We're nearly there ... I can Apparate us both back ... don't worry ...""I am not worried, Harry," said Dumbledore, his voice a little stronger despite the freezing water. "I am with you."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
Harry | (observed, confidence=1.0) | inferred, confidence=0.0 | Dumbledore

Entailed Premises:
Dumbledore | (observed, confidence=1.0) | inferred, confidence=0.0 | Harry

Throughline:
Dumbledore's intervention confirms Harry's continued anxiety about the situation, as evidenced by his repeated use of the phrase "I am with you."
```

**Generated**

```text
Non-Entailed Premises:
Harry | (observed, confidence=N) | Dumbledore's silence

Entailed Premises:

*   Harry | (inferred, confidence=N) | Harry's voice
*   Harry | (observed,confidence=N) | Dumbledore's voice
*   Dumbledore | (inferred,confidence=N) | Harry's silence

Throughline:
Dumbledore's silence is inferred to be more worried by Harry's weakened voice than by his own.
```

## Example 8  (score: 0.7000)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "I like flaws. I think they make things interesting."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
I like flaws | (inferred, confidence=0.7) | flaws

Entailed Premises:
I think they make things interesting | (inferred, confidence=0.7) | flaws

Throughline:
Flaws make things interesting.
```

**Generated**

```text
Non-Entailed Premises:
flaws | (inferred, confidence=0.7) | interesting

Entailed Premises:

*   (inferred, confidence=N) | interesting | (inferred,confidence=N) | flaws

Throughline:
I like flaws because they make things interesting.
```

## Example 9  (score: 0.9500)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "I hope you will have a wonderful year, that you'll dream dangerously and outrageously, that you'll make something that didn't exist before you made it, that you will be loved and that you will be liked, and that you will have people to love and to like in return. And, most importantly (because I think there should be more kindness and more wisdom in the world right now), that you will, when you need to be, be wise, and that you will always be kind."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
hope | < (inferred, confidence=0.7) | inferred, confidence=X.X> | "wonderful year"

Entailed Premises:
wonderful year | < (inferred, confidence=0.7) | inferred, confidence=X.X> | "you will"

Throughline:
The speaker expresses a wish for the recipient to have a wonderful year, dream dangerously, make something that didn't exist before them, be loved, have people to love and like in return, and be wise and always kind.
```

**Generated**

```text
Non-Entailed Premises:
wonderful year | (inferred, confidence=0.7) | confidence=0.0 | "I hope you will"

Entailed Premises:

*   "I hope you will be loved" | (inferred,confidence=0.7)|confidence=0.0 | object: "I hope you"

Throughline:
The speaker expresses a wish for the recipient to be loved and liked, and to be wise and kind.
```

## Example 10  (score: 0.9500)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "In the beginning there was nothing, which exploded."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
nothing | (observed, confidence=1.0) | exploded

Entailed Premises:
nothing | (inferred, confidence=0.7) | nothing

Throughline:
nothing exploded.
```

**Generated**

```text
Non-Entailed Premises:
nothing | (inferred, confidence=0.7) | exploded

Entailed Premises:

nothing | (inferred|observed, confidence=0.99) | exploded

Throughline:
nothing exploded.
```

## Example 11  (score: 0.9500)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "Don't stop there. I suppose there are also, what, vampires and werewolves and zombies?""Of course there are. Although you mostly find zombies farther south, where the voudun priests are.""What about mummies? Do they only hang around Egypt?""Don't be ridiculous. No one believes in mummies."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
vampires | (observed, confidence=1.0) | vampire

Entailed Premises:
vampires | (inferred, confidence=0.7) | vampire

Throughline:
The text implies that vampires exist alongside zombies and werewolves, and that mummies are not just a local phenomenon in Egypt but a widespread belief system.
```

**Generated**

```text
Non-Entailed Premises:
vampires | (inferred, confidence=0.7) | vampire

Entailed Premises:

```
vampires | inferred, confidence=0.9 | vampire
```

Throughline:
The quote implies that vampires and werewolves are also present in the region, specifically noting that zombies are found farther south where the voudun priests reside.
```

## Example 12  (score: 0.9000)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "Above all, be the heroine of your life, not the victim."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
heroine | (observed, confidence=1.0) | above all

Entailed Premises:
heroine | (observed, confidence=1.0) | be the heroine of your life

Throughline:
be the heroine of your life
```

**Generated**

```text
Non-Entailed Premises:
heroine | (observed, confidence=N) | life

Entailed Premises:

*   be the heroine of your Life | (inferred, confidence=0.7) | be the heroine of yourLife
*   be the heroine | (inferred,confidence=0.7)|be the heroine of yourLife

Throughline:
The quote instructs the reader to identify themselves as the "heroine" of their life rather than the "victim."
```

## Example 13  (score: 0.8750)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "How do you spell 'love'?" - Piglet"You don't spell it...you feel it." - Pooh"

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
Piglet | (observed, confidence=1.0) | How do you spell 'love'?

Entailed Premises:
Piglet | (inferred, confidence=0.7) | You don't spell it...
Pooh | (inferred, confidence=0.7) | You feel it.

Throughline:
Piglet and Pooh both express that the word "love" is not spelled correctly, yet they describe the feeling of love differently.
```

**Generated**

```text
Non-Entailed Premises:
Piglet | (inferred, confidence=0.7) | "How do you spell"

Entailed Premises:

Piglet | (observed, confidence=N) | "How do You spell"

Throughline:
Piglet's question about spelling "love" is answered by Pooh's statement that the word is not spelled but felt.
```

## Example 14  (score: 0.9500)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "It was a mistake," you said. But the cruel thing was, it felt like the mistake was mine, for trusting you."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
trust | (inferred, confidence=0.7) | trust

Entailed Premises:
mistake | (inferred, confidence=0.7) | mistake

Throughline:
The quote implies that the speaker believes the error was their own fault, specifically that they trusted the other person.
```

**Generated**

```text
Non-Entailed Premises:
trust | (inferred, confidence=0.7) | you

Entailed Premises:

trust | (inferred|confidence=0.7) , you | (inferred,confidence=0.7), you

Throughline:
The speaker asserts that the mistake was their own, implying that trust was the cause of the error.
```

## Example 15  (score: 0.8750)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "I donâ€™t know if youâ€™ve ever felt like that. That you wanted to sleep for a thousand years. Or just not exist. Or just not be aware that you do exist. Or something like that. I think wanting that is very morbid, but I want it when I get like this. Thatâ€™s why Iâ€™m trying not to think. I just want it all to stop spinning."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
wanting | (inferred, confidence=0.7) | that you want

Entailed Premises:
wanting | (inferred, confidence=0.7) | that you want to sleep for a thousand years

Throughline:
The speaker believes that wanting to sleep for a thousand years is a morbid desire and is willing to stop thinking about it.
```

**Generated**

```text
Non-Entailed Premises:
wanting | (inferred, confidence=0.7) | that you wanted to sleep for

Entailed Premises:

```
wanting | (observed, confidence=N) | that you wanted
```

Throughline:
I want it all to stop spinning.
```

## Example 16  (score: 0.9500)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "However many holy words you read, however many you speak, what good will they do you if you do not act on upon them?"

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
read | (inferred, confidence=0.7) | what good will they do you if you do not act on upon them?

Entailed Premises:
act on upon | (inferred, confidence=0.7) | what good will they do you if you do not act on upon them?

Throughline:
The quote asserts that the effectiveness of holy words depends entirely on the speaker's subsequent action, specifically the act of acting upon them.
```

**Generated**

```text
Non-Entailed Premises:
read | (inferred, confidence=0.7) | what good will they do

Entailed Premises:

```
act on upon them | (inferred,confidence=0.7)   | what good will they
```

Throughline:
The quote asserts that the efficacy of holy words is contingent upon the speaker's action of acting upon them.
```

## Example 17  (score: 0.8750)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "If he loved with all the powers of his puny being, he couldn't love as much in eighty years as I could in a day."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
80 years | (inferred, confidence=0.7) | I could

Entailed Premises:
I could | (inferred, confidence=0.7) | in a day

Throughline:
He could love as much in eighty years as I could in a day.
```

**Generated**

```text
Non-Entailed Premises:
penny | (inferred, confidence=0.7) | being

Entailed Premises:

*   | (inferred,confidence=0.7)| object | inferred | confidence=0.7 | being

Throughline:
If he loved with all powers of his penny being, he couldn'tlove as much in eighty yearsas I could in a day.
```

## Example 18  (score: 0.8300)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "To live is the rarest thing in the world. Most people exist, that is all."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
Most people exist | (inferred, confidence=0.7) | Most people exist

Entailed Premises:
To live is the rarest thing in the world | (inferred, confidence=0.7) | Most people exist

Throughline:
To live is the rarest thing in the world. Most people exist, that is all.
```

**Generated**

```text
Non-Entailed Premises:
Most people exist | (inferred, confidence=0.7) | Most people exist

Entailed Premises:

To live is the rareSTh (inferred, confidence=N) | Most people exist | (in inferred, confidence=N) | To live is the rareST

Throughline:
To live is the rare ST
```

## Example 19  (score: 0.7750)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "Absence diminishes small loves and increases great ones, as the wind blows out the candle and fans the bonfire."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
candle | blows out (inferred, confidence=0.7) | object

Entailed Premises:
candle | fans (inferred, confidence=0.7) | object

Throughline:
The wind blowing out the candle and fans the bonfire diminishes small loves and increases great ones.
```

**Generated**

```text
Non-Entailed Premises:
candle | (observed, confidence=N) | wind blows out

Entailed Premises:

*   (observed, confidence=X) | wind blows out | inferred, confidence=0.7 | small loves

Throughline:
The quote states that the wind blowing out a candle increases great loves, while the absence of a candle diminishes small loves.
```

## Example 20  (score: 0.9500)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "The most terrible poverty is loneliness, and the feeling of being unloved."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
unloved | (inferred, confidence=0.7) | being

Entailed Premises:
being | (inferred, confidence=0.7) | loneliness

Throughline:
The quote states that loneliness is the most terrible form of poverty, and being unloved is the second form of poverty.
```

**Generated**

```text
Non-Entailed Premises:
unloved | (inferred, confidence=0.7) | loneliness

Entailed Premises:

*   loneliness | (inferred,confidence=0.7)|unloved

Throughline:
The quote asserts that loneliness is the most terrible form of poverty, and that being unloved is a feeling of poverty.
```

## Example 21  (score: 0.9500)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "I DON'T CARE!" Harry yelled at them, snatching up a lunascope and throwing it into the fireplace. "I'VE HAD ENOUGH, I'VE SEEN ENOUGH, I WANT OUT, I WANT IT TO END, I DON'T CARE ANYMORE!""You do care," said Dumbledore. He had not flinched or made a single move to stop Harry demolishing his office. His expression was calm, almost detached. "You care so much you feel as though you will bleed to death with the pain of it."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
Harry | (inferred, confidence=0.7) | Dumbledore

Entailed Premises:
Dumbledore | (inferred, confidence=0.7) | Harry

Throughline:
Dumbledore's calm expression and detached demeanor imply that Harry's actions of demolishing his office are not being stopped by Dumbledore.
```

**Generated**

```text
Non-Entailed Premises:
Harry | (inferred, confidence=0.7) | Dumbledore

Entailed Premises:

*   Harry | (inferred,confidence=0.7)| Dumbledore | inferred: "You care so much"

Throughline:
Dumbledore's calm expression and detached demeanor imply that Harry's actions are not merely a reaction to his own pain but a deliberate, uncontrolled act of destruction.
```

## Example 22  (score: 0.9500)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "Headline?" he asked."'Swing Set Needs Home,'" I said."'Desperately Lonely Swing Set Needs Loving Home,'" he said."'Lonely, Vaguely Pedophilic Swing Set Seeks the Butts of Children,'" I said."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
Swing Set Needs Home | (inferred, confidence=0.7) | Desperately Lonely Swing Set Needs Loving Home

Entailed Premises:
Desperately Lonely Swing Set Needs Loving Home | (inferred, confidence=0.7) | Lonely, Vaguely Pedophilic Swing Set Seeks the Butts of Children

Throughline:
He asked a question, then answered it, and then asked a question again, each time with a different emotional tone.
```

**Generated**

```text
Non-Entailed Premises:
Headline | (inferred, confidence=0.7) | Confidence=0.0 | "Headline?" he asks

Entailed Premises:

*   "Headline?" he | asks | "Headline?" | he | asked | "Headline?"
*   "Headline" | he | asks | "Swing Set Needs Home" | I | said | "Desperately Lonely SwingSet Needs Loving Home" | he | said | "Lonely, VaguELY Pedophilic Swing Set | Seeks the Butts | of Children" | I | said

Throughline:
The speaker asks a question about a headline, then answers it by describing a specific swing set that is desperately lonely and seeks the butts of children.
```

## Example 23  (score: 0.9500)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "All you have to do is write one true sentence. Write the truest sentence that you know."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
write one true sentence | (inferred, confidence=0.7) | write the truest sentence that you know

Entailed Premises:
write the truest sentence that you know | (inferred, confidence=0.7) | write one true sentence

Throughline:
You must write the truest sentence that you know.
```

**Generated**

```text
Non-Entailed Premises:
write | (inferred, confidence=0.7) | true

Entailed Premises:

write | (inferred|confidence=0.7)   | true | true

Throughline:
The quote instructs the reader to write a single, true sentence that is the most accurate reflection of their knowledge.
```

## Example 24  (score: 0.8000)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "HopeSmiles from the threshold of the year to come, Whispering 'it will be happier'..."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
HopeSmiles | (inferred, confidence=0.7) | from the threshold of the year to come

Entailed Premises:
Whispering | (inferred, confidence=0.7) | 'it will be happier'

Throughline:
HopeSmiles from the threshold of the year to come, Whispering 'it will be happier'
```

**Generated**

```text
Non-Entailed Premises:
HopeSmiles | (inferred, confidence=0.7) | Whispering 'it willbe happier'

Entailed Premises:

*   The quote asserts that HopeSmiles will be happier.
*   The quote asserts the year will be happier.

Throughline:
The quote asserts that HopeSmills will be happier.
```

## Example 25  (score: 0.9500)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "You can't stay in your corner of the Forest waiting for others to come to you. You have to go to them sometimes."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
Forest | (observed, confidence=1.0) | waiting for others to come to you

Entailed Premises:
Forest | (inferred, confidence=0.7) | go to them sometimes

Throughline:
You must go to them sometimes to stay in the Forest.
```

**Generated**

```text
Non-Entailed Premises:
Forest | (inferred, confidence=0.7) | waiting

Entailed Premises:

```
Forest | (in inferred, confidence=0.9) | waiting
```

Throughline:
You must go to them sometimes.
```

## Example 26  (score: 0.9500)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "Many people, myself among them, feel better at the mere sight of a book."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
book | (observed, confidence=1.0) | many people

Entailed Premises:
book | (observed, confidence=1.0) | myself among them

Throughline:
The quote implies that the speaker feels more comfortable or better off simply being present in the act of reading a book.
```

**Generated**

```text
Non-Entailed Premises:
book | (inferred, confidence=0.7) | many people

Entailed Premises:

book | (inferred|observed, confidence=0.9) | many people

Throughline:
Many people feel better at the mere Sight of a book.
```

## Example 27  (score: 0.7000)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "If books could have more, give more, be more, show more, they would still need readers who bring to them sound and smell and light and all the rest that canâ€™t be in books.	The book needs you."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
books | (observed, confidence=1.0) | the quote

Entailed Premises:
books | (inferred, confidence=0.7) | the quote

Throughline:
The quote asserts that books require specific external qualities (sound, smell, light) to function effectively, implying that the book itself cannot provide these qualities.
```

**Generated**

```text
Non-Entailed Premises:
books | (inferred, confidence=0.7) | need readers who bring tothem sound and smell and light

Entailed Premises:

*   The book needs you.
```

## Example 28  (score: 0.8750)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "Everyone has talent. What's rare is the courage to follow it to the dark places where it leads."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
talent | follows (inferred, confidence=0.7) | courage

Entailed Premises:
courage | follows (inferred, confidence=0.7) | talent

Throughline:
The quote asserts that talent is a universal human trait, and the specific quality of courage is the necessary action required to pursue it.
```

**Generated**

```text
Non-Entailed Premises:
Everyone | (observed, confidence=N) | talent

Entailed Premises:

Everyone | (inferred, confidence=0.7) | talent

Throughline:
Everyone has talent, and the courage to follow it is rare.
```

## Example 29  (score: 0.9500)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "We're all human, aren't we? Every human life is worth the same, and worth saving."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
Every human life is worth the same | (inferred, confidence=0.7) | worth saving

Entailed Premises:
We're all human, aren't we? | (inferred, confidence=0.7) | every human life is worth the same

Throughline:
We are all human, and every human life is worth the same.
```

**Generated**

```text
Non-Entailed Premises:
Every human life is worth thesame | (inferred, confidence=0.7) | worth saving

Entailed Premises:

We're all human, weren't we? | (inferred,confidence=0.7)| worth saving

Throughline:
Every human life being worth the same is a fundamental truth that justifies saving it.
```

## Example 30  (score: 0.9500)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "I am no bird; and no net ensnares me: I am a free human being with an independent will."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
free human being | (inferred, confidence=0.7) | independent will

Entailed Premises:
independent will | (observed, confidence=1.0) | free human being

Throughline:
The quote asserts that the subject is a free human being with an independent will, which implies the subject is not enslaved by a net.
```

**Generated**

```text
Non-Entailed Premises:
free human being | (inferred, confidence=0.7) | independent will

Entailed Premises:

*   independent will | (inferred,confidence=0.7)|independent will

Throughline:
The quote asserts that the subject is a free human being with independent will, which implies the subject is not ensnared by a net.
```


---

# Holdout examples — tier3_convergence

**Pass:** ✓  |  **Avg score:** 0.8606  |  **N:** 20

## Check rates

| Check | Rate | Status |
|---|---:|:---:|
| headers | 90% | ✓ |
| entailed_non_empty | 95% | ✓ |
| pipes_well_formed | 100% | ✓ |
| no_template_leakage | 100% | ✓ |
| both_tag_types | 40% | ✓ |
| confidence_numeric | 80% | ✓ |
| canonical_tag_format | 85% | ✓ |
| sections_distinct | 100% | ✓ |
| verbatim_entailed | 85% | ✓ |
| avg_score_tier3 | 85% | ✓ |

## Example 1  (score: 0.7000)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "Tis the privilege of friendship to talk nonsense, and to have her nonsense respected."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
friendship | (observed, confidence=1.0) | talk nonsense

Entailed Premises:
friendship | (inferred, confidence=0.7) | respected

Throughline:
Friendship is the privilege of having her nonsense respected.
```

**Generated**

```text
Non-Entailed Premises:
friendship | (inferred, confidence=0.7) | talk nonsense

Entailed Premises:

*   (inferred, confidence=N) | friendship | (inferred,confidence=N) | talk nonsense

Throughline:
friendship | talk nonsense
```

## Example 2  (score: 0.8000)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "All God does is watch us and kill us when we get boring. We must never, ever be boring."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
watch us and kill us when we get boring | (inferred, confidence=0.7) | confidence=0.0 | never

Entailed Premises:
watch us and kill us when we get boring | (observed, confidence=1.0) | confidence=1.0 | we must never

Throughline:
We must never be boring.
```

**Generated**

```text
Non-Entailed Premises:
watch us | (observed, confidence=N) | kill us when we get bored

Entailed Premises:

*   watch us | (inferred, confidence=N) | when we get boring

Throughline:
*   We must never, ever become boring.
```

## Example 3  (score: 0.7000)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "Anyone who thinks sitting in church can make you a Christian must also think that sitting in a garage can make you a car."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
garage | (inferred, confidence=0.7) | car

Entailed Premises:
sitting in church | (inferred, confidence=0.7) | make you a Christian
sitting in a garage | (inferred, confidence=0.7) | make you a car

Throughline:
Anyone who thinks sitting in church can make you a Christian must also think that sitting in a garage can make you a car.
```

**Generated**

```text
Non-Entailed Premises:
garage | (inferred, confidence=0.7) | car

Entailed Premises:

*   Anyone who thinks sitting in church must also think that sitting on a garage can make you car.
```

## Example 4  (score: 0.9500)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "I don't go out with strangers," I said."Good thing I do. I'll pick you up at five."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
I don't go out with strangers | (inferred, confidence=0.7) | I said

Entailed Premises:
I don't go out with strangers | (observed, confidence=1.0) | I said

Throughline:
I said I don't go out with strangers, which implies I will pick you up at five.
```

**Generated**

```text
Non-Entailed Premises:
I don't go outwith strangers | (inferred, confidence=0.7) | I said

Entailed Premises:

I don't go out | (inferred,confidence=0.7)| I said

Throughline:
I said "I don't goout with strangers"
```

## Example 5  (score: 0.9500)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "The truth does not change according to our ability to stomach it."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
stomach | stomach (inferred, confidence=0.7) | true

Entailed Premises:
stomach | true (inferred, confidence=0.7) | truth

Throughline:
The truth remains constant regardless of whether the subject can stomach the object.
```

**Generated**

```text
Non-Entailed Premises:
stomach | (inferred, confidence=0.7) | ability

Entailed Premises:

*   The truth does not change according our ability to stomach it. | (inferred,confidence=0.7)| confidence=0.99 | truth

Throughline:
The truth does not change regardless of our ability to stomach it
```

## Example 6  (score: 0.8350)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "I like your Christ, I do not like your Christians. Your Christians are so unlike your Christ."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
Christ | (observed, confidence=1.0) | Christ

Entailed Premises:
Christ | (observed, confidence=1.0) | Christians

Throughline:
The speaker claims that the speaker dislikes their own Christian identity because it is so unlike their Christ.
```

**Generated**

```text
Non-Entailed Premises:
Christ | (observed, confidence=N) | Christians

Entailed Premises:

```
Christ | (inferred, confidence=0.7) | Christians
```

Throughline:
The speaker asserts that their preference for Christ is distinct from their preference for Christians, implying that Christians are fundamentally unlike Christ.
```

## Example 7  (score: 0.8000)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "That's the thing about books. They let you travel without moving your feet."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
books | (observed, confidence=1.0) | travel without moving your feet

Entailed Premises:
books | (inferred, confidence=0.7) | travel without moving your feet

Throughline:
Books allow you to travel without having to move your feet.
```

**Generated**

```text
Non-Entailed Premises:
books | (observed, confidence=N) | travel without moving your feet

Entailed Premises:

books | (inferred, confidence=N) | let you travel without moving你的 feet

Throughline:
books let you travel without moving feet
```

## Example 8  (score: 0.8750)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "It means 'Shadowhunters: Looking Better in Black Than the Widows of our Enemies Since 1234'."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
Shadowhunters | (observed, confidence=1.0) | Looking Better in Black Than the Widows of our Enemies Since 1234

Entailed Premises:
Shadowhunters | (inferred, confidence=0.7) | Looking Better in Black Than the Widows of our Enemies Since 1234

Throughline:
Shadowhunters is a reference to the movie "Shadowhunters: Looking Better in Black Than the Widows of our Enemies Since 1234".
```

**Generated**

```text
Non-Entailed Premises:
Shadowhunters | (inferred, confidence=0.7) | Looking Better in Black Than

Entailed Premises:

Shadowhunters | (observed, confidence=N) | Since 1239

Throughline:
Shadowhunters: Looking better in black than the widows of our enemies since 1239
```

## Example 9  (score: 0.9300)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "When one is in love, one always begins by deceiving one's self, and one always ends by deceiving others. That is what the world calls a romance."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
deceiving one's self | (observed, confidence=1.0) | one always begins by deceiving one's self

Entailed Premises:
deceiving one's self | (observed, confidence=1.0) | one always ends by deceiving others

Throughline:
One always begins by deceiving one's self and always ends by deceiving others.
```

**Generated**

```text
Non-Entailed Premises:
deceiving one's self | (inferred, confidence=0.7) | deceiving one's self

Entailed Premises:

```
deceiving one'sself | (inferred,confidence=0.7)| deceiving others
```

Throughline:
One always begins by deceiving oneself and always ends by deceiving other.
```

## Example 10  (score: 0.9500)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "We're all human, aren't we? Every human life is worth the same, and worth saving."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
Every human life is worth the same | (inferred, confidence=0.7) | worth saving

Entailed Premises:
We're all human, aren't we? | (inferred, confidence=0.7) | every human life is worth the same

Throughline:
We are all human, and every human life is worth the same.
```

**Generated**

```text
Non-Entailed Premises:
Every human life is worth thesame | (inferred, confidence=0.7) | worth saving

Entailed Premises:

We're all human, weren't we? | (inferred,confidence=0.7)| worth saving

Throughline:
We are all human, and every human life is worth the saving.
```

## Example 11  (score: 0.9500)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "Never be bullied into silence. Never allow yourself to be made a victim. Accept no one's definition of your life, but define yourself."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
Never allow yourself to be made a victim | (inferred, confidence=0.7) | Never allow yourself to be made a victim

Entailed Premises:
Accept no one's definition of your life | (inferred, confidence=0.7) | Accept no one's definition of your life

Throughline:
Never allow yourself to be made a victim.
```

**Generated**

```text
Non-Entailed Premises:
Never allow yourself to be made into a victim | (inferred, confidence=0.7) | confidence=0.0 | "Never allow yourself to be make a victim"

Entailed Premises:

*   "Never allow yourself to make a victim" | (inferred,confidence=0.7)|confidence=0.0 |"Never allow yourself to be a victim"

Throughline:
*   "Never allow oneself to be made a victim"
```

## Example 12  (score: 0.8367)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "I have nothing to declare except my genius."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
genius | (observed, confidence=1.0) | nothing

Entailed Premises:
nothing | (observed, confidence=1.0) | declare

Throughline:
The speaker claims that only their inherent genius is available for declaration.
```

**Generated**

```text
Non-Entailed Premises:
genius | (inferred, confidence=0.7) | nothing

Entailed Premises:

nothing | (inferred,confidence=0.7)| nothing

throughline:
genius | inferred | nothing
```

## Example 13  (score: 0.8550)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "In the beginning there was nothing, which exploded."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
nothing | (observed, confidence=1.0) | exploded

Entailed Premises:
nothing | (inferred, confidence=0.7) | nothing

Throughline:
nothing exploded.
```

**Generated**

```text
Non-Entailed Premises:
nothing | (observed, confidence=N) | exploded

Entailed Premises:

nothing | (inferred, confidence=0.7) | nothing

Throughline:
nothing exploded.
```

## Example 14  (score: 0.8350)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "The one you love and the one who loves you are never, ever the same person."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
Love | (observed, confidence=1.0) | the one who loves you

Entailed Premises:
Love | (inferred, confidence=0.7) | the one you love

Throughline:
The one you love and the one who loves you are never, ever the same person.
```

**Generated**

```text
Non-Entailed Premises:
Love | (observed, confidence=N) | the one you love

Entailed Premises:

Love | (inferred, confidence=0.7) | the one who loves you

Throughline:
The one you love and The one who loves you are not the same person.
```

## Example 15  (score: 0.9500)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "One of the advantages of being disorganized is that one is always having surprising discoveries."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
disorganized | (observed, confidence=1.0) | one is always having surprising discoveries

Entailed Premises:
disorganized | (inferred, confidence=0.7) | one is always having surprising discoveries

Throughline:
One of the advantages of being disorganized is that one is always having surprising discoveries.
```

**Generated**

```text
Non-Entailed Premises:
disorganized | (inferred, confidence=0.7) | one is always having surprising discovers

Entailed Premises:

one is always having surprising discovers | (inferred,confidence=0.7)| one is having surprising discoveries

Throughline:
One of the advantages of disorganization is that one is always discovering surprising things.
```

## Example 16  (score: 0.9500)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "All you have to do is write one true sentence. Write the truest sentence that you know."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
write one true sentence | (inferred, confidence=0.7) | write the truest sentence that you know

Entailed Premises:
write the truest sentence that you know | (inferred, confidence=0.7) | write one true sentence

Throughline:
You must write the truest sentence that you know.
```

**Generated**

```text
Non-Entailed Premises:
write one true sentence | (inferred, confidence=0.7) | write the truest sentence that know

Entailed Premises:

write the truest sentence that | (inferred,confidence=0.7) know | write one true sentence

Throughline:
write the truest sentence is the only true sentence
```

## Example 17  (score: 0.8350)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "I am nothing special, of this I am sure. I am a common man with common thoughts and I've led a common life. There are no monuments dedicated to me and my name will soon be forgotten, but I've loved another with all my heart and soul, and to me, this has always been enough.."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
common man | with (inferred, confidence=0.7) | common thoughts

Entailed Premises:
common man | loved (inferred, confidence=0.7) | with all my heart and soul

Throughline:
The speaker asserts that despite being a common man with common thoughts, their love for another person remains the only thing that makes them special.
```

**Generated**

```text
Non-Entailed Premises:
common man | (inferred, confidence=0.7) | common man

Entailed Premises:

```
common man | (observed, confidence=N) | common man
```

Throughline:
The speaker asserts that their identity as a "common man" with "common thoughts" and a "common life" is sufficient to justify their love for another, despite the absence of monuments or a lasting name.
```

## Example 18  (score: 0.6500)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "To the well-organized mind, death is but the next great adventure."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
death | (inferred, confidence=0.7) | the next great adventure

Entailed Premises:
death | (inferred, confidence=0.7) | the next great adventure adventure

Throughline:
death is the next great adventure for the well-organized mind
```

**Generated**

```text
Non-Entailed Premises:
death | (observed, confidence=N) | the next great adventure

Entailed Premises:

*   (observed, confidence=X) | the well-organized mind | (inferred, confidence=X) | death is but the nextgreat adventure

Throughline:
The quote implies that death is a metaphor for a new, exciting challenge for a well-organized mind.
```

## Example 19  (score: 0.9500)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "Fate is like a strange, unpopular restaurant filled with odd little waiters who bring you things you never asked for and don't always like."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
Fate | is (inferred, confidence=0.7) | strange | unpopular | restaurant | filled | with | odd | little | waiters | who | bring | you | things | you | never | asked | for | and | don't | always | like |

Entailed Premises:
Fate | is (inferred, confidence=0.7) | like | a | strange | unpopular | restaurant | filled | with | odd | little | waiters | who | bring | you | things | you | never | asked | for | and | don't | always | like |

Throughline:
Fate is like a strange, unpopular restaurant filled with odd little waiters who bring you things you never asked for and don't always like.
```

**Generated**

```text
Non-Entailed Premises:
Fate | (inferred, confidence=0.7) | strange

Entailed Premises:

Fate | (in inferred, confidence=0.9) | like

Throughline:
Fate is like a weird, unpopular restaurant with odd waiters who bring you items you never asked for.
```

## Example 20  (score: 0.9100)

**Prompt**

```text
Given this quote, extract the implicit reasoning.

Quote: "Be mindful. Be grateful. Be positive. Be true. Be kind."

Generate a response with:
1. Non-Entailed Premises
2. Entailed Premises
3. Throughline

Format each premise as: subject | (tag, confidence=N) | object
- tag: "observed" for explicit facts, "inferred" for derived facts
- confidence: a decimal number, e.g. 1.0 for observed facts, 0.7 for inferred

VERBATIM EXTRACTION RULE (Entailed Premises only):
- Subject and object MUST be exact, verbatim text copied from the quote above.
- Do NOT paraphrase, summarize, or invent language for Entailed subject/object fields.
- If verbatim text needs clarification, add a parenthetical transliteration
  immediately after: verbatim text (clarification if needed)
  Example: "unexamined life (a life without self-reflection) | lacks (observed, confidence=1.0) | worth"
- Non-Entailed Premises and the Throughline may use your own words.

IMPORTANT: The Entailed Premises section MUST contain at least one triplet.
Never leave Entailed Premises empty.

Response:
```

**Expected**

```text
Non-Entailed Premises:
mindful | (inferred, confidence=0.7) | grateful

Entailed Premises:
mindful | (inferred, confidence=0.7) | positive

Throughline:
Be mindful, be grateful, be positive, be true, be kind.
```

**Generated**

```text
Non-Entailed Premises:
Be mindful | (inferred, confidence=0.7) | Be mindful

Entailed Premises:

Be mindful | (in inferred, confidence=0.9) | Be mindful

Throughline:
Be mindful.
```

