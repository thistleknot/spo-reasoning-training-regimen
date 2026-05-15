# SPO Holdout Inference Examples

**Adapter:** output/spo_verbatim_3ep_v7/adapter
**Examples:** 20

---

## Example 1 — score: 0.650

**Input quote:** but for my own part, if a book is well written, I always find it too short.

**Gold output:**
```
Non-Entailed Premises:
book | observed | well written

Entailed Premises:
book | inferred | too short

Throughline:
The author admits that a well-written book is often too short for them.
```

**Model output:**
```
Non-Entailed Premises:
book | observed | well written

Entailed Premises:

book | inferred | too short

Throughline:
The author states that they find a book too short if it is well written.
```

---

## Example 2 — score: 0.650

**Input quote:** Why do you go away? So that you can come back. So that you can see the place you came from with new eyes and extra colors. And the people there see you differently, too. Coming back to where you started is not the same as never leaving.

**Gold output:**
```
Non-Entailed Premises:
*   "Why do you go away?" | inferred|confidence=0.8 | "So that you can come back"

Entailed Premises:
*   "So that you can come back" | inferred|confidence=0.8 | "So that you can see the place you came from with new eyes and extra colors"

Throughline:
*   The quote implies that returning to the origin is not the same as never leaving, suggesting that the experience of returning is distinct from the experience of never leaving.
```

**Model output:**
```
Non-Entailed Premises:
people | inferred | go away

Entailed Premises:

people | inferred | come back

Throughline:
The quote implies that returning to the origin is not the same as leaving, and that the act of coming back allows for a new perspective and experience.
```

---

## Example 3 — score: 0.650

**Input quote:** Once you learn to read, you will be forever free.

**Gold output:**
```
Non-Entailed Premises:
once | inferred | learn | to | read | you | will | be | forever | free

Entailed Premises:
once | inferred | learn | to | read | you | will | be | forever | free
once | inferred | learn | to | read | you | will | be | forever | free | learn | to | read | you | will | be | forever | free | learn | to | read | you | will | be | forever | free | learn | to | read | you | will | be | forever | free | learn | to | read | you | will | be | forever | free | learn | to | read | you | will | be | forever | free | learn | to | read | you | will | be | forever | free | learn | to | read | you | will | be | forever | free | learn | to | read | you | will | be | forever | free | learn | to | read | you | will | be | forever | free | learn | to | read | you | will | be | forever | free | learn | to | read | you | will | be | forever | free | learn | to | read | you | will | be | forever | free | learn | to | read | you | will | be | forever | free | learn | to | read | you | will | be | forever | free | learn | to | read | you | will | be | forever | free | learn | to | read | you | will | be | forever | free | learn | to | read | you | will | be | forever | free | learn | to | read | you | will | be | forever | free | learn | to | read | you | will | be | forever | free | learn | to | read | you | will | be | forever | free | learn | to | read | you | will | be | forever | free | learn | to | read | you | will |

Throughline:
N/A
```

**Model output:**
```
Non-Entailed Premises:
read | inferred | forever free

Entailed Premises:

read | inferred | forever

Throughline:
Once you learn to read will be forever free.
```

---

## Example 4 — score: 0.650

**Input quote:** People shouldn't be afraid of their government. Governments should be afraid of their people.

**Gold output:**
```
Non-Entailed Premises:
government | feared | people

Entailed Premises:
people | afraid | government

Throughline:
People should not fear their government, and governments should not fear their people.
```

**Model output:**
```
Non-Entailed Premises:
government | inferred | people

Entailed Premises:

people | inferred | government

Throughline:
People should not be afraid of their government, and governments should be afraid of their own people.
```

---

## Example 5 — score: 0.618

**Input quote:** Waiting is painful. Forgetting is painful. But not knowing which to do is the worst kind of suffering.

**Gold output:**
```
Non-Entailed Premises:
Waiting | inferred | painful

Entailed Premises:
Forgetting | inferred | painful

Throughline:
Waiting, forgetting, and knowing which to do are all forms of suffering.
```

**Model output:**
```
Non-Entailed Premises:
Waiting | inferred | painful

Entailed Premises:

| observed | inferred | painful |
| :--- | :--- | :---
| | | |

Throughline:
Waiting is painful.
```

---

## Example 6 — score: 0.650

**Input quote:** Alone. Yes, that's the key word, the most awful word in the English tongue. Murder doesn't hold a candle to it and hell is only a poor synonym.

**Gold output:**
```
Non-Entailed Premises:
"Alone" | observed|inferred, confidence=0.0 | "the most awful word in the English tongue"

Entailed Premises:
"Alone" | inferred|inferred, confidence=0.0 | "the most awful word in the English tongue"

Throughline:
The quote asserts that the word "Alone" is the most awful word in the English tongue, implying that "Alone" is the key word in the context of the sentence.
```

**Model output:**
```
Non-Entailed Premises:
Alone | inferred | the most awful word in English tongue

Entailed Premises:

Alone | inferred | murder doesn't hold a candle

Throughline:
Alone is the most awful word in english tongue
```

---

## Example 7 — score: 0.650

**Input quote:** I wonder how many people don't get the one they want, but end up with the one they're supposed to be with.

**Gold output:**
```
Non-Entailed Premises:
people | inferred | don't get the one they want

Entailed Premises:
people | inferred | end up with the one they're supposed to be with

Throughline:
The quote suggests that people often fail to realize the one they are meant to be with.
```

**Model output:**
```
Non-Entailed Premises:
people | inferred | don't get the one

Entailed Premises:

people | inferred | end up with the one

Throughline:
The quote suggests that a common human experience is people who do not receive what they desire but instead end up with the one that is supposed to be with them.
```

---

## Example 8 — score: 0.650

**Input quote:** Imagine smiling after a slap in the face. Then think of doing it twenty-four hours a day.

**Gold output:**
```
Non-Entailed Premises:
"Imagine smiling after a slap in the face." | inferred|confidence=0.0 | "do it twenty-four hours a day"

Entailed Premises:
"do it twenty-four hours a day" | inferred|confidence=0.0 | "smiling after a slap in the face"

Throughline:
The quote implies that one should imagine smiling after a slap in the face and then perform this action twenty-four hours a day.
```

**Model output:**
```
Non-Entailed Premises:
smiling | inferred | after a slap in the_face

Entailed Premises:

After a slap in the face | inferred | think of doing it twenty four hours a day

Throughline:
The quote suggests that smiling after a slap in a face is a metaphor for doing it twenty-four hours per day.
```

---

## Example 9 — score: 0.725

**Input quote:** They who can give up essential liberty to obtain a little temporary safety deserve neither liberty nor safety.

**Gold output:**
```
Non-Entailed Premises:
"necessary" | inferred | "necessary"

Entailed Premises:
<subject> | <relation> (observed|inferred, confidence=X.X) | <object>

Throughline:
The quote asserts that individuals who voluntarily surrender essential liberty in exchange for temporary safety are not entitled to either liberty or safety.
```

**Model output:**
```
Non-Entailed Premises:
<liberty> | <inferred> (confidence=0.9) | <safety>

Entailed Premises:

<liberty> | >inferred> (confidence=X) | <safety>X>

Throughline:
The quote asserts that individuals who voluntarily surrender essential liberty to achieve temporary safety are morally and logically incapable of holding either liberty or safety.
```

---

## Example 10 — score: 0.650

**Input quote:** Is it possible for home to be a person and not a place?

**Gold output:**
```
Non-Entailed Premises:
home | inferred | person

Entailed Premises:
home | inferred | place

Throughline:
home can be both a person and a place.
```

**Model output:**
```
Non-Entailed Premises:
home | inferred | person

Entailed Premises:

home | inferred | person
home | inferred | place

Throughline:
home can be a person and not be a place
```

---

## Example 11 — score: 0.650

**Input quote:** If we find ourselves with a desire that nothing in this world can satisfy, the most probable explanation is that we were made for another world.

**Gold output:**
```
Non-Entailed Premises:
desire | satisfies | world

Entailed Premises:
world | made | for

Throughline:
We were made for another world.
```

**Model output:**
```
Non-Entailed Premises:
desire | inferred | nothing in this world can

Entailed Premises:

desire | inferred | we were made for another

Throughline:
The quote suggests that a desire for nothing in the world can only be explained by being made for another world.
```

---

## Example 12 — score: 0.650

**Input quote:** When things break, it's not the actual breaking that prevents them from getting back together again. It's because a little piece gets lost - the two remaining ends couldn't fit together even if they wanted to. The whole shape has changed.

**Gold output:**
```
Non-Entailed Premises:
lost | inferred | the two remaining ends couldn't fit together even if they wanted to

Entailed Premises:
the two remaining ends couldn't fit together even if they wanted to | inferred | the whole shape has changed

Throughline:
The two remaining ends couldn't fit together even if they wanted to because the whole shape has changed.
```

**Model output:**
```
Non-Entailed Premises:
lost | inferred | the two remaining ends couldn'T fit together even if they want to

Entailed Premises:

lost | inferred | the whole shape has changed

Throughline:
The quote states that the inability of the two remaining ends to fit together is the reason things break, not the actual breaking itself.
```

---

## Example 13 — score: 0.590

**Input quote:** Love is not affectionate feeling, but a steady wish for the loved person's ultimate good as far as it can be obtained.

**Gold output:**
```
Non-Entailed Premises:
steady wish for the loved person's ultimate good as far as it can be obtained | inferred | love is not affectionate feeling

Entailed Premises:
steady wish for the loved person's ultimate good as far as it can be obtained | inferred | love is not affectionate feeling
steady wish for the loved person's ultimate good as far as it can be obtained | observed | love is not affectionate feeling, but a steady wish for the loved person's ultimate good as far as it can be obtained

Throughline:
N/A
```

**Model output:**
```
Non-Entailed Premises:
Love | is | a steady wish for the beloved person's ultimate good as it can be obtained

Entailed Premises:

Love | is | a | steady | wish | for | the | loved | person's | ultimate | good | as | far | as | it | can | be | obtained

Throughline:
Love is not affectionate, but a steady wish.
```

---

## Example 14 — score: 0.650

**Input quote:** Maybe this world is another planetâ€™s hell.

**Gold output:**
```
Non-Entailed Premises:
world | inferred | hell

Entailed Premises:
world | inferred | planet

Throughline:
The quote suggests that the world is a hell for a planet.
```

**Model output:**
```
Non-Entailed Premises:
world | inferred | hell

Entailed Premises:

world | inferred | planet's hell

Throughline:
The quote suggests that the world is a metaphor for hell.
```

---

## Example 15 — score: 0.650

**Input quote:** Books have to be heavy because the whole world's inside them.

**Gold output:**
```
Non-Entailed Premises:
books | observed | the whole world's inside them

Entailed Premises:
books | inferred | the whole world's inside them

Throughline:
books are heavy because the whole world's inside them
```

**Model output:**
```
Non-Entailed Premises:
books | inferred | world

Entailed Premises:

books | inferred | world's inside them

Throughline:
Books are heavy because the whole world is inside them.
```

---

## Example 16 — score: 0.650

**Input quote:** For me, I am driven by two main philosophies: know more today about the world than I knew yesterday and lessen the suffering of others. You'd be surprised how far that gets you.

**Gold output:**
```
Non-Entailed Premises:
know more today about the world than I knew yesterday | inferred|confidence=0.0 | suffer

Entailed Premises:
know more today about the world than I knew yesterday | inferred|confidence=0.0 | lessen the suffering of others

Throughline:
knowing more about the world today is a way to lessen the suffering of others.
```

**Model output:**
```
Non-Entailed Premises:
know more today about the world | inferred | today

Entailed Premises:

<subject> | <relation> (observed|inferred, confidence=X.X) | <object>

Throughline:
The quote states that the speaker is driven by two main philosophics: knowing more about the world today and lessening the suffering of others.
```

---

## Example 17 — score: 0.787

**Input quote:** Promise me you'll never forget me because if I thought you would, I'd never leave.

**Gold output:**
```
Non-Entailed Premises:
promise | never forget | me

Entailed Premises:
promise | will never leave | me

Throughline:
The speaker promises the listener that they will never forget the speaker because the listener would never leave.
```

**Model output:**
```
Non-Entailed Premises:
promise | never | forget

Entailed Premises:

subject | relation (observed|inferred, confidence=0.9) | object
promise | never | never leave

Throughline:
The speaker promises the listener that they will never forget the speaker because the speaker would never leave.
```

---

## Example 18 — score: 0.650

**Input quote:** Stories can conquer fear, you know. They can make the heart bigger.

**Gold output:**
```
Non-Entailed Premises:
stories | inferred | conquer

Entailed Premises:
stories | inferred | make

Throughline:
stories can make the heart bigger.
```

**Model output:**
```
Non-Entailed Premises:
stories | inferred | conquer

Entailed Premises:

stories | inferred | make the heart bigger

Throughline:
Stories can conquer fear by making the heart bigger.
```

---

## Example 19 — score: 0.650

**Input quote:** All I ever wanted was to reach out and touch another human being not just with my hands but with my heart.

**Gold output:**
```
Non-Entailed Premises:
touching | with | human

Entailed Premises:
touching | with | heart

Throughline:
The quote asserts that the speaker's ultimate desire is to physically reach out and emotionally connect with another human being.
```

**Model output:**
```
Non-Entailed Premises:
touch | observed | another human being

Entailed Premises:

touch | inferred | another human being

Throughline:
The quote implies that the speaker's primary motivation was to connect with another human being through a physical and emotional bond.
```

---

## Example 20 — score: 0.650

**Input quote:** Perhaps when we find ourselves wanting everything, it is because we are dangerously close to wanting nothing.

**Gold output:**
```
Non-Entailed Premises:
dangerously close to wanting nothing | inferred | wanting nothing

Entailed Premises:
wants nothing | inferred | wanting nothing

Throughline:
wants nothing is dangerous because it is dangerously close to wanting nothing
```

**Model output:**
```
Non-Entailed Premises:
dangerously close | inferred | wanting nothing

Entailed Premises:

wanting nothing | inferred | wanting everything

Throughline:
The quote suggests that a state of wanting everything is a dangerous precursor to a state of wanting nothing.
```

---
