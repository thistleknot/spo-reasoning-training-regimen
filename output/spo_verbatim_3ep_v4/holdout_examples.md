# SPO Holdout Inference Examples

**Adapter:** output/spo_verbatim_3ep_v4/adapter
**Examples:** 20

---

## Example 1 — score: 0.610

**Input quote:** Do you remember back at the hotel when you promised that if we lived, youâ€™d get dressed up in a nurseâ€™s outfit and give me a sponge bath?" asked Jace."It was Simon who promised you the sponge bath.""As soon as Iâ€™m back on my feet, handsome," said Simon."I knew we should have left you a rat.

**Gold output:**
```
Non-Entailed Premises:
Jace | asked | Jace

Entailed Premises:
Simon | said | Simon

Throughline:
Simon promised the sponge bath to Jace.
```

**Model output:**
```
Non-Entailed Premises:
Jace | inferred | Jace

Entailed Premises:

Throughline:
Jace | observed | Jace
```

---

## Example 2 — score: 0.650

**Input quote:** It's not the load that breaks you down, it's the way you carry it.

**Gold output:**
```
Non-Entailed Premises:
*   "It's not the load that breaks you down" | inferred|confidence=0.99 | "it's"

Entailed Premises:
*   "It's not the load that breaks you down" | inferred|confidence=0.99 | "it's"

Throughline:
*   The quote asserts that the method of carrying a load rather than its weight is the primary factor in breaking someone down.
```

**Model output:**
```
Non-Entailed Premises:
way you carry it | inferred | it's the way you

Entailed Premises:

way you carry it | observed | it's the way

Throughline:
The quote states that the way you carry the load is the factor that breaks you down, not the load itself.
```

---

## Example 3 — score: 0.650

**Input quote:** It is what you read when you don't have to that determines what you will be when you can't help it.

**Gold output:**
```
Non-Entailed Premises:
*   "It" | inferred | "what"
*   "don't" | inferred | "that"
*   "determines" | inferred | "what"
*   "you" | inferred | "will"
*   "be" | inferred | "when"
*   "can't" | inferred | "help"
*   "it" | inferred | "it"

Entailed Premises:
*   "It" | inferred | "what"
*   "you" | inferred | "will"
*   "don't" | inferred | "that"
*   "determines" | inferred | "what"
*   "what" | inferred | "what"
*   "when" | inferred | "when"
*   "can't" | inferred | "help"
*   "it" | inferred | "it"

Throughline:
*   "It is what you read when you don't have to that determines what you will be when you can't help it."
```

**Model output:**
```
Non-Entailed Premises:
read | inferred | what you will be

Entailed Premises:

read | inferred | what will be when you can’t help it

Throughline:
The quote states that the content of what you read when you are not required to determine what you will be when it is not possible to help.
```

---

## Example 4 — score: 0.650

**Input quote:** Piglet sidled up to Pooh from behind. "Pooh!" he whispered."Yes, Piglet?""Nothing," said Piglet, taking Pooh's paw. "I just wanted to be sure of you.

**Gold output:**
```
Non-Entailed Premises:
Piglet | inferred | Pooh

Entailed Premises:
Piglet | inferred | Pooh

Throughline:
Piglet's statement that he just wanted to be sure of Pooh confirms the premise that he was trying to verify the identity of the animal.
```

**Model output:**
```
Non-Entailed Premises:
Piglet | inferred | Pooh

Entailed Premises:

Throughline:
Piglet whispered to Pooh
```

---

## Example 5 — score: 0.650

**Input quote:** I have the choice of being constantly active and happy or introspectively passive and sad. Or I can go mad by ricocheting in between.

**Gold output:**
```
Non-Entailed Premises:
active and happy | inferred | constantly

Entailed Premises:
constantly active and happy | inferred | introspectively passive and sad

Throughline:
The quote presents a choice between being constantly active and happy or introspectively passive and sad, or alternatively, going mad by ricocheting in between.
```

**Model output:**
```
Non-Entailed Premises:
active | inferred | constantly

Entailed Premises:

active | inferred | constantly
introspectively | inferred | passive
happy | inferred | constantly
sad | inferred | introspectively
passive | inferred | introspectingly
mad | inferred | ricocheting

Throughline:
The quote presents a binary choice between active/happy and introspectively/passive/sad states, with the final option being a state of madness resulting from ricocheting.
```

---

## Example 6 — score: 0.650

**Input quote:** Nothing takes the taste out of peanut butter quite like unrequited love.

**Gold output:**
```
Non-Entailed Premises:
unrequited love | inferred | nothing

Entailed Premises:
unrequited love | observed | nothing

Throughline:
unrequited love is the only thing that takes the taste out of peanut butter.
```

**Model output:**
```
Non-Entailed Premises:
unrequited love | inferred | nothing

Entailed Premises:

Throughline:
unrequited Love is the only thing that takes the taste out of Peanut Butter.
```

---

## Example 7 — score: 0.650

**Input quote:** Then the Grinch thought of something he hadn't before! What if Christmas, he thought, doesn't come from a store. What if Christmas...perhaps...means a little bit more!

**Gold output:**
```
Non-Entailed Premises:
Grinch thought | inferred | Grinch thought

Entailed Premises:
Grinch thought | inferred | Grinch thought

Throughline:
Grinch thought of something he hadn't before!
```

**Model output:**
```
Non-Entailed Premises:
Christmas | inferred | means a little bit more

Entailed Premises:

Throughline:
The Grinch thought of something that hadn't been before.
```

---

## Example 8 — score: 0.650

**Input quote:** And still, after all this time,The sun never says to the earth,"You owe Me."Look what happens withA love like that,It lights the Whole Sky.

**Gold output:**
```
Non-Entailed Premises:
*   "And still, after all this time," | inferred | "The sun never says to the earth"
*   "You owe Me." | inferred | "Look what happens withA love like that,It lights the Whole Sky."

Entailed Premises:
*   "And still, after all this time," | inferred | "The sun never says to the earth"
*   "You owe Me." | inferred | "Look what happens withA love like that,It lights the Whole Sky."

Throughline:
*   The quote establishes a paradoxical relationship where the sun's silence ("never says to the earth") serves as a metaphor for the speaker's debt ("You owe Me"), which in turn leads to a conclusion that the described love ("A love like that") is the source of the sky's illumination ("It lights the Whole Sky").
```

**Model output:**
```
Non-Entailed Premises:
The sun | never says to the earth | observed

Entailed Premises:

The sun | never says | inferred

Throughline:
The sun never says to earth, you owe me.
```

---

## Example 9 — score: 0.650

**Input quote:** The only way out of the labyrinth of suffering is to forgive.

**Gold output:**
```
Non-Entailed Premises:
forgive | inferred | the only way out of the labyrinth of suffering

Entailed Premises:
forgive | inferred | the only way out of the labyrinth of suffering

Throughline:
Forging forgiveness is the only way out of the labyrinth of suffering.
```

**Model output:**
```
Non-Entailed Premises:
forgiveness | inferred | the only way out of the

Entailed Premises:

Throughline:
The only way out of a labyrinth of suffering is to be forgiving.
```

---

## Example 10 — score: 0.610

**Input quote:** And so it goes...

**Gold output:**
```
Non-Entailed Premises:
And so it goes | inferred | And so it goes

Entailed Premises:
And so it goes | inferred | And so it goes

Throughline:
The phrase "And so it goes" serves as a concluding statement that implies a continuation of the previous narrative or situation.
```

**Model output:**
```
Non-Entailed Premises:
And so it goes | inferred | And so it goes

Entailed Premises:

Throughline:
And so it go
```

---

## Example 11 — score: 0.650

**Input quote:** Writing is a socially acceptable form of schizophrenia.

**Gold output:**
```
Non-Entailed Premises:
writing | inferred | writing

Entailed Premises:
writing | inferred | writing

Throughline:
Writing is a socially acceptable form of schizophrenia.
```

**Model output:**
```
Non-Entailed Premises:
Writing | inferred | schizophrenia

Entailed Premises:

Throughline:
Writing is a socially acceptable forms of schizophrenia.
```

---

## Example 12 — score: 0.650

**Input quote:** Do you want to know who you are? Don't ask. Act! Action will delineate and define you.

**Gold output:**
```
Non-Entailed Premises:
Act | inferred | Action

Entailed Premises:
Act | inferred | Action

Throughline:
Act will define you.
```

**Model output:**
```
Non-Entailed Premises:
Act | inferred | you

Entailed Premises:

Throughline:
Act! Action will delineate you.
```

---

## Example 13 — score: 0.550

**Input quote:** Believe nothing you hear, and only one half that you see.

**Gold output:**
```
Non-Entailed Premises:
"Believe nothing you hear" | inferred|confidence=0.0 | "only one half that you see"

Entailed Premises:
"Believe nothing you hear" | inferred|confidence=0.0 | "only one half that you see"

Throughline:
The quote asserts that one should not believe in anything heard but only in what is seen.
```

**Model output:**
```
Non-Entailed Premises:
belief | inferred | nothing

Entailed Premises:

throughline:
belief | inferred| nothing

Non-Entailed Premisses:
belief | inferred<| nothing

Entailed Premise:
belief | inferred <| nothing
```

---

## Example 14 — score: 0.650

**Input quote:** It hurts to let go. Sometimes it seems the harder you try to hold on to something or someone the more it wants to get away. You feel like some kind of criminal for having felt, for having wanted. For having wanted to be wanted. It confuses you, because you think that your feelings were wrong and it makes you feel so small because it's so hard to keep it inside when you let it out and it doesn't coma back. You're left so alone that you can't explain. Damn, there's nothing like that, is there? I've been there and you have too. You're nodding your head.

**Gold output:**
```
Non-Entailed Premises:
<it> | <inferred> (confidence=0.0) | <it>

Entailed Premises:
<it> | <inferred> (confidence=0.0) | <it>

Throughline:
It is implied that the speaker feels guilty for wanting to be wanted and that letting go causes them to feel small and alone.
```

**Model output:**
```
Non-Entailed Premises:
letting-go | inferred | it

Entailed Premises:

letting-go | inferred| it | it

Throughline:
The quote suggests that letting go is painful and confusing, often leading to a feeling of being small or alone.
```

---

## Example 15 — score: 0.760

**Input quote:** You know the world is going crazy when the best rapper is a white guy, the best golfer is a black guy, the tallest guy in the NBA is Chinese, the Swiss hold the America's Cup, France is accusing the U.S. of arrogance, Germany doesn't want to go to war, and the three most powerful men in America are named "Bush", "Dick", and "Colin." Need I say more?

**Gold output:**
```
Non-Entailed Premises:
"world" | inferred|confidence=0.0 | "crazy"

Entailed Premises:
"best rapper" | inferred|confidence=0.0 | "white guy"

Throughline:
The quote uses a series of specific, factual examples to illustrate a general statement about the absurdity of certain stereotypes.
```

**Model output:**
```
Non-Entailed Premises:
world | is | crazy

Entailed Premises:

world | is | crazy | inferred | confidence=0.9

Throughline:
The quote asserts that the world is currently in a state of extreme chaos or madness.
```

---

## Example 16 — score: 0.800

**Input quote:** But until a person can say deeply and honestly, "I am what I am today because of the choices I made yesterday," that person cannot say, "I choose otherwise.

**Gold output:**
```
Non-Entailed Premises:
Choices | inferred | made yesterday

Entailed Premises:
Choices | inferred | made yesterday

Throughline:
Choices made yesterday are the basis for the person's identity.
```

**Model output:**
```
Non-Entailed Premises:
"I am what I am today" | inferred|confidence=0.0 | "I choose otherwise"

Entailed Premises:

Throughline:
Until a person can say deeply, honestly, "I am whatever I am today because of choices I made yesterday", that person cannot say, 'I choose otherwise'.
```

---

## Example 17 — score: 0.650

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
people | inferred | coming back to where you started

Entailed Premises:

Throughline:
Coming back to where you started does not mean you never left.
```

---

## Example 18 — score: 0.650

**Input quote:** Believe in yourself. You are braver than you think, more talented than you know, and capable of more than you imagine.

**Gold output:**
```
Non-Entailed Premises:
believe | in | yourself

Entailed Premises:
believe | in | yourself | you | are | braver | than | you | think

Throughline:
Believe in yourself to be braver than you think, more talented than you know, and capable of more than you imagine.
```

**Model output:**
```
Non-Entailed Premises:
believe | in | yourself

Entailed Premises:

> believe in yourself | in | yourself

Throughline:
Believe in yourself.
```

---

## Example 19 — score: 0.630

**Input quote:** This is a good sign, having a broken heart. It means we have tried for something.

**Gold output:**
```
Non-Entailed Premises:
broken heart | observed | having a broken heart

Entailed Premises:
broken heart | observed | having a broken heart

Throughline:
Having a broken heart is a good sign because it indicates that the person has tried for something.
```

**Model output:**
```
Non-Entailed Premises:
broken heart | inferred | having a broken heart

Entailed Premises:

broken heart | inferred | trying for something

Throughline:
Having a broken heart is a sign that we have tried for something.
```

---

## Example 20 — score: 0.650

**Input quote:** I might be in love with you." He smiles a little. "I'm waiting until I'm sure to tell you, though.

**Gold output:**
```
Non-Entailed Premises:
He smiles | inferred | He smiles

Entailed Premises:
He smiles | inferred | He smiles

Throughline:
He smiles a little, and he is waiting until he is sure to tell you.
```

**Model output:**
```
Non-Entailed Premises:
He | smiles a little | observed

Entailed Premises:

He | smiles a little| observed | He

Throughline:
He smiles a little.
```

---
