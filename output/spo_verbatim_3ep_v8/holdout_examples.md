# SPO Holdout Inference Examples

**Adapter:** output/spo_verbatim_3ep_v8/adapter
**Examples:** 20

---

## Example 1 — score: 0.800

**Input quote:** You can't stay in your corner of the Forest waiting for others to come to you. You have to go to them sometimes.

**Gold output:**
```
Non-Entailed Premises:
Forest | (observed, confidence=1.0) | waiting for others to come to you

Entailed Premises:
Forest | (inferred, confidence=0.7) | go to them sometimes

Throughline:
You must go to them sometimes to stay in the Forest.
```

**Model output:**
```
Non-Entailed Premises:
Forest | (observed, confidence=X) | waiting for others to come

Entailed Premises:

```
Forest | (inferred, confidence=X) | going to them sometimes
```

Throughline:
You must go to them sometimes.
```

---

## Example 2 — score: 0.950

**Input quote:** If we listened to our intellect we'd never have a love affair. We'd never have a friendship. We'd never go in business because we'd be cynical: "It's gonna go wrong." Or "She's going to hurt me." Or,"I've had a couple of bad love affairs, so therefore . . ." Well, that's nonsense. You're going to miss life. You've got to jump off the cliff all the time and build your wings on the way down.

**Gold output:**
```
Non-Entailed Premises:
cynical | (inferred, confidence=0.7) | business

Entailed Premises:
cynical | (observed, confidence=1.0) | never have a love affair

Throughline:
The quote argues that cynicism is the root cause of a lack of love, friendship, and business success.
```

**Model output:**
```
Non-Entailed Premises:
 intellect | (inferred, confidence=0.7) | never have a love affair

Entailed Premises:  
 never have a love affair | (inferred,confidence=0.7)| never have a friendship

Throughline:
We'd never have a love affair, friendship, or business because we'd be cynical.
```

---

## Example 3 — score: 0.950

**Input quote:** She had always wanted words, she loved them; grew up on them. Words gave her clarity, brought reason, shape.

**Gold output:**
```
Non-Entailed Premises:
Words | (inferred, confidence=0.7) | clarity

Entailed Premises:
Words | (observed, confidence=1.0) | reason

Throughline:
Words gave her clarity, brought reason, shape.
```

**Model output:**
```
Non-Entailed Premises:
Words | (inferred, confidence=0.7) | clarity

Entailed Premises:

```
Words | (in inferred, confidence=0.9) | clarity
```

Throughline:
Words gave her clarity, broughtreason, shape.
```

---

## Example 4 — score: 0.850

**Input quote:** Keep in mind that people change, but the past doesn't.

**Gold output:**
```
Non-Entailed Premises:
people | (inferred, confidence=0.7) | change

Entailed Premises:
past | (inferred, confidence=0.7) | doesn't

Throughline:
The past is immutable, regardless of how people change.
```

**Model output:**
```
Non-Entailed Premises:
people | change (inferred, confidence=0.7) | past

Entailed Premises:

<subject> | (inferred, confidence=X.X) | <object>
past | (inferred,confidence=X.X) | people

Throughline:
The past is immutable, while people are mutable.
```

---

## Example 5 — score: 0.800

**Input quote:** Above all, be the heroine of your life, not the victim.

**Gold output:**
```
Non-Entailed Premises:
heroine | (observed, confidence=1.0) | above all

Entailed Premises:
heroine | (observed, confidence=1.0) | be the heroine of your life

Throughline:
be the heroine of your life
```

**Model output:**
```
Non-Entailed Premises:
heroine | (observed, confidence=X) | life

Entailed Premises:

*   be the heroine of your Life | (inferred, confidence=X) | not the victim

Throughline:
be the heroine of your life not the victim
```

---

## Example 6 — score: 0.950

**Input quote:** However many holy words you read, however many you speak, what good will they do you if you do not act on upon them?

**Gold output:**
```
Non-Entailed Premises:
read | (inferred, confidence=0.7) | what good will they do you if you do not act on upon them?

Entailed Premises:
act on upon | (inferred, confidence=0.7) | what good will they do you if you do not act on upon them?

Throughline:
The quote asserts that the effectiveness of holy words depends entirely on the speaker's subsequent action, specifically the act of acting upon them.
```

**Model output:**
```
Non-Entailed Premises:
read | (inferred, confidence=0.7) | what good will they do

Entailed Premises:

```
read | (in inferred, confidence=0.9) | what good will them do
```

Throughline:
what good will they do you *if* you do not act on them
```

---

## Example 7 — score: 0.950

**Input quote:** So it's not gonna be easy. It's going to be really hard; we're gonna have to work at this everyday, but I want to do that because I want you. I want all of you, forever, everyday. You and me... everyday.

**Gold output:**
```
Non-Entailed Premises:
hard | (inferred, confidence=0.7) | everyday

Entailed Premises:
everyday | (inferred, confidence=0.7) | work

Throughline:
The speaker asserts that the task is difficult and requires daily effort, yet they explicitly state their motivation is to do it forever with everyone.
```

**Model output:**
```
Non-Entailed Premises:
hard | (inferred, confidence=0.7) | we're gonna have to

Entailed Premises:

```
hard | (in inferred, confidence=0.9) | we're gonna do it everyday
```

Throughline:
We are going to have to work at this every day.
```

---

## Example 8 — score: 0.950

**Input quote:** A human being is a part of the whole called by us universe, a part limited in time and space. He experiences himself, his thoughts and feeling as something separated from the rest, a kind of optical delusion of his consciousness. This delusion is a kind of prison for us, restricting us to our personal desires and to affection for a few persons nearest to us. Our task must be to free ourselves from this prison by widening our circle of compassion to embrace all living creatures and the whole of nature in its beauty.

**Gold output:**
```
Non-Entailed Premises:
human being | (observed, confidence=1.0) | universe

Entailed Premises:
human being | (inferred, confidence=0.7) | universe

Throughline:
The quote argues that a human being is a part of the whole universe, which creates a delusion of separation that acts as a prison restricting the individual to personal desires and affection for specific others, necessitating a task to free oneself by expanding compassion to embrace all living creatures and nature.
```

**Model output:**
```
Non-Entailed Premises:
human being | (inferred, confidence=0.7) | universe

Entailed Premises:

human being | (in inferred, confidence=0.9) | universe

Throughline:
The quote argues that a human being is a part limited in time, space, and consciousness, which acts as a prison restricting them to personal desires and affection, and that the solution to this prison is to widen the circle of compassion to embrace the whole of nature.
```

---

## Example 9 — score: 0.913

**Input quote:** How many slams in an old screen door? Depends how loud you shut it. How many slices in a bread? Depends how thin you cut it. How much good inside a day? Depends how good you live 'em. How much love inside a friend? Depends how much you give 'em.

**Gold output:**
```
Non-Entailed Premises:
How many slams in an old screen door? | (inferred, confidence=0.7) | confidence=0.0 | Depends how loud you shut it.

Entailed Premises:
How many slices in a bread? | (inferred, confidence=0.7) | confidence=0.0 | Depends how thin you cut it.

Throughline:
The structure of the quote uses a conditional question format ("How many X in Y?") where the answer depends on a specific variable (loudness or thickness) rather than a fixed number.
```

**Model output:**
```
Non-Entailed Premises:
How many slams in a screen door | (inferred, confidence=0.7) | confidence=0.0 | "How many slams"

Entailed Premises:

How many slams in the old screen door | (in inferred, confidence=0.9) | confidence=0.x | "How many slums"

Throughline:
The quote uses rhetorical questions to illustrate the importance of action and effort in achieving a goal.
```

---

## Example 10 — score: 0.950

**Input quote:** Love does not begin and end the way we seem to think it does. Love is a battle, love is a war; love is a growing up.

**Gold output:**
```
Non-Entailed Premises:
Love | (inferred, confidence=0.7) | is a battle

Entailed Premises:
Love | (inferred, confidence=0.7) | is a growing up

Throughline:
Love is a growing up.
```

**Model output:**
```
Non-Entailed Premises:
Love | (inferred, confidence=0.7) | a battle

Entailed Premises:

Love | (inferred|confidence=0.7)|a war

Throughline:
Love is a battle and a war.
```

---

## Example 11 — score: 0.800

**Input quote:** Fate is like a strange, unpopular restaurant filled with odd little waiters who bring you things you never asked for and don't always like.

**Gold output:**
```
Non-Entailed Premises:
Fate | is (inferred, confidence=0.7) | strange | unpopular | restaurant | filled | with | odd | little | waiters | who | bring | you | things | you | never | asked | for | and | don't | always | like |

Entailed Premises:
Fate | is (inferred, confidence=0.7) | like | a | strange | unpopular | restaurant | filled | with | odd | little | waiters | who | bring | you | things | you | never | asked | for | and | don't | always | like |

Throughline:
Fate is like a strange, unpopular restaurant filled with odd little waiters who bring you things you never asked for and don't always like.
```

**Model output:**
```
Non-Entailed Premises:
Fate | (observed, confidence=X) | strange, unpopular restaurant filled

Entailed Premises:

Fate | (inferred, confidence=X) | like a strange, unpopular

Throughline:
Fate is like a weird, unliked restaurant with odd waiters who bring you items you never asked for.
```

---

## Example 12 — score: 0.850

**Input quote:** When the power of love overcomes the love of power, the world will know peace.

**Gold output:**
```
Non-Entailed Premises:
power | overcomes (inferred, confidence=0.7) | love | power

Entailed Premises:
power | overcomes (inferred, confidence=0.7) | love | power | world | peace

Throughline:
When the power of love overcomes the love of power, the world will know peace.
```

**Model output:**
```
Non-Entailed Premises:
power of love | overcomes (inferred, confidence=0.7) | love of power

Entailed Premises:

*   (observed, confidence=X) | inferred, confidence=X.0 | world will know peace

Throughline:
The quote asserts that when the power of love surpasses the love of power, peace will be known by the world.
```

---

## Example 13 — score: 0.950

**Input quote:** You can't stop the futureYou can't rewind the pastThe only way to learn the secret...is to press play.

**Gold output:**
```
Non-Entailed Premises:
play | (inferred, confidence=0.7) | press

Entailed Premises:
press | (inferred, confidence=0.7) | play

Throughline:
The quote asserts that the only way to learn the secret is to press play.
```

**Model output:**
```
Non-Entailed Premises:
You can't stop the past | (inferred, confidence=0.7) | confidence=0.0 | "You can't rewind the future"

Entailed Premises:

The only way to learn a secret | (inferred,confidence=0.7)|confidence=0.0 | is to press play.

Throughline:
The only way to learn secret is to press play.
```

---

## Example 14 — score: 0.850

**Input quote:** I don't go out with strangers," I said."Good thing I do. I'll pick you up at five.

**Gold output:**
```
Non-Entailed Premises:
I don't go out with strangers | (inferred, confidence=0.7) | I said

Entailed Premises:
I don't go out with strangers | (observed, confidence=1.0) | I said

Throughline:
I said I don't go out with strangers, which implies I will pick you up at five.
```

**Model output:**
```
Non-Entailed Premises:
I don't go out | with strangers (inferred, confidence=0.7) | confidence=0.0 | "I don't go"

Entailed Premises:

I don't go out (inferred, confidence=X) | with strangers (in inferred, confidence=X) | confidence=0.X | "I don't"

Throughline:
I don't | go out with strangers (inferred | confidence=X) | "I don't | go"
```

---

## Example 15 — score: 0.950

**Input quote:** Tis the privilege of friendship to talk nonsense, and to have her nonsense respected.

**Gold output:**
```
Non-Entailed Premises:
friendship | (observed, confidence=1.0) | talk nonsense

Entailed Premises:
friendship | (inferred, confidence=0.7) | respected

Throughline:
Friendship is the privilege of having her nonsense respected.
```

**Model output:**
```
Non-Entailed Premises:
friendship | (inferred, confidence=0.7) | talk nonsense

Entailed Premises:

friendship | (in inferred, confidence=0.9) | respected

Throughline:
Tis the privilege of friendshp to talk nonsense, and t have her nonsense respected.
```

---

## Example 16 — score: 0.835

**Input quote:** Headline?" he asked."'Swing Set Needs Home,'" I said."'Desperately Lonely Swing Set Needs Loving Home,'" he said."'Lonely, Vaguely Pedophilic Swing Set Seeks the Butts of Children,'" I said.

**Gold output:**
```
Non-Entailed Premises:
Swing Set Needs Home | (inferred, confidence=0.7) | Desperately Lonely Swing Set Needs Loving Home

Entailed Premises:
Desperately Lonely Swing Set Needs Loving Home | (inferred, confidence=0.7) | Lonely, Vaguely Pedophilic Swing Set Seeks the Butts of Children

Throughline:
He asked a question, then answered it, and then asked a question again, each time with a different emotional tone.
```

**Model output:**
```
Non-Entailed Premises:
Headline | (inferred, confidence=0.7) | "Headline?" he asks

Entailed Premises:

```
Headline | (observed, confidence=X) | "Headline" he asks
```

Throughline:
The speaker asks a question about the headline, then responds by describing a "Desperately Lonely Swing set needs loving home," followed by a description of a "Lonely, VaguELY Pedophilic Swing set seeks the Butts of Children."
```

---

## Example 17 — score: 0.950

**Input quote:** The truth does not change according to our ability to stomach it.

**Gold output:**
```
Non-Entailed Premises:
stomach | stomach (inferred, confidence=0.7) | true

Entailed Premises:
stomach | true (inferred, confidence=0.7) | truth

Throughline:
The truth remains constant regardless of whether the subject can stomach the object.
```

**Model output:**
```
Non-Entailed Premises:
stomach | (inferred, confidence=0.7) | ability

Entailed Premises:

*   | (inferred,confidence=0.7)| confidence=0.99 | stomach

Throughline:
The truth does not change regardless of the subject's ability to stomach it.
```

---

## Example 18 — score: 0.875

**Input quote:** It means 'Shadowhunters: Looking Better in Black Than the Widows of our Enemies Since 1234'.

**Gold output:**
```
Non-Entailed Premises:
Shadowhunters | (observed, confidence=1.0) | Looking Better in Black Than the Widows of our Enemies Since 1234

Entailed Premises:
Shadowhunters | (inferred, confidence=0.7) | Looking Better in Black Than the Widows of our Enemies Since 1234

Throughline:
Shadowhunters is a reference to the movie "Shadowhunters: Looking Better in Black Than the Widows of our Enemies Since 1234".
```

**Model output:**
```
Non-Entailed Premises:
Shadowhunters | (inferred, confidence=0.7) | Looking Better in Black Than

Entailed Premises:

Shadowhunters | (observed, confidence=X) | Looking Better inBlack Than the Widows of their Enemies Since 1240

Throughline:
Shadowhunters: Looking better in black than the widows of their enemies since 1234
```

---

## Example 19 — score: 0.950

**Input quote:** Never be bullied into silence. Never allow yourself to be made a victim. Accept no one's definition of your life, but define yourself.

**Gold output:**
```
Non-Entailed Premises:
Never allow yourself to be made a victim | (inferred, confidence=0.7) | Never allow yourself to be made a victim

Entailed Premises:
Accept no one's definition of your life | (inferred, confidence=0.7) | Accept no one's definition of your life

Throughline:
Never allow yourself to be made a victim.
```

**Model output:**
```
Non-Entailed Premises:
Never allow yourself to be made into a victim | (inferred, confidence=0.7) | confidence=0.0 | "Never allow yourself to be a victim"

Entailed Premises:

*   Never allow yourself to be avictim | (inferred,confidence=0.7)|confidence=0.0 | accept no one's definition ofyour life

Throughline:
*   Accept no one's definitionofyour life, but define yourself.
```

---

## Example 20 — score: 0.875

**Input quote:** Absence diminishes small loves and increases great ones, as the wind blows out the candle and fans the bonfire.

**Gold output:**
```
Non-Entailed Premises:
candle | blows out (inferred, confidence=0.7) | object

Entailed Premises:
candle | fans (inferred, confidence=0.7) | object

Throughline:
The wind blowing out the candle and fans the bonfire diminishes small loves and increases great ones.
```

**Model output:**
```
Non-Entailed Premises:
candle | blows (inferred, confidence=0.7) | out

Entailed Premises:

```
candle | blows out (inferred, confidence=X) | out
```

Throughline:
The quote states that the wind blows out the small loves and increases the great ones.
```

---
