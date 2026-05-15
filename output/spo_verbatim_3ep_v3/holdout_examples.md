# SPO Holdout Inference Examples

**Adapter:** output/spo_verbatim_3ep_v3/adapter
**Examples:** 20

---

## Example 1 — score: 0.921

**Input quote:** What you're supposed to do when you don't like a thing is change it. If you can't change it, change the way you think about it. Don't complain.

**Gold output:**
```
Non-Entailed Premises:
complain | is (inferred, confidence=0.5) | not a recommended action

Entailed Premises:
change perspective | is (observed, confidence=1.0) | when you can't change it
change action | is (observed, confidence=1.0) | when you don't like a thing
don't complain | is (inferred, confidence=0.5) | as a general directive

Throughline:
When faced with dissatisfaction, one should either alter the external circumstance or adjust their internal perception, rather than engaging in complaint.
```

**Model output:**
```
Non-Entailed Premises:
change_it | is (inferred, confidence=0.5) | implies a direct action
change_the_way_you_think | is (inferred) | implies a cognitive adjustment
complain | is (inferred/observed) | implies a negative emotional response

Entailed Premises:

change_it | is (observed, confidence = 1.0) | is the recommended action when you don't like something
change_the_way_you_thing | is (inferred), | change the way you think
change_the_way_you | is (observed), | change the way
change_it | is the recommended action
change_the_way | is (inferred from change_the_way_you), | change the way

Throughline:
If you cannot change the object, you should change your perspective on it.
```

---

## Example 2 — score: 0.670

**Input quote:** It's the friends you can call up at 4 a.m. that matter.

**Gold output:**
```
Non-Entailed Premises:
calling up | is an (observed, confidence=1.0) | action
matters | implies (observed, confidence=1.0) | importance
4 a.m. | is a (observed, confidence=1.0) | time of night
friends | are (observed, confidence=1.0) | people

Entailed Premises:
availability at 4 a.m. | indicates (observed, confidence=1.0) | true friendship
friends | are defined by (observed, confidence=1.0) | availability at 4 a.m.
friends you can call up at 4 a.m. | are (observed, confidence=1.0) | true friends

Throughline:
True friendship is characterized by the willingness and ability to be available during times of crisis or late-night need, making such friends the most valuable.
```

**Model output:**
```
Non-Entailed Premises:
time | is (observed, confidence=X) | 4 a.m.
time | is (inferred, confidence=X) | friends matter
time | is (observable, confidence=X) | can be called up

Entailed Premises:

time | is (independent, confidence=X) | not friends
time | is (dependent, confidence=X)| friends
time | is not friends
time | can be called up at 4 a m
time | is friends
time | is friends you can call up

Throughline:
Friends are the most important thing in life.
```

---

## Example 3 — score: 0.660

**Input quote:** It does not matter how slowly you go as long as you do not stop.

**Gold output:**
```
Non-Entailed Premises:
metaphor | is (inferred, confidence=0.5) | 'going' represents life's journey or effort
advice | is (inferred, confidence=0.5) | one should prioritize persistence over speed

Entailed Premises:
comparison | is (inferred, confidence=0.5) | continuous slow movement is superior to halted movement
action | is (inferred, confidence=0.5) | movement or progress is possible at any speed
condition | is (inferred, confidence=0.5) | stopping prevents any form of progress
goal | is (inferred, confidence=0.5) | the primary objective is to avoid stopping

Throughline:
Persistence in moving forward, regardless of speed, is superior to stopping.
```

**Model output:**
```
Non-Entailed Premises:
speed | is (observed, confidence=X) | irrelevant to stopping
time | is (observed/derived, confidence=X) | a measure of progress
speed | is (inferred, confidence=X) | does not equal stopping

Entailed Premises:

speed | is (obsessed, confidence=X) | not stopping
speed | is (derived, confidence=X) > 0
speed | is (observable, confidence=X) | exists
time | is (inferred/observed, confidence=X>0) | exists
time > 0 | is (inferred) | true
time | is (observable, confident=X>0) | true
time > 0 is (inferred) > true

Throughline:
One must not stop to progress.
```

---

## Example 4 — score: 0.669

**Input quote:** Tell the truth, or someone will tell it for you.

**Gold output:**
```
Non-Entailed Premises:
Speaker | is (observed, confidence=1.0) | has personal knowledge of the truth
Someone | is (observed, confidence=1.0) | knows the specific truth in question
Truth | is (observed, confidence=1.0) | always easy to tell

Entailed Premises:
Power dynamic | is (inferred, confidence=0.5) | the speaker holds moral authority or social leverage over the listener
Speaker | is (observed, confidence=1.0) | urges telling the truth
Consequence | is (inferred, confidence=0.5) | if one does not tell the truth, someone else will
Alternative | is (observed, confidence=1.0) | someone else reveals the truth
Motivation | is (inferred, confidence=0.5) | to avoid the negative social or personal impact of being exposed by another
Truth | is (observed, confidence=1.0) | exists as an objective state of affairs

Throughline:
Honesty is a strategic necessity to maintain agency and avoid the humiliation or consequences of being exposed by others.
```

**Model output:**
```
Non-Entailed Premises:
someone | is (observed, confidence=X) | exists
someone | is (inferred, confidence=X) | has the power to tell the truth
someone | is (obscured, confidence=X) | is not telling the truth
someone | has the power to lie
someone | is (observable, confidence=X) | can be observed
someone | is (possible, confidence=X) | could lie

Entailed Premises:

someone | is (inference, confidence=X) | tells the truth
someone | could (inference, confidence=0.5) | lie
someone | is possible (inference, confidence=x) | could lie
someone | is observable (inference, confidence=y) | can be observed

Throughline:
The speaker warns that telling the truth is a risky act because it invites potential deception.
```

---

## Example 5 — score: 0.579

**Input quote:** God created war so that Americans would learn geography.

**Gold output:**
```
Non-Entailed Premises:
Geography | is (observed, confidence=1.0) | difficult to learn (assumed)
God | uses (observed, confidence=1.0) | war as a teaching tool (inferred method)
God | is (observed, confidence=1.0) | benevolent (assumed)
Americans | are (observed, confidence=1.0) | ignorant of geography (assumed)
War | is (observed, confidence=1.0) | evil (assumed)

Entailed Premises:
God's purpose | is (observed, confidence=1.0) | Americans learning geography through war
War | causes (observed, confidence=1.0) | Americans to learn geography
Americans | learn (observed, confidence=1.0) | geography
God | created (observed, confidence=1.0) | war
God | created (observed, confidence=1.0) | geography (implied by purpose)

Throughline:
God intentionally used war as a pedagogical tool to teach Americans geography.
```

**Model output:**
```
Non-Entailed Premises:
God | is (observed, confidence=X) | created war
God | is (inferred, confidence=X) | has a purpose
God | is (observable, confidence=X) | exists
God | is (unobservable, confidence=X)

Entailed Premises: Americans | is (inferred/observed, confidence=X or Y) | learn geography
God | is (created, confidence=X) | war
God | is created (observed, confidence=Y) | war
God's purpose | is (inferred) | to teach Americans geography
Americans | is (observed/derived, confidence=X) | learn geography

Throughline:
God created war to teach Americans geography.
```

---

## Example 6 — score: 0.744

**Input quote:** Nothing in life is to be feared, it is only to be understood. Now is the time to understand more, so that we may fear less.

**Gold output:**
```
Non-Entailed Premises:
Life | is (observed, confidence=1.0) | contains things to be understood
Nothing | is (observed, confidence=1.0) | to be feared in life
Now | is (observed, confidence=1.0) | the time to understand more

Entailed Premises:
Understanding | is (observed, confidence=1.0) | reduces fear
Understanding | is (inferred, confidence=0.5) | the method to reduce fear
Fear | is (observed, confidence=1.0) | associated with lack of understanding
We | is (observed, confidence=1.0) | should understand more
We | is (inferred, confidence=0.5) | should fear less

Throughline:
Understanding is the primary means to overcome fear, therefore one should seek to understand rather than fear.
```

**Model output:**
```
Non-Entailed Premises:
Nothing in life | is (observed, confidence=X) | to be feared
Nothing in life | has (observed, confidence=E) | fear
Nothing in life | can (observed, confidence=Y) | be understood
Nothing in life | should (observed, confidence=W) | be understood

Entailed Premises:

1. | is (observed/derived, confidence=0.5) | Understanding is the primary means to reduce fear
2. | is (obsined, confidence=0.9) | Understanding is the only way to reduce fear
3. | is (obsessed, confidence=0.7) | Fear is a state that can be reduced through understanding
4. | is (obsessive, confidence=0.8) | Understanding is the necessary condition for reducing fear
5. | is (obsession, confidence=0.6) | Understanding is the goal of life
6. | is (obsessions, confidence=0.4) | Understanding is the method to achieve fear reduction
7. | is (obsessional, confidence=0.3) | Understanding is the means to achieve fear reduction
8. | is (obsesses, confidence=0.2) | Understanding is the end goal of life
9. | is (observes, confidence=0.1) | Understanding is the process of reducing fear
10. | is (observers, confidence=0.0) | Understanding is the action to reduce fear

Throughline:
Understanding is the necessary condition for achieving a state of reduced fear.
```

---

## Example 7 — score: 0.685

**Input quote:** Come sleep with me: We won't make Love, Love will make us.

**Gold output:**
```
Non-Entailed Premises:
We | is (inferred, confidence=0.5) | won't make Love
Come | is (observed, confidence=1.0) | sleep with me

Entailed Premises:
I | is (inferred, confidence=0.5) | am proposing a sexual encounter
We | is (inferred, confidence=0.5) | will not actively perform the act of making love
Love | is (inferred, confidence=0.5) | an autonomous agent or force
We | is (inferred, confidence=0.5) | will be acted upon by Love

Throughline:
Love is an independent force that will bring us together, rather than us actively creating the connection through our own actions.
```

**Model output:**
```
Non-Entailed Premises:
Sleep | is (observed, confidence=X) | a state of rest
Love | is (observed/Inferred, confidence=X) | an emotional connection
We | is (observed/inferred, confidence=X) -> We are the subjects of the action

Entailed Premises:

We | is (obspected, confidence=X) -> The subjects of the action
Love | is (inferred, confidence=X) → The cause of the action
We | is (inferred/observed, confidence=X/observed) -> We will not make Love
Love | is (observable, confidence=X) → Love will make us

Throughline:
Love is the cause of our relationship.
```

---

## Example 8 — score: 0.736

**Input quote:** Vanity and pride are different things, though the words are often used synonymously. A person may be proud without being vain. Pride relates more to our opinion of ourselves, vanity to what we would have others think of us.

**Gold output:**
```
Non-Entailed Premises:
Pride | is (observed, confidence=1.0) | often used synonymously with vanity
Words | is (observed, confidence=1.0) | are often used synonymously
Vanity | is (observed, confidence=1.0) | often used synonymously with pride

Entailed Premises:
Pride | is (observed, confidence=1.0) | different from vanity
Vanity | is (observed, confidence=1.0) | different from pride
Vanity | is (inferred, confidence=0.5) | related to what we would have others think of us
Pride | is (inferred, confidence=0.5) | related to our opinion of ourselves
A person | is (observed, confidence=1.0) | may be proud without being vain

Throughline:
Vanity and pride are distinct concepts: pride concerns self-opinion, while vanity concerns the impression made on others.
```

**Model output:**
```
Non-Entailed Premises:
Vanity | is (observed, confidence=X) | distinct from Pride
Pride | is (observed/Inferred, confidence=X) | a state of self-regard
Pride | is used (observed, confidence=Y) | synonymously with Vanity
Vanity | is used (obsessed, confidence=Y) | as a synonym for Pride

Entailed Premises:

Pride | is (inferred, confidence=X) |- Pride relates more to our own opinion of ourselves
Vanity | is defined as (observed, confidence=M) |- what we would have others thought of us
Pride | is not (observed, confidence=N) |- Vanity
Vanity | relates to (observed, confidence=P) |- what we would be proud of
Pride | is distinct from (observed, confidence=S) |- Vanity
Vain | is (observed/inferred, confidence=T) |- Vanity
Pride | is related to (observed/inferred/obsessed, confidence=U) |- our opinion of ourselves
Vain | is related to (inferred, confidence=V) |- our opinion of others
Pride | is a state of (observed/inferred) |- our opinion of our own self
Vanity | is a state of being (observed/inferred/proud) |- what we would think of others

Throughline:
Pride is a state of self-regarding opinion, whereas Vanity is a state of self-observing opinion regarding others.
```

---

## Example 9 — score: 0.571

**Input quote:** Man often becomes what he believes himself to be. If I keep on saying to myself that I cannot do a certain thing, it is possible that I may end by really becoming incapable of doing it. On the contrary, if I have the belief that I can do it, I shall surely acquire the capacity to do it even if I may not have it at the beginning.

**Gold output:**
```
Non-Entailed Premises:
I | is (observed, confidence=1.0) | keep on saying to myself that I cannot do a certain thing
I | is (observed, confidence=1.0) | have the belief that I can do it
I | is (observed, confidence=1.0) | may not have the capacity at the beginning
I | is (inferred, confidence=0.5) | shall surely acquire the capacity to do it
I | is (inferred, confidence=0.5) | may end by really becoming incapable of doing it
Man | is (observed, confidence=1.0) | becomes what he believes himself to be

Entailed Premises:
negative self-belief | is (inferred, confidence=0.5) | leads to perceived incapability
positive self-belief | is (inferred, confidence=0.5) | leads to acquired capacity
self-belief | is (observed, confidence=1.0) | determines outcome of action
belief | is (observed, confidence=1.0) | influences self-perception

Throughline:
Self-belief is a causal factor in determining one's actual capabilities and outcomes.
```

**Model output:**
```
Non-Entailed Premises:
I | is (observed, confidence=X) | keep on saying to myself
I | is (inferred, confidence=X) | have the belief that I cannot do a thing
I | is (possible, confidence=X) | end by really becoming incapable
I | is (possessed, confidence=X) | acquire the capacity to do something

Entailed Premises:

I | is (independent of, confidence=X) | having the belief that I can't do something
I | is (dependent on, confidence=X) | believing that I can do something
I | has (observed, confidence=C) | the belief that I can
I | is (causal, confidence=X) | acquiring the capacity to do something
I | can (observed, confidence=Y) | acquire the capacity
I | is (conditional, confidence=Y) | if I have the belief I can do it
I | is (conditioned on, confidence=Y) | having the belief I can do something
The belief that I can do something | is (independent, confidence=X) | the belief that the speaker can do something
The capacity to do something | is (observed/possible, confidence=X) |(independent of, confidence=Y) | the belief that one can do it
The belief that one can do something | is possible, confidence=X | the belief that one cannot do it
The belief one can do something | has (observed/possible/conditional, confidence=X) | capacity to do it
The belief the speaker can do something | is causal, confidence=X | the capacity to do it
I | has (independent of, confident=X) | the belief I can do the thing
I | is causal, confidence=Y | the belief I can't do the thing
I has (observed/conditional, confidence=X | independent of, confidence=Y | the capacity to
```

---

## Example 10 — score: 0.737

**Input quote:** Only people who are capable of loving strongly can also suffer great sorrow, but this same necessity of loving serves to counteract their grief and heals them.

**Gold output:**
```
Non-Entailed Premises:
people incapable of loving strongly | is (inferred, confidence=0.5) | cannot suffer great sorrow

Entailed Premises:
people capable of loving strongly | is (observed, confidence=1.0) | can suffer great sorrow
people capable of loving strongly | is (observed, confidence=1.0) | have a necessity of loving
necessity of loving | is (observed, confidence=1.0) | heals them
necessity of loving | is (observed, confidence=1.0) | counteracts grief

Throughline:
People who are capable of loving strongly possess an inherent mechanism (the necessity of loving) that mitigates and ultimately heals the great sorrow they are capable of experiencing.
```

**Model output:**
```
Non-Entailed Premises:
sorrow | is (observed, confidence=X) | a negative emotion
sorrow | is caused by (observed, confidence=Y) | grief
grief | is caused by (inferred, confidence=Z) | loving strongly
loving strongly | is a necessary condition for (observed, confidence=W) | suffering great sorrow

Entailed Premises:

loving strongly | is (observed/derived, confidence=1.5) | a necessary condition for (inferred, confidence=W) | loving strongly
suffering great sorrow | is (observed) | a consequence of (observed, confidence=A) | loving strongly
grief | is (observed/inferred, confidence=B) | a consequence of loving strongly
loving strong | is (observed/deduced, confidence=C) | a necessary condition to (observed, confidence=D) | counteract grief
loving strong | heals (observed, confidence=E) | grieving individuals

Throughline:
Loving strongly is a necessary condition for suffering great sorrow, and this same capacity to love strongly serves to counteract grief and heal the individual.
```

---

## Example 11 — score: 0.660

**Input quote:** Confront the dark parts of yourself, and work to banish them with illumination and forgiveness. Your willingness to wrestle with your demons will cause your angels to sing.

**Gold output:**
```
Non-Entailed Premises:
dark parts of yourself | is (observed, confidence=1.0) | exist within the self
forgiveness | is (inferred, confidence=0.5) | a method to banish dark parts
illumination | is (inferred, confidence=0.5) | a method to banish dark parts
angels | is (inferred, confidence=0.5) | sing as a reaction to wrestling with demons

Entailed Premises:
wrestling with demons | is (observed, confidence=1.0) | causes angels to sing
banishing dark parts with illumination and forgiveness | is (inferred, confidence=0.5) | involves wrestling with demons
willingness to wrestle with demons | is (observed, confidence=1.0) | causes angels to sing
confronting dark parts of yourself | is (inferred, confidence=0.5) | involves wrestling with demons

Throughline:
Wrestling with one's inner demons leads to a positive spiritual outcome (angels singing).
```

**Model output:**
```
Non-Entailed Premises:
confronting the dark parts of yourself | is (observed, confidence=X) | requires illumination and forgiveness
your demons | is (observed/derived, confidence=X) | are demons
your angels | is (observed/inferred, confidence=X) | sing when you wrestle with demons

Entailed Premises:

your willingness to wrestle with demons | is (inferred, confidence=X) -> causes your angels to sing
your willingness to wrestle | is (inferred/observed, confidence=X)-> causes your angels to sing

Throughline:
Wrestling with demons causes your angels to sing.
```

---

## Example 12 — score: 0.703

**Input quote:** Sometimes it takes a good fall to really know where you stand

**Gold output:**
```
Non-Entailed Premises:
position | is (inferred, confidence=0.5) | physical_location
fall | is (observed, confidence=1.0) | physical_tumbling

Entailed Premises:
self_reflection | is (inferred, confidence=0.5) | reveals_true_state
true_state | is (inferred, confidence=0.5) | clarity_on_position
negative_event | is (inferred, confidence=0.5) | causes_self_reflection
fall | is (observed, confidence=1.0) | negative_event

Throughline:
Adversity or failure provides the necessary perspective to understand one's true character or situation.
```

**Model output:**
```
Non-Entailed Premises:
good fall | is (observed, confidence=X) | a physical event
fall | is (observed/derived, confidence=X) | indicates a state of being

Entailed Premises:

good fall | is (inferred, confidence=0.5) | a state of being
fall | is (inferred/observed, confidence=0.9) | takes time to be known
fall | is (observable, confidence=1.5) | takes time to know where you stand
fall | is (observational, confidence=1.9) | takes a good fall to know where you stand

Throughline:
It takes time to understand one's current state of being.
```

---

## Example 13 — score: 0.690

**Input quote:** I love deadlines. I love the whooshing noise they make as they go by.

**Gold output:**
```
Non-Entailed Premises:
the speaker | is (observed, confidence=1.0) | a procrastinator
the speaker | ignores (observed, confidence=1.0) | deadlines until the last minute
deadlines | are (observed, confidence=1.0) | annoying to others
deadlines | cause (observed, confidence=1.0) | stress

Entailed Premises:
the speaker | loves (observed, confidence=1.0) | the whooshing noise made by deadlines
the speaker | loves (observed, confidence=1.0) | deadlines
deadlines | pass by (observed, confidence=1.0) | the speaker
deadlines | make (observed, confidence=1.0) | a whooshing noise

Throughline:
The speaker enjoys the impending arrival of deadlines, likely because the pressure they create is motivating or enjoyable to them.
```

**Model output:**
```
Non-Entailed Premises:
I | is (observed, confidence=X) | love deadlines
I | is (inferred, confidence=X) | the whooshing noise is pleasant
I | is (observable, confidence=X) | loves the whooshing noise
I | is (unobserved, confidence=X, inferred) | the whooshed noise is pleasant

Entailed Premises:

I | is (independent of, confidence=X) | I love deadlines
I | loves (observed, confidence=Y) | the whooshes
I | is (dependent of, confidence=X, independent of) | the whooshings
I | is (observant, confidence=X) | sees the whooshing noise

Throughline:
I enjoy the auditory experience of deadlines.
```

---

## Example 14 — score: 0.746

**Input quote:** If you don't read the newspaper, you're uninformed. If you read the newspaper, you're mis-informed.

**Gold output:**
```
Non-Entailed Premises:
The person | is (inferred, confidence=0.5) | mis-informed
The person | is (observed, confidence=1.0) | reads the newspaper

Entailed Premises:
The person | is (observed, confidence=1.0) | does not read the newspaper
The person | is (inferred, confidence=0.5) | uninformed

Throughline:
If you don't read the newspaper, you're uninformed.
```

**Model output:**
```
Non-Entailed Premises:
reading the newspaper | is (observed, confidence=X) | an action
reading the newspaper | causes (observed, confidence=Y) | mis-informed
not reading the newspaper | is (inferred, confidence=Z) | uninformed

Entailed Premises:

reading the newspaper | is a (observed, confidence=W) | action
reading the newspaper (action) | is (observed/derived, confidence=W) | causes (observed/derived) | mis-informed

Throughline:
Reading the newspaper is a necessary condition for being informed.
```

---

## Example 15 — score: 0.746

**Input quote:** When people talk, listen completely. Most people never listen.

**Gold output:**
```
Non-Entailed Premises:
you | is (inferred, confidence=0.5) | should listen completely when people talk

Entailed Premises:
people | is (observed, confidence=1.0) | talk
most people | is (inferred, confidence=0.5) | do not listen completely
most people | is (inferred, confidence=0.5) | never listen completely

Throughline:
Most people fail to listen completely.
```

**Model output:**
```
Non-Entailed Premises:
most people | is (inferred, confidence=0.5) | do not listen completely
people | is (observed, confidence=X) | talk

Entailed Premises:

most people | is (observed/derived, confidence=0.9) | never listen completely
people | can (observed/derived/implication, confidence=1.5) | talk completely
people | is/are (observed/derived/inference, confidence=1.2) | talk

Throughline:
Most people fail to listen completely when they talk.
```

---

## Example 16 — score: 0.711

**Input quote:** So you're always honest," I said."Aren't you?""No," I told him. "I'm not.""Well, that's good to know, I guess.""I'm not saying I'm a liar," I told him. He raised his eyebrows. "That's not how I meant it, anyways.""How'd you mean it, then?""I just...I don't always say what I feel.""Why not?""Because the truth sometimes hurts," I said."Yeah," he said. "So do lies, though.

**Gold output:**
```
Non-Entailed Premises:
I am not always honest | is (inferred, confidence=0.5) | I am not telling the whole truth
Lies sometimes hurt | is (inferred, confidence=0.5) | Deception has negative consequences
I am not always honest | is (observed, confidence=1.0) | I am not honest
I don't always say what I feel | is (inferred, confidence=0.5) | I hide my feelings
The truth sometimes hurts | is (inferred, confidence=0.5) | Honesty can be painful

Entailed Premises:
I am not always honest | is (inferred, confidence=0.5) | I do not always say what I feel
Lies sometimes hurt | is (inferred, confidence=0.5) | Lies can cause harm
I am not always honest | is (inferred, confidence=0.5) | I am not a liar
The truth sometimes hurts | is (inferred, confidence=0.5) | Truth can cause harm
I do not always say what I feel | is (inferred, confidence=0.5) | The truth sometimes hurts

Throughline:
Honesty is not absolute; withholding feelings or telling lies can also cause harm, making the distinction between truth and lies more complex than simple moral binaries.
```

**Model output:**
```
Non-Entailed Premises:
I | is (inferred, confidence=0.5) | I am not a liar
I | is (observed, confidence=X) | I am not always honest
I | is (explicit, confidence=X) | The truth sometimes hurts
I | is (implicit, confidence=X) | My honesty is conditional
I | is (observation, confidence=X) | He raised his eyebrows

Entailed Premises:

I | is (inference, confidence=0.9) | I am not saying I'm a liars
I | is (implication, confidence=0.7) | I don't always say what feels
I | is (implied, confidence=0.8) | I am not honest
I | is（observation, confidence=1.5） | I am not always truthful
I | is（inference, confidence=1.2） | The truth hurts
I | is（implication, confidence=1.1） | Lies are also true
I | is（explicit, confidence=1.3） | I am not honest

Throughline:
I am honest, but I am not always truthful because the truth sometimes hurts.
```

---

## Example 17 — score: 0.653

**Input quote:** A trap is only a trap if you don't know about it. If you know about it, it's a challenge.

**Gold output:**
```
Non-Entailed Premises:
trap | is (observed, confidence=1.0) | a device for catching
challenge | is (observed, confidence=1.0) | a difficult task

Entailed Premises:
lack_of_knowledge_of_a_trap | is (observed, confidence=1.0) | maintains its classification as a trap
knowledge_of_a_trap | is (observed, confidence=1.0) | changes its classification to challenge
trap | is (inferred, confidence=0.5) | a situation with negative potential
challenge | is (inferred, confidence=0.5) | a situation requiring overcoming effort

Throughline:
Perception (specifically awareness) determines whether a hazardous situation is perceived as an insurmountable trap or a manageable challenge.
```

**Model output:**
```
Non-Entailed Premises:
trap | is (observed, confidence=X) | a trap
trap | is (inferred, confidence=X) | requires knowledge to be effective

Entailed Premises:

trap | is (obsessed, confidence=X) | only a trap if you know about it
trap | is (observable, confidence=X) | is a trap
trap | requires (obsessed, confidence=1.5) | knowledge to be effective
challenge | is (inferred from, confidence=X) | trap
challenge | is (obsessed from, confidence=1.2) | trap

Throughline:
Knowledge transforms a trap into a challenge.
```

---

## Example 18 — score: 0.655

**Input quote:** I suffer from girlnextdooritis where the guy is friends with you and that's it.

**Gold output:**
```
Non-Entailed Premises:
The speaker | is (inferred, confidence=0.5) | disappointed or frustrated by this limitation
The speaker | is (inferred, confidence=0.5) | female (based on the term 'girlnextdoor')
The guy | is (observed, confidence=1.0) | friends with the speaker
The guy | is (inferred, confidence=0.5) | male (based on the term 'girlnextdoor')

Entailed Premises:
The speaker | is (observed, confidence=1.0) | suffers from 'girlnextdooritis'
The attraction pattern | is (inferred, confidence=0.5) | limited to the friendship context ('and that's it')
The speaker's attraction | is (inferred, confidence=0.5) | excludes romantic escalation beyond friendship
'girlnextdooritis' | is (inferred, confidence=0.5) | refers to a specific romantic/sexual attraction pattern
The speaker's attraction | is (inferred, confidence=0.5) | triggered by the existence of friendship
The attraction pattern | is (inferred, confidence=0.5) | involves a guy who is friends with the speaker

Throughline:
The speaker experiences a specific form of unrequited or stalled romantic attraction where they are attracted to men who are already established as their friends, but this attraction does not extend to pursuing a romantic relationship beyond that friendship.
```

**Model output:**
```
Non-Entailed Premises:
speaker | is (observed, confidence=X) | has a girlfriend
speaker | is (inferred, confidence=X) | suffers from girlnextdooratitis
speaker | is (obscured, confidence=X) | is a man

Entailed Premises:

speaker | is (inherited, confidence=X) | experiences girlnextdooritis
speaker | is (suffering, confidence=X) | from girlnextdooritys
girlnextdooritys | is (inherited by, confidence=X) | the speaker
speaker | is (has, confidence=X) | a girlfriend
girlnextdooritis | is (inherited from, confidence=X) | girlnextdooritys

Throughline:
The speaker is suffering from a condition (girlnextdooritys) that is inherited from their girlfriend.
```

---

## Example 19 — score: 0.612

**Input quote:** Even if you cannot change all the people around you, you can change the people you choose to be around. Life is too short to waste your time on people who donâ€™t respect, appreciate, and value you. Spend your life with people who make you smile, laugh, and feel loved.

**Gold output:**
```
Non-Entailed Premises:
You | should (observed, confidence=1.0) | spend your life with people who make you smile, laugh, and feel loved

Entailed Premises:
People who don't respect, appreciate, and value you | are (observed, confidence=1.0) | people not worth spending time on
You | can (observed, confidence=1.0) | change the people you choose to be around
You | cannot (observed, confidence=1.0) | change all the people around you
Life | is (observed, confidence=1.0) | too short to waste time on people who don't respect, appreciate, and value you
People who make you smile, laugh, and feel loved | are (observed, confidence=1.0) | people worth spending time with

Throughline:
You should prioritize and choose to spend time with people who respect, appreciate, and value you, while distancing yourself from those who do not.
```

**Model output:**
```
Non-Entailed Premises:
You | is (observed, confidence=X) | can change the people you are around
You | is (inferred, confidence=X) | have the capacity to change the people you choose
You | is (possible, confidence=X) | cannot change all the people you are around

Entailed Premises:

You | is (inference, confidence=X) | should spend your life with people who respect, appreciate, and make you smile
You | is (observation, confidence=X) | people who don't respect, appreciate, or value you should not be around
You | is（inference, confidence=X） | people who make you smile and laugh should be around
You | has (inference, confidence=1.5) | a desire to spend your life with people
You | is (premise, confidence=1.1) | life is too short to waste time on people who don't respect you
You | is (conclusion, confidence=1.2) | people who make you feel loved should be around

Throughline:
You should prioritize spending your life with people who treat you well, as this is the most effective way to improve your life.
```

---

## Example 20 — score: 0.730

**Input quote:** Just because you call an electric eel a rubber duck doesn't make it a rubber duck, does it? And God help the poor bastard who decides they want to take a bath with the duckie. (Jace Wayland)

**Gold output:**
```
Non-Entailed Premises:
rubber duck | is (observed, confidence=1.0) | a bath toy
God | is (inferred, confidence=0.5) | helps those in danger
duckie | is (observed, confidence=1.0) | a nickname for rubber duck
poor bastard | is (inferred, confidence=0.5) | the person considering bathing with the eel

Entailed Premises:
bathing with an electric eel | is (observed, confidence=1.0) | dangerous
calling an electric eel a rubber duck | is (inferred, confidence=0.5) | does not transform it into a rubber duck
naming | is (observed, confidence=1.0) | does not change the nature of an object
electric eel | is (observed, confidence=1.0) | not a rubber duck
Jace Wayland | is (observed, confidence=1.0) | the speaker of the quote

Throughline:
Mislabeling a dangerous entity as harmless does not alter its true nature or the risks it poses.
```

**Model output:**
```
Non-Entailed Premises:
Jace Wayland | is (observed, confidence=X) | the speaker
Jace Wayland's character | is (inferred, confidence=X) | a person who is poor
Jace Wayland is (observed, confident=X) | a person
Jace Wayland wants to take a bath with a duckie

Entailed Premises:

Jace Wayland | wants (observed, confidence=Y) | to take a bath with an electric eel
Jace Wayland calls an electric eel a duckie
Jace Wayland thinks an electric eel is a rubber duck
Jace Wayland does not think an electric eel is an electric eel
God | helps (observed, confidence=W) | the poor bastard who decides to take a bath with electric eel
Jade | is (observed/implicit, confidence=W) | a duckie
Jade | is not (observed/implicit/contrast, confidence=W) | an electric eel

Throughline:
Jace WayLAND is a person who thinks an electric eel should be called a rubber duck, and he is poor, so he decides to take a bath in a duckie.
```

---
