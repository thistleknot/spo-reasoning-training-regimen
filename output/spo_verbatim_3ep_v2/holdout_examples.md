# SPO Holdout Inference Examples

**Adapter:** output/spo_verbatim_3ep_v2/adapter
**Examples:** 20

---

## Example 1 — score: 0.819

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
change_it | is (inferred, confidence=0.5) | implies a negative outcome
change_the_way_you_think | is (inferred/observed, confidence=0.3) | implies a negative or neutral outcome
complain | is (inferred) | implies a negative action

Entailed Premises:

change_it | is (observed, confidence=1.0) | is the recommended action when you don't like something
change_the_way_you_thing | is (inferred_observed, confidence=0) | is the alternative action when you can't change it
change_it | is a valid action
change_the_way_you think about it | is a valid action

Throughline:
If you cannot change a thing, you should change your perspective on it.
```

---

## Example 2 — score: 0.734

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
time | is (observed, confidence=1.0) | 4 a.m.
friends | is (observed/derived, confidence=0.5) | matter
time | is (inferred, confidence=0.2) | irrelevant to friendship

Entailed Premises:

time | is (inference, confidence=0.3) | 4 a.p.m.
time | is (observation, confidence=1.5) | 4 a.a.m.
time | has (inference, confidence=-0.5) | 100% of its value
time | is (fact, confidence=1.2) | 4 a.e.m.
time | matters (inference, confidence=(-0.5, 0.5)) | 4 a.m.e.m.
time (friends) | is (inference/observation, confidence=0.7) | 4 a.g.m.
time (time) | is (inferred/observation, confidence=1) | 4 a.

Throughline:
The quality of friendship is determined by the quality of the time spent with friends.
```

---

## Example 3 — score: 0.714

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
time | is (observed, confidence=1.0) | a variable that can be controlled
speed | is (observed/derived, confidence=0.5) | irrelevant to the outcome
stopping | is (observed/inferred, confidence=0.2) | a necessary condition for success

Entailed Premises:

1.  | is (observed/fact, confidence=1.5) | stopping is a necessary condition for success
2.  | is (inferred, confidence=0) | stopping is not a sufficient condition for success
3.  | is (observation, confidence=1.2) | the outcome is independent of the rate of stopping
4.  | is (fact, confidence=1.3) | the outcome is not dependent on the rate of stopping
5.  | is (factual, confidence=1.1) | the outcome is dependent on the rate of going
6.  | is (logical, confidence=0.8) | going at a slow rate does not stop the outcome
7.  | is (logic, confidence=0.7) | going at a fast rate does not stop the result
8.  | is (conclusion, confidence=0.6) | the outcome is determined by the rate of going, not the rate of stopping
9.  | is (premise, confidence=0.9) | going at a low rate does not stop the process
10. | is (premise/observation, confidence=0.4) | going at a high rate does not stop the event

Throughline:
The
```

---

## Example 4 — score: 0.631

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
someone | is (observed, confidence=1.0) | exists
someone | is (inferred, confidence=0.5) | has the power to tell the truth
someone | is (obscured, confidence=0.2) | has the power of the truth

Entailed Premises:

someone | is (inference, confidence=0.3) | has the power
someone | is (observation, confidence=1.5) | will tell the truth
someone's power | is (inference/observation, confidence=0.7) | to tell the truth
the truth | is (inference) | something that can be told
someone | is (fact, confidence=1.2) | will tell the Truth
someone | is (factual, confidence=1.3) | will tell the fact
someone | is (opinion, confidence=0.8) | will tell the facts
someone | is (action, confidence=0.9) | will tell the truths
someone | is (consequence, confidence=0.6) | will be told the truth
someone | has (inference, confidence=-0.3) | the power to tell the truths
someone's power | has (inference/observation/observation, confidence=1) | the power to speak the truth
someone | will (inference, confidence=(-0.2)) | tell the truth
someone will (inference, confident=0.5) tell the truth
someone is (inference, confident=-0.3) telling the truth
someone will tell the truth
the speaker | is (in
```

---

## Example 5 — score: 0.798

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
God | is (observed, confidence=1.0) | created war
God | is (inferred, confidence=0.5) | created geography
Americans | is (inferred/observed, confidence=0.3) | learned geography

Entailed Premises:

God | is (independent of) | created geography
God | is (dependent of) | created war
Americans | is (observed/derived, confidence=0.2) | learned geography
Americans | is independent of | created geography
Americans' learning of geography | is (independent/observed, confidence=-0.5) | caused by God
God | is (created) | war
God | is independent of | Americans' learning of geography

Throughline:
God created war to teach Americans about geography.
```

---

## Example 6 — score: 0.748

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
understanding | is (observed, confidence=1.0) | a necessary condition for fear reduction
nothing in life | is (observed/assumed, confidence=1.5) | to be feared
nothing in life | can (observed/assumes, confidence=1.25) | be understood
nothing in life | should (observed/assumption, confidence=1.1) | be feared

Entailed Premises:

understanding | is (inferred, confidence=0.5) | a prerequisite for fear reduction
understanding | is a means (observed/assumptions, confidence=0.75) | to fear less
understanding | is to be understood (observed/assumptive, confidence=1.3) | more
understanding | is the time (observed/assumpment, confidence=0.8) | to fear less

Throughline:
Understanding is the necessary condition for reducing fear in life.
```

---

## Example 7 — score: 0.803

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
Sleep | is (observed, confidence=1.0) | a state of rest
Love | is (observed/Inferred, confidence=0.5) | a relationship
We | is (observed/inferred, confidence=0) | are the subjects of the action

Entailed Premises:

We | is (obsERVED, confidence=1.5) | are the subjects
Love | is (inferred, confidence=0,5) | will make us
We | is (inferred/observed, confidence=0,2) | will not make Love
Sleep | is (inferred/predicted, confidence=0,3) | with me

Throughline:
Love is the agent that causes the desired outcome (sleeping with me).
```

---

## Example 8 — score: 0.830

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
Vanity | is (observed, confidence=1.0) | a state of self-love
Pride | is (observed/derived, confidence=0.5) | a state of superiority
Pride | is often used synonymously with vanity (observed/derived) | is a state of superiority
Vanity | is often used synonymous with pride (observed/derived/derived) | is often used synonymyously with pride

Entailed Premises:

Pride | is (inferred, confidence=0.75) | a state related to our opinion of ourselves
Pride | is not (inferred, confidence=-0.5) | necessarily vain
Vanity | is not (in inferred, confidence=-0.75) necessarily proud
Pride | is related to our opinion of others (observed/derived/inferred) | vanity is related to what we would have other people think of us
Pride | is a state of self-loft (observed/derived/factual) | vanity is a state of self-obsession
Pride | is more about our opinion of ourselves (observed/derived/) | vanity is more about what we would have others thought of us

Throughline:
Pride is a state of self-perception and self-love, whereas vanity is a state of projection of others' perceptions of one's self.
```

---

## Example 9 — score: 0.786

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
Man | is (observed, confidence=1.0) | often becomes what he believes he is
Man | is (inferred, confidence=0.5) | capable of acquiring new capacities through belief
Man | is (obscured, confidence=0.0) | incapable of acquiring new capacities through believing

Entailed Premises:

1.  | is (observed/derived, confidence=0.75) | believing one can do something leads to acquiring the capacity to do it
2.  | is (inferred/observed, confidence=0.25) | believing one cannot do something leads to becoming incapable of doing it
3.  | is (obscurred, confidence=0.3) | believing one can't do something is a sufficient condition for becoming incapable
4.  | is (implicit, confidence=0.1) | believing one can does not guarantee acquiring the capacity to do something

Throughline:
Belief in one's ability to perform an action is a sufficient condition for acquiring the capacity to perform that action, whereas belief in one's inability is a sufficient condition for losing the capacity to perform that same action.
```

---

## Example 10 — score: 0.720

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
sorrow | is (observed, confidence=1.0) | a negative emotional state
sorrow | is caused by (observed, confidence=X) | grief
grief | is caused by (inferred, confidence=X) | sorrow
sorrow | is a result of (observed, confidence=x) | grief

Entailed Premises:

people | is (inferred, confidence=1.5) | can suffer great sorrow
people | is (inference, confidence=1.25) | can also suffer grief
people | is (observed/derived, confidence=1.75) | are capable of loving strongly
people | is (observation, confidence=1.1) | can also suffer sorrow
sorrow | serves to (inference, confidence=X) | counteract grief
sorrow | heals (inference, confidence=x) | people

Throughline:
Loving strongly is a necessary condition for suffering sorrow, which in turn serves to heal the individual from grief.
```

---

## Example 11 — score: 0.863

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
confronting dark parts of yourself | is (observed, confidence=1.0) | involves wrestling with demons
confronting dark part of yourself | is (inferred, confidence=0.5) | involves working to banish them
confronting dark self | is (inferred/observed, confidence=0.3) | involves working to illuminate them
confronting self | is (independent of) | angels singing

Entailed Premises:

confronting dark parts | is (inferred) | involves working to sing
confronting dark selves | is (inferred from) | angels sing
confronting self with demons | is (inferred to be) | involves wrestling
confronting dark aspects | is (inferred by) | angels sing
wrestling with demons | is (observed) | involves confronting dark parts
confronting dark aspect of self | is (inference) | involves wrestling with dark parts
confrontation | is (inferred context) | involves wrestling with darkness
confrontation | involves (observed) | wrestling with demons
conclusion | is (inferred conclusion) | angels sing

Throughline:
Wrestling with one's demons leads to the singing of angels.
```

---

## Example 12 — score: 0.820

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
good fall | is (observed, confidence=1.0) | a physical event
fall | is (observed/derived, confidence=0.5) | a metaphor for a significant event

Entailed Premises:

good fall | is (inferred, confidence=0.3) | a significant event
fall | is a metaphor for a significant change
knowing where you stand | is (observed/inferred, confidence=0) | requires a good fall
good fall | is a metaphor for significant change
significant change | is (observed/derived, confidence=0.2) | requires a good change

Throughline:
Significant change requires a good change.
```

---

## Example 13 — score: 0.758

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
I | is (observed, confidence=1.0) | love deadlines
I | is (inferred, confidence=0.5) | love the whooshing sound
I | is (observation, confidence=1.5) | love deadlines

Entailed Premises:

I | is (inference, confidence=0.75) | love the noise of deadlines
I | is（inference, confidence=1.75）| love the whooshing of deadlines
I | loves the whooshing noise of deadlines
I loves deadlines
I | is in (inference, confidence=-0.5) | loves the whoosh noise of deadlines

Throughline:
I love the excitement and anticipation of deadlines.
```

---

## Example 14 — score: 0.753

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
reading the newspaper | is (observed, confidence=1.0) | a necessary condition for being informed
being uninformed | is (observed/derived, confidence=0.5) | a state of not reading the newspaper
mis-informed | is (obsERVED, confidence=0.0) | a state of reading the newspaper

Entailed Premises:

1.  | is (observed/inferred, confidence=0.2) | reading the newspaper | is (inferred, confidence=0) | a sufficient condition for being mis-informed
2.  | is (inferred/observed, confidence=0.3) | not reading the newspaper | is a sufficient condition for being uninformed
3.  | is (derived, confidence=0) for being mis-informed | is (inferred) | being uninformed
4.  | is (obscured, confidence=0.1) | being mis-informed | requires (inferred) | not reading the paper
5.  | is (implicit, confidence=0.4) | being mis-informing | requires (inferred, confidence=-0.5) | reading the newspaper
6.  | is (explicit, confidence=1.5) | being uninformed | requires (inference, confidence=-0.2) | not reading the news

Throughline:
Reading the newspaper is a necessary condition for being uninformed, and reading the newspaper is a sufficient condition for mis-informed.
```

---

## Example 15 — score: 0.860

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
most people | is (inferred, confidence=0.5) | never listen completely
people | is (inferred/observed, confidence=1.0) | talk

Entailed Premises:

most people | is (observed, confidence=0.0) | never listen completely

Throughline:
Most people fail to listen completely when they talk.
```

---

## Example 16 — score: 0.741

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
I | is (observed, confidence=1.0) | I am honest
I | is (obscured, confidence=0.3) | I don't always say what feels
I | is (implicit, confidence=0.2) | I am not always honest

Entailed Premises:

I | is (inference, confidence=0.8) | I am not saying I am a liar
I | has (inference, confidence=-0.5) | The truth sometimes hurts
I | is (observation, confidence=1.5) | I don't say what I feel
I | is (explicit, confidence=1.3) | I am honest

Throughline:
I am honest, but I do not always say what I feel, and I am not a liar.
```

---

## Example 17 — score: 0.729

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
trap | is (observed, confidence=1.0) | a trap
trap | is (inferred, confidence=0.5) | a challenge

Entailed Premises:

trap | is (inference, confidence=0.3) | a trap if you don’t know about it
trap | is (observation, confidence=1.5) | a trap if known
trap | is (definition, confidence=1.2) | a challenge if known
trap | definition | is (inference) | a trap if unknown
trap | definition | includes (inference, confidence=-0.5) | challenge

Throughline:
Knowing about a trap makes it a challenge, whereas not knowing makes it a trap.
```

---

## Example 18 — score: 0.735

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
I | is (observed, confidence=1.0) | suffer from girlnextdoorsitis
I | is (inferred, confidence=0.5) | the guy is friends with me

Entailed Premises:

I | is (inference, confidence=0.75) | the guy's friendship is the cause of my suffering
I | is (observation, confidence=1.5) | the situation is described as 'girlnextdoorsitis'
I | is (definition, confidence=1.75) | girlnextdoorsitis is a condition characterized by a specific type of friendship
I | is (context, confidence=1.25) | the situation involves a guy who is friends with me

Throughline:
The speaker's suffering is caused by a specific type of relationship with a male friend.
```

---

## Example 19 — score: 0.742

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
You | is (observed, confidence=1.0) | can change the people you change
You | is (inferred, confidence=0.5) | can change the time you spend with people
You | is (obscured, confidence=0.0) | can't change the people you change

Entailed Premises:

You | is (inference, confidence=0.75) | can change people you choose to be
You | is (observation, confidence=1.5) | can't change all the people around
You | is (premise, confidence=1.25) | can change time spent with people
You | can change people you change
You choose to be around | is (inference/observation, confidence=0.8) | people who make you smile
You choose to be with | is (inference) | people who make people smile
You choose to change | is (inference/premise, confidence=0.9) | people you choose to be

Throughline:
You should spend your life with people who respect, appreciate, and make you smile, as this is the most effective way to improve your life.
```

---

## Example 20 — score: 0.770

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
Jace Wayland | is (observed, confidence=1.0) | the speaker
Jace Wayland's character | is (inferred, confidence=0.5) | a person who is poor
Jace Wayland is (observed, confident=1.0) |||
Jace Wayland calls an electric eel a duck

Entailed Premises:

Jace Wayland | calls (observed, confidence=X) | an electric eel a "duckie"
Jace Wayland wants (observed, confidence=x) | to take a bath with a duckie
Jace Wayland thinks (observed, confidence=y) | an electric eeling is a rubber duck
Jace Wayland believes (observed, confidence=z) | an electric eely is a rubber duck
God | is (observed/impersonal, confidence=1.5) | helps the poor bastard who decides to take a bath with an electric eel
Jace Wayland considers (observed, confidence=w) | an electric eelly to be a rubber duck

Throughline:
Jace WayLAND is a person who is poor and believes that an electric eel is a rubber duck.
```

---
