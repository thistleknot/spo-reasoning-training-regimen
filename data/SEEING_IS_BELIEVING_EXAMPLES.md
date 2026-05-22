# Seeing Is Believing — Before / After Comparisons

Real model outputs on holdout quotes (not seen during training).
**Before**: original QLoRA adapter trained on the structured 967-quote corpus.
**After**: SPO-fine-tuned adapter trained on 2366-quote best-of-N corpus
  (top-3 greedy-diverse completions per quote, frozen-judge scored).

Before adapter: `output/full_run_qwen35_0.8b_3ep/base-plus-facts/adapter`  
After adapter:  `output/spo_best_of_n/adapter`  
Holdout fraction: 10% of `data/train_best_of_n.jsonl` (seed=42)

---

## Quote 1

> “I am lonely, yet not everybody will do. I don't know why, some people fill the gaps and others emphasize my loneliness. In reality those who satisfy me are those who simply allow me to live with my ''idea of them.”

*(holdout reward: 0.487)*

### Before

```
Non-Entailed Premises:
I | is (observed) | am lonely
I | is (inferred) | don't know why
I | is (obscured) | some people emphasize my loneliness
I | is (unobserved) | some people fill the gaps
I | is (implied) | some people satisfy me

Entailed Premises:

I | is (inference) | some people fill my gaps
I | is（inference） | some people emphasize my lonely state
I | is（observation） | I am lonely
I | am lonely
I don't know why
those who satisfy me | is (inference/observation) | allow me to live with their idea of them
I | is (observation) | some people satisfy my loneliness
I | am lonely

Throughline:
Those who allow me to live with the idea of them are the ones who satisfy me.
```

### After

```
Non-Entailed Premises:
I | is (observed) | am lonely
I | is (inferred) | do not know why

Entailed Premises:

I | is (obsERVED) | do not know the reason for my loneliness
I | is (oberved) | some people fill the gaps
I | is (OBERVED) | others emphasize my loneliness
I |is (inferred) | some people satisfy me by allowing me to live with my 'idea of them'
I | is (Inferred) | those who satisfy me are the ones who allow me to live with their 'idea of them'

Throughline:
Those who satisfy me are the people who allow me to live up to my 'idea of them'.
```

---

## Quote 2

> “A book lying idle on a shelf is wasted ammunition.”

*(holdout reward: 0.498)*

### Before

```
Non-Entailed Premises:
shelf | is (observed) | present
book | is (observed lying) | idle
ammunition | is (observed being) | wasted

Entailed Premises:

book | is (inferred) | idle
book | is (inference) | wasted ammunition
ammunition | is characterized by (observed) | being wasted
book | is characterized by (inferred) | being idle
shelf | is characterized by (observation) | lying idle on a shelf

Throughline:
A book lying idle on the shelf is wasted ammunition.
```

### After

```
Non-Entailed Premises:
shelf | is (observed) | exists
book | is (observed] | exists
ammunition | is (observed)] | exists

Entailed Premises:

book | is (inferred) | idle
idle book | is (inferred] | wasted ammunition
ammunition | is a weapon | is used for destruction
wasted ammunition | is (inferred)] | wasted

Throughline:
A book lying idle on the shelf is wasted ammunition.
```

---

## Quote 3

> “Never be afraid to raise your voice for honesty and truth and compassion against injustice and lying and greed. If people all over the world...would do this, it would change the earth.”

*(holdout reward: 0.446)*

### Before

```
Non-Entailed Premises:
raising your voice | is (observed) | against injustice
raising your voice | can be (observed) | done by people
people | are (observed) | over the world
lying | is (observed to be) | a form of injustice
lying | is (inferred to be) | a type of greed
honesty | is (observed as) | a form of truth
compassion | is (observed in) | a form of honesty
people | are (inferred to be doing) | the act of raising their voice
earth | is (observed by) | the subject of the change

Entailed Premises:

raising your voice for honesty | is (inferred) | a positive action
raising your voice for honesty and compassion | is (inferred from) | the context of changing the earth
raising your voice for truth | is (inferred as) | a positive action that leads to change
raising your voice for compassion | is (in inferred) | a positive action leading to change
raising your voices for honesty | is (observed that) | would change the earth
raising your voices for truth | is (observed for) | would change the world
raising your voices for lying | is (inferred that) | would not change the earth
raising voices for lying | is a negative action
raising voices for greed | is (inferred by) | would not change earth
raising voices for injustice | is (observed against) | would not change Earth
raising voices for lying and greed | is (observed not to change earth)
raising voices for honesty and truth | is (obsposed to change earth)
people | are (obviously) | doing the act of raising voices
people | are (implicitly) | capable of doing so
```

### After

```
Non-Entailed Premises:
raising your voice | is (observed) | a moral action
raising your voice | can be (observed) | done by people
people | are (observed) | over the world
raising your voice | would change (observed) | the earth

Entailed Premises:

raising your voice | is a (observed) | action
raising your voice for honesty | is (observed/implicated) | a form of compassion
raising your voice for truth | is (observed/implied) | a form of honesty
raising your voice for compassion | is (observed/mimicked) | a form of truth
raising your voice for justice | is (observed/inferred) | a form of lying
raising your voice for greed | is (observed/falsely implied) | a form of injustice
people | are (inferred) | capable of raising their voice
people | are (implied) | capable of lying
people | are (implicated) | capable of lying

Throughline:
Raising one's voice for honesty, truth, and compassion against injustice, lying, and greed would fundamentally alter the course of history.
```

---

## Quote 4

> “Can you be a girl for a few seconds?""I'm always a girl" I frown."You know what I mean. Like a silly, annoying girl"I twirl my hair around my finger. "Kay.”

*(holdout reward: 0.403)*

### Before

```
Non-Entailed Premises:
I | is (observed) | am always a girl
I | is (inferred) | am a girl
I | has (observed) | twirled hair around finger
I | is (obscured) | a girl

Entailed Premises:

I | is (inference) | am a girl for a fewseconds
I | is (observation) | am always a girls
I | is (interpretation) | am a silly, annoying girl
I | is inferred to be (inference) | a girl
I | observed (observed) | I twirled hair around my finger
I | is inferred (observed) | Kay
I | is inferred from (inference) | I am a girl
I observed (observed) I frown
I | is inferred as (inference) | annoying

Throughline:
I am a girl.
```

### After

```
Non-Entailed Premises:
I | is (observed) | I am always a girl

Entailed Premises:

I | is (inferred) | I am a girl
I | is (inference) | I am annoying
I | is (observation) | I twirl my hair around a finger
I | is (response) | Kay
I | is (interpretation) | I am a silly, annoying girl

Throughline:
The speaker identifies as a girl who is annoying and silly.
```

---

## Quote 5

> “Dumbledore says people find it far easier to forgive others for being wrong than being right.”

*(holdout reward: 0.523)*

### Before

```
Non-Entailed Premises:
Dumbledore | is (observed) | a wizard
Dumbledore | is a character in the Harry Potter universe
Forgiveness | is (observed a common human trait) | a moral choice
People | is (observed to find) | it easier to forgive than to be right
Being right | is (observed as a state of being) | a moral choice

Entailed Premises:

Forgiveness | is (inferred) | easier than being right
Forgiveness | is a choice
Forgiveness | is easier than being right
People | is (inferred to be) | find it easier to forgive others for their wrongs
Forgiveness | is harder than being right
Forgiving others for being wrong | is (inferred from the comparison) | easier than being correct
Forgiveness | is an action
Being right | is a state of being
Forgiving others | is an action
People | is (to be) | find it far easier to be forgiving than to be right

Throughline:
Forgiving others for their mistakes is morally easier than being right.
```

### After

```
Non-Entailed Premises:
Dumbledore | is (observed) | the speaker
Dumbledore | is not (observed) | a person
Dumbledore | is a wizard
Dumbledore | is the protagonist of the Harry Potter series
Dumbledore | is known for being a good person
Dumbledore | has a history of being wrong
Dumbledore | is often criticized for being wrong
Dumbledore's actions | are (observed) | sometimes criticized
Dumbledore's actions are | are (observed to be) | wrong

Entailed Premises:

Dumbledore | is (inferred) | a person
Forgiveness | is (inferred to be) | easier than being right
Forgiveness | is easier than being right
Dumbledore | is wrong
Dumbledore | finds it easier to forgive others for their wrongness than for their rightness

Throughline:
Dumbledore finds it easier to forgive people for their wrongness than to be right.
```

---

## Quote 6

> “Happy Hunger Games! And may the odds be ever in your favor.”

*(holdout reward: 0.565)*

### Before

```
Non-Entailed Premises:
The odds | are (observed) | in your favor
Happy Hunger Games | are (observed events) | happening
May the odds | be (observed) | ever in your favor

Entailed Premises:

The odds | are (inferred) | in your favor because of the Happy Hunger Games
Happy Hunger Games | is (inferred) | a positive event
May the odds | are (inference) | in your favor

Throughline:
The Happy Hunger Games are a source of hope and favor for the speaker.
```

### After

```
Non-Entailed Premises:
The speaker | is (observed) | happy
The speaker | is happy (observed) | Hunger Games
The speaker | is in favor of (observed) | odds

Entailed Premises:

The speaker | is (inferred) | the odds are in favor of the speaker
The speaker | is hoping (observed) | for the odds to be in favor
The speaker | is asking (observed) | may the odds be ever
The speaker | is speaking to (observed) | a listener
The speaker | is using (observed) | the phrase 'Happy Hunger Games'
The speaker | is expressing (observed) | hope
The speaker | is appealing to (observed) │ odds

Throughline:
The speaker is expressing hope that the speaker's situation or interests will be favorable.
```

---

## Quote 7

> “The reason I talk to myself is because Iâ€™m the only one whose answers I accept.”

*(holdout reward: 0.428)*

### Before

```
Non-Entailed Premises:
I | is (observed) | talk to myself
I | is (inferred) | am the only one whose answers are accepted
I | is (obscured) | am the only person whose answers I accept

Entailed Premises:

I | is (inference) | am the only individual whose answers I accept
I | is (observation) | talk to myself

Throughline:
I am the only one who accepts my own answers.
```

### After

```
Non-Entailed Premises:
I | is (observed) | talk to myself
I | is (inferred) | am the only one whose answers are accepted

Entailed Premises:

I | is (inference) | am the only person whose answers I accept
I | is (observation) | talk to myself

Throughline:
I am the only person who accepts my own answers.
```

---

## Quote 8

> “Remember, if the time should come when you have to make a choice between what is right and what is easy, remember what happened to a boy who was good, and kind, and brave, because he strayed across the path of Lord Voldemort. Remember Cedric Diggory.”

*(holdout reward: 0.475)*

### Before

```
Non-Entailed Premises:
Lord Voldemort | is (observed) | a dark entity
Cedric Diggory | is (observed to be remembered) | a boy
Cedric Daggory | is (obsERVED to be remembered) | Cedric Diggory
Cedric Diddory | is (obsessed to be remembered) | the boy
Cedric | is (obsessed) | the boy
Lord Voldemort | is a villain
Cedric | was good
Cedric | strayed across the path
Cedric | had a path
Lord Voldemort | was brave
Cedric | did not know what was right
Cedric | knew what was easy
Cedric | chose the easy path
Cedric Dodd | is (obsessed with) | Cedric Diddory
Cedric | made a choice
Cedric | could have chosen the right path
Cedric's choice | was (obsessed) | easy
Cedric's path | was (obsessed to be) | easy
Creditor | is (obsessed by) | Cedric Dodd
Cedric | has a path
Cedric was good
Cedred | is (obsessed for) | Cedric Daggory
Cedric was brave
Cedred | was good
Creditor | has a path
Lord Voldemort's path | was (inferred) | easy
Lord Voldemort's path was (inferred)| easy
Lord Voldemort's choice | was (inferred to be) | easy

Entailed Premises:

Cedric Diggore | is (obsessed about) | Cedric Deggory
Cedric's character | was (obsessed with)| good
Cedric's actions |
```

### After

```
Non-Entailed Premises:
Cedric Diggory | is (observed) | a boy
Lord Voldemort | is (observed, inferred) | a dark entity
Cedric Daggory | is (obsERVED) | a boy
Cedric Deggory | is (observd) | a boy

Entailed Premises:

Cedric Diggor | is (inferred) | strayed across the path
Cedric Dggory | is (inferred, inferred) | was good
Cedric Dggy | is (inferred), inferred) | was brave
Cedric Ddgy | is (infrned), infrned) | was kind
Cedric Ddgory | is (infrnd), infrnd) | was good

Throughline:
Cedric Diggy was a good, kind, and brave boy who strayed across the path, leading to his fate.
```

---

## Quote 9

> “Imagine smiling after a slap in the face. Then think of doing it twenty-four hours a day.”

*(holdout reward: 0.619)*

### Before

```
Non-Entailed Premises:
slap in the face | is (observed) | an event
smiling | is (observed, implied) | a positive emotion
doing it twenty-four hours a daily | is (observed/implies) | a routine

Entailed Premises:

doing it twenty-four hours daily | is (inferred) | a way to maintain the positive emotion
slap in the faces | is (inferred, implied) | a source of the positive emotion
smiling | is an action | is (observed/implied) | taken after a slap in the faces
slap in the facing | is (observed/inferred) | a slap in the face
doing it twenty-four days a day | is (inferred/implies) | the same positive emotion
slap | is (observed/facts) | a slap in a face

Throughline:
Maintaining a positive emotional state through a daily routine of self-correction (smiling after a slap) is a sustainable strategy.
```

### After

```
Non-Entailed Premises:
slap in the face | is (observed) | an event
smiling after a slap in the faces | is (observed） | a reaction
doing it twenty-four hours a daily | is (observed ) | a habit

Entailed Premises:

doing it twenty-four hours daily | is (inferred) | implies a daily routine
smiling after a slip in the face | is an example of doing it twenty-four times a day
slap in the faces | is an example of smiling after a slip in the faces
smiling after a slp in the face | is a form of doing it twenty-four a day

Throughline:
Smiling after a slap in a face is a daily habit.
```

---

## Quote 10

> “Men go to far greater lengths to avoid what they fear than to obtain what they desire.”

*(holdout reward: 0.536)*

### Before

```
Non-Entailed Premises:
Men | is (observed) | have desires
Men | is (inferred) | are rational beings
Men | is (obscured) | are often driven by irrational impulses
Men | is (assumed) | are capable of rational choice
Men | is (implied) | are capable of avoiding fear
Men | is (implicit) | are capable of obtaining desires

Entailed Premises:

Men | is (obsured) | prioritize safety over gratification
Men | is (derived) | prioritize avoiding fear over acquiring desire
Men | is (implicated) | prioritize safety over desire
Men | is inferred | prioritize avoiding fear over obtaining desire
Men | is implied | prioritize safety over desire

Throughline:
Men prioritize avoiding fear over obtaining what they desire.
```

### After

```
Non-Entailed Premises:
Fear | is (observed) | a negative emotion
Desire | is (observed） | a positive emotion
Lengths | is (observed ) | physical or mental actions
Men | is (observed ） | a demographic group

Entailed Premises:

Fear | is (inferred) | causes avoidance of desires
Fear | is (inherited) | causes avoidance of fears
Men | is (inferred ) | have a preference for safety over excitement
Men | is (oberved) | go to far greater lengths
Men | is (induced) | avoid what they fear
Men | is (derived) | avoid what they desire
Men | is (result) | go to far less lengths to obtain what they desire

Throughline:
Fear is a stronger motivator for avoidance of negative outcomes than desire is for positive outcomes.
```

---

## Quote 11

> “The truth." Dumbledore sighed. "It is a beautiful and terrible thing, and should therefore be treated with great caution.”

*(holdout reward: 0.512)*

### Before

```
Non-Entailed Premises:
Dumbledore | is (observed) | sighed
Dumbledore | is a wizard | is (observed, inferred) | true
Dumbledore | is an advisor | is (observed), inferred) | true
The truth | is (observed). | is (observed): true
The truth | should be (observed). | should be (obsERVED): true
The truth is (observed).| is (observed): beautiful
The truth is (oberved).| is (oberved): terrible

Entailed Premises:

Dumbledore | is (inferred) | treats the truth with great caution
Dumbledore | is in (observed) | a state of being
Dumbledore | is aware of (observed) | the nature of the truth
Dumbledore | is concerned about (observed) | whether the truth is beautiful or terrible
Dumbledore | is considering (observed) | if the truth is beautiful or not
Dumbledore | is evaluating (observed) | two distinct attributes of the truth
D Dumbledore | is considering (obsERVED) | the potential for the truth to be beautiful
Dumbledore | is assessing (observed) | potential negative attributes of the truth
The truth | is a (observed) | terrible thing
The truth | is beautiful (observed) |
The truth | is terrible (observed) |

Throughline:
Dumbledore evaluates the truth as a complex entity that possesses both positive and negative attributes, necessitating a cautious approach to its treatment.
```

### After

```
Non-Entailed Premises:
Dumbledore | is (observed) | sighed
Dumbledore | is a wizard | is (observed, inferred) | true
Dumbledore | is the master | is (observed), inferred) | true
The truth | is (observed}, inferred) | true

Entailed Premises:

The truth | is (inferred) | a beautiful thing
The truth | is a terrible thing
The truth | should be treated with great caution

Throughline:
The truth is a beautiful and terrible phenomenon that requires careful handling.
```

---

## Quote 12

> “You do have a story inside you; it lies articulate and waiting to be written â€” behind your silence and your suffering.”

*(holdout reward: 0.585)*

### Before

```
Non-Entailed Premises:
You | is (observed) | have a story inside you
You | is (inferred) | are silent
You | is (obscured) | are suffering
You | is (unarticulated) | are waiting to be written

Entailed Premises:

You | is (inference) | have a story
You | is (observation) | lie articulate
You | is (premise) | are waiting to have their story written
You | is (postcondition) | are waiting to write
You | is (interpretation) | your silence and suffering are the means to write your story
You | is(inference) | your silence and sufferings are the container for your story
You |is (inference) | your story is articulate
You | is(inferred) | your story lies behind your silence and suffering

Throughline:
Your silence and suffering are the necessary conditions that allow your hidden story to be articulated and written.
```

### After

```
Non-Entailed Premises:
You | is (observed) | have a story inside you

Entailed Premises:

You | is (obsessed) | have a story
your silence | is (inferred) | a barrier to writing the story
your suffering | is (inferred to be) | a barrier to expressing the story
the story | is (inferred from) | lies articulate and waiting tobe written
the story | is characterized by (obsessed) | being written behind silence and suffering

Throughline:
The speaker perceives the speaker's internal narrative as a hidden, articulate, and waiting story that is currently suppressed by silence and suffering.
```

---

## Quote 13

> “Life is a book and there are a thousand pages I have not yet read.”

*(holdout reward: 0.463)*

### Before

```
Non-Entailed Premises:
Life | is (observed) | a book
Life | is (inferred) | a physical object with pages
Life | is (obscured) | by the fact that I have not read all pages

Entailed Premises:

Life | is (obsessed) | a book
I | is (obsessed by) | the fact that I have read fewer pages than there are thousand pages
I | is (inferred by) | the statement that there are a thousand pages
I | has (obsessed by) |- the statement that I have not yet read all pages
I | is a reader | is (obsessed to) | read all pages
I have (obsessed by)|- not read all pages
I am (obsessed by)| a reader
I | is (obscurred by) | the statement
I | is (factual) | have not read all pages
There | is (obsessed with) | a thousand pages
There | is a (obsessed by) thousand pages

Throughline:
I am a reader who has not yet read all the pages of life.
```

### After

```
Non-Entailed Premises:
Life | is (observed) | a book
Life | is (inferred) | has a finite number of pages
Life | is (observation) | has a thousand pages

Entailed Premises:

Life | is (inference) | has a finite total number of pages
Life [is (inference) | a book]
Life [is (observation) | has not yet been read]

Throughline:
Life is a book with a finite number of pages that have not yet been read.
```

---

## Quote 14

> “The worst type of crying wasn't the kind everyone could see--the wailing on street corners, the tearing at clothes. No, the worst kind happened when your soul wept and no matter what you did, there was no way to comfort it. A section withered and became a scar on the part of your soul that survived. For people like me and Echo, our souls contained more scar tissue than life.”

*(holdout reward: 0.550)*

### Before

```
Non-Entailed Premises:
the worst crying was tearing at clothes | is (observed) | true
the worst crying was wailing on street corners | is (observed, false)
people like me and Echo | is (observed), true
the worst kind of crying happened when your soul weeped | is (observed', true)
the worst kind of suffering was a section that wept | is (observed'')', true
the worst kind happened when no matter what you did there was no way to be comforted | is (obsessed', true)
the soul wept | is (inferred), true
the soul wept and no way to comfort it | is (inferred', true)
the section that wept and became a scar on your soul | is (inferred'), true
the soul contained more scar tissue than the rest of your soul | is (obsessed'), true
the soul weep and no way to comfort | is (obsessed), true
the soul contained scar tissue | is (obsessed)', true
the soul contained life | is (obsessed)'s', true
the soul contained people like me | is (obsessed)

Entailed Premises:

the soul wept | was (obsessed) | true
the soul weeping and no way to comfort the soul | was (obsessed), true

the soul wept and became a section that survived | was (obsessed)', true

the soul contained more scar tissues than life | was (obsessed)'s), true

the soul contained scar tissue that survived | was (inferred), true

the soul contains more scar tissue than life | is (obsessive)', true

the soul contains scar tissue | is (inferred)', true

the soul weep and there is no way to be comfort |
```

### After

```
Non-Entailed Premises:
street corners | is (observed) | where crying happens
people | is (observed） | can see the worst kind of crying
people | is (inferred) | have souls that contain more scar tissue than life

Entailed Premises:

worst kind of crying | is (inferred to be) | the kind that happens when your soul wept
worst kind of crying (wailing on street corners, tearing at clothes) | is (inferred not to be) | the worst kind
worst kind of cry | is (inferred NOT to be) | the wailing on street corners
worst kind ofcry | is (inferred as) | the tearing at clothes
worst kind ofcri | is (inferred that the worst kind happened when) | your soul wept
people | is (explicitly stated) | have souls that contained more scar tissue thanlife
people | is (implicit) | have souls that survived a section that wept
people |is (inferred) | their souls contained more scar tissuethan life

Throughline:
The worst kind of crying is not the visible tears on the street, but the internal, soul-level pain that persists even after one's actions, leaving a permanent scar on one's soul.
```

---

## Quote 15

> “Clary felt suddenly annoyed. "When the self-congratulatory part of the evening is over, maybe we could get back to saving my best friend from being exsanguinated to death?""Exsanguinated," said Jace, impressed. "That's a big word.""And you're a big-""Tsk tsk," he interupted. "No swearing in church.”

*(holdout reward: 0.748)*

### Before

```
Non-Entailed Premises:
Clary | is (observed) | a big friend
Jace | is (observed by Tsk tsk) | a big friend

Entailed Premises: Jace | is (obsined) | annoyed by Clary's comment
Jace | is impressed by (observed) | the word 'exsanguinated'
Jace | is annoyed by (observed) the self-congratulation in Clary's comment
Clary | is speaking (observed) in a self-congratulatory way
Clary | is talking (observed) about saving her best friend from being exanguinated to death
Jace | is interrupting (observed) by (obsined) Tsk tsk
Jace | is telling (observed) Clary not to swear in church
Jace | is surprised by (observed) Clay's use of the word 'big'
Jace | has (observed) a big friend named Clary
Jace | is talking (obsined) to Clary
Jace's tone | is (observed to be) Tsk tsk

Throughline:
Clary made a self-congratulating comment about saving her best friend, which Jace found amusing and annoyed, leading to a humorous interruption where he reminded her not to swear in church.
```

### After

```
Non-Entailed Premises:
Clary | is (observed) | felt suddenly annoyed
Jace | is (observed] | interupted by Clary
Jace | is a big-word speaker (observed) | Tsk tsk
Jace | is impressed (observed) | by Clary's use of 'exsanguinated'
Clary | is a big-word person (observed) | 'Exsanguinated'

Entailed Premises:

Clary | is (inferred) | uses a slang term for a serious medical condition
Clary | is impressed (obsined) | by Jace's use of 'ex'
Jace | is annoyed (observed) | about Clary's use of slang
Jace | is interested (observed) | in saving Clary from being exsanguinate
Jace | is proud (observed) | of Clary's use of the word 'big'
Jace | uses slang (observed) | for 'exsanguinated', which is a big word
Jace | is offended (observed) | because Clary used slang in church
Jace | is impulsive (observed) | to interrupt Clary
Jace's speech is characterized by swearing (observed) | but he is not allowed to swear in church

Throughline:
Clary is using slang to express a serious, urgent concern about her best friend's death, which Jace finds offensive and inappropriate for a religious setting.
```

---
