# Lessons Learned


Three compounding bugs caused the initial SPO training to produce worse output than the base adapter. Each one is subtle and worth documenting because they are easy to repeat in any RL-from-feedback setup.

### 1. Greedy decoding without repetition controls loops forever — but the wrong controls break headers

`model.generate()` with `do_sample=False` can lock onto a high-probability token sequence and repeat it indefinitely. The model is not broken — it is being perfectly greedy. A single repeated line scores just as well on format metrics as a unique one, so the problem is invisible to offline evaluation.

**Wrong fix (common trap):** `repetition_penalty=1.3` penalises **all** tokens that appear earlier in the full input+output context — including the prompt itself. If the prompt contains `Non-Entailed Premises` in an instruction list (which it does), `repetition_penalty` prevents the model from outputting those exact tokens, so it produces garbled variants like `Non-EntailedPremise:` or `Non-Entailed Prems:`. Similarly, `no_repeat_ngram_size=4` blocks any 4-gram that appears in the prompt from being generated, and `Non Entailed Premises` is exactly 3–4 tokens in Qwen's tokenizer, making this setting destructive.

**Correct fix:** Use `no_repeat_ngram_size=6` (or higher) without `repetition_penalty`. A 6-gram constraint prevents exact-line echoing without blocking the 3–4-token header sequences. A well-trained adapter will not loop; if it does, prefer `no_repeat_ngram_size=6` over `repetition_penalty`.

```python
out = model.generate(
    **inputs,
    max_new_tokens=384,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
    no_repeat_ngram_size=6,   # safe: does not block header tokens
    # repetition_penalty=1.3  # NEVER: breaks headers that echo prompt words
)
```

### 2. Format-only reward metrics actively reinforce repetition

The original SPO reward checked only the *presence* of `|...|...|` delimiters and an `observed`/`inferred` tag. A line repeated eight times passes all three checks eight times and earns a reward of 1.0. The reward function was measuring compliance with a template, not quality of reasoning.

**Fix:** Add a uniqueness gate before any format check. If `unique_lines / total_lines < 0.5`, return 0.0 immediately. Also detect tautological triplets where the subject appears verbatim in the object field (e.g. `the speaker | is | is the speaker`) and deduct proportionally. Reward functions must penalise the failure modes they are meant to prevent, not just reward the happy path.

### 3. Scoring gold outputs against gold ground truths yields uniform reward

This is the most insidious bug. When `compute_step()` evaluates gold `output_texts` against gold `ground_truths`, every clean sample scores ~1.0. All training examples receive equal weight. SPO becomes plain SFT with extra steps and no signal. The training loss may look fine; avg_reward hovering at a flat ~0.8 is the only diagnostic clue.

**Fix:** Pre-score each training sample *before* the training loop using data-quality signals that are independent of format compliance: unique-premises ratio, mean predicate specificity, and subject dominance ratio. Embed these scores as `precomputed_reward` in the training batch and use them in `compute_step()` instead of live evaluation. This separates reward differentiation (done offline at dataset construction time) from reward evaluation (which is inherently circular when gold data is both input and ground truth).

### 4. SPO-as-weighted-SFT cannot fix pre-trained abbreviation habits


After 5 epochs on 967 records, the adapter still outputs `Non-Entailed Prems:` and `Entailed Prims:` instead of the full header names from the training corpus. Every training record has the correct full names; SPO is applied on top; the model ignores the correction anyway.

The reason is architectural: SPO as implemented here is **offline weighted SFT**. `compute_step()` upweights loss on high-quality gold tokens and downweights loss on lower-quality ones. It never generates a bad output at training time and penalises it. The base model's abbreviation tendency — reinforced by Qwen pre-training — wins because it is never directly penalised in the gradient signal.

**Fix (future):** Use online RL (GRPO or PPO). Generate a candidate output, score it with `evaluate_triplet_correctness`, compute a reward signal, and backpropagate through the actual generated tokens. This is the only training loop that can penalise a garbled header that the model chose to produce.

**Short-term mitigation:** Constrained decoding (prefix forcing) or a few-shot prefix in the inference prompt that starts the output with the correct headers (`Non-Entailed Premises:\n`) can force correct headers at generation time without retraining.

### 5. For small models, balance the training data across (regimen × length) strata

Single-regimen training on the facts format produced clean, full-length headers throughout. Once training was extended to a second regimen sharing the same semantic domain (logical entailment reasoning), headers degraded to abbreviations (`Entailed Prims:`, `Non-Entailed Prems:`). The original single-regimen training was fine; the problem was introduced entirely by combining regimens.

**Mechanism:** Both regimens use nearly identical header vocabulary ("Entailed Premises:", "Non-Entailed Premises:"). When trained jointly, the gradients from each regimen are co-aligned but not identical — they point at the same token neighborhood with slightly different downstream structure expectations. The model converges to a weighted average over that neighborhood, and the base model's pre-training bias toward abbreviated forms (which were never fully suppressed, just outvoted by the single-regimen signal) re-emerges as the dominant pattern. A secondary amplifier is length imbalance: if one regimen's training records are systematically longer, those records dominate the gradient signal for the shared header tokens, reinforcing that regimen's abbreviation habits.

**Why text prompts cannot solve this:** A text tag like `[FACTS]` occupies the same embedding space as all other vocabulary. For a 0.6B parameter model, learning a clean conditional `[FACTS] → full header names` is infeasible when the tag is semantically adjacent to the content tokens and the regimens share near-identical output vocabulary. This is scale-dependent: larger models (7B+) can learn such conditionals from text prompts alone; small models cannot.

**Why special tokens are also wrong:** Special tokens require downstream users to inject a model-internal routing token they have no natural reason to know about. This creates a hidden contract — the model silently degrades for any user who doesn't know the required prefix. It's bad API design disguised as a training fix.

**The correct fix — stratified sampling across (regimen × length) strata:**
The problem is that the combined training mix is dominated by whichever regimen happens to have more or longer records. Fix this by ensuring every (regimen, prompt-length bucket) cell contributes equally to each training epoch.

The `sample_mixture()` function in `src/run_ablation_matrix.py` now implements this with two-level balancing:

1. **Between regimens** — the caller supplies explicit mixing weights (e.g., 60% base / 25% facts / 15% syllogism). These weights are respected as-is.
2. **Within each regimen** — records are bucketed by their prompt length using quantile cuts (so buckets are always equally populated regardless of the actual length distribution). An equal quota is drawn from each bucket, preventing a regimen's long-prompt records from drowning out its short-prompt examples.

```python
# src/run_ablation_matrix.py — sample_mixture()
#
# Within each regimen, length quantiles are computed on that regimen's own
# records, so bucket boundaries adapt to the distribution rather than
# using a fixed character-count threshold.
# Bucket quota is regimen_quota // n_active_buckets — equal weight per bucket.
```

No special tokens. No separate adapters. No hidden prompt contract. 2× adapter storage cost becomes zero.

**Remaining failure mode if the base adapter already has abbreviation bias baked in:** SPO (offline weighted SFT) cannot correct abbreviation habits — it only reweights gold tokens and never generates bad headers to penalise them. If the starting checkpoint was itself trained on a misbalanced multi-regimen mix, the abbreviation pattern will survive SPO. The fix is to apply stratified sampling at the **base training stage**, not just the fine-tuning stage.

**Short-term mitigation (no retraining):** Constrained decoding — use a `LogitsProcessor` to hard-constrain the first K tokens to the correct header sequence. The model generates the right headers without retraining or special tokens in the prompt.

### 6. Evaluation scripts must use the same prompt format as training

Post-training gate verification showed BASE REASONING producing `NO_HEADER` on every sample despite training successfully. The model scored avg_header_score=0.125 on the initial gate run. Increasing `max_new_tokens` from 192 to 512 made no difference. The actual cause: the verification script passed `record["input_text"]` directly to the tokenizer instead of wrapping it in the chat template via `build_generation_prompt(tokenizer, text)`.

**Why it matters:** An instruct model trained with the Qwen chat format (`<|im_start|>user...<|im_end|><|im_start|>assistant`) expects to see that exact structure at inference time. When given raw text continuation input, the model never enters its "assistant turn" generation mode and produces either nothing or instruction echo rather than the expected structured output. The model was not broken — the evaluation harness was using the wrong prompt format.

**Fix:**
```python
from src.chat_format import build_generation_prompt, strip_response_preamble

# Training used:  build_training_conversation(tokenizer, input_text, output_text)
# Inference must: build_generation_prompt(tokenizer, input_text)
chat_prompt = build_generation_prompt(tokenizer, record["input_text"])
# Then strip <think>...</think> from output before scoring:
output = strip_response_preamble(decoded_output)
score = contract.header_score(output)
```

**After fix:** FACTS avg_header_score=1.000, BASE REASONING avg_header_score=0.917, overall=0.958. Gate: PASS.

**Rule of thumb:** Any script that evaluates a fine-tuned instruct model must mirror the exact tokenizer call chain used during training. If training used `apply_chat_template`, evaluation must too. A mismatch is silent — there is no error; the model simply produces irrelevant output. The symptom (NO_HEADER, generic text, instruction echo) is easily misdiagnosed as a training problem or a token-budget problem when the root cause is purely a format mismatch in the evaluation harness.

### 7. `repetition_penalty` silently corrupts structured headers when the prompt contains the header tokens

The inference examples artifact (`examples/inference_examples.md`) was regenerated after stratified retraining using `repetition_penalty=1.3, no_repeat_ngram_size=4` — the same settings that had prevented line-looping in the old adapter. The new outputs had header scores of 0.00–0.33; the gate reported `FAIL`. The model was not regressing. The generation parameters were destroying the structured output.

`repetition_penalty` applies a multiplicative penalty to **any token that has appeared anywhere in the full input+output sequence so far**, including the prompt. The prompt's instruction list contains `Non-Entailed Premises` and `Entailed Premises`. Under a penalty of 1.3, those tokens become less likely in the output, so the model generates close-but-wrong variants (`Non-EntailedPremise:`, `Non-Entailed Prems:`). `no_repeat_ngram_size=4` has the same effect: it blocks any 4-gram appearing in the prompt from being regenerated, and `Non Entailed Premises` is 3–4 tokens in Qwen's tokenizer.

**Fix:** Remove `repetition_penalty` entirely. Use `no_repeat_ngram_size=6` — large enough to prevent exact-line looping but larger than the header sequences. After this change all 8 inference examples scored 1.00 (gate avg=1.000).

**Diagnostic signature:** a model that passes the gate without these penalties but fails with them, producing headers like `Non-EntailedPremise:` or `Non-Entailed Prems:`, is almost certainly hitting this penalty-vs-prompt-token conflict. Check whether the expected output headers appear verbatim in the prompt before adding any repetition control.

### 8. The eval pipeline must use the exact same generation params as inference — always

The ablation eval (`run_ablation_matrix.py::generate_completion`) had `repetition_penalty=1.3` and `no_repeat_ngram_size=4` baked in even after lesson 7 was documented and the README was updated. The README fix was applied only to `examples/inference_examples.md`; the eval function was never touched. Result: all three ablation experiments returned `avg_quality=0.0` — every single generated output had garbled headers, every single quality check failed. The adapters themselves were fine; the eval was lying.

**Fix:** Treat generation params as a single source of truth. Define them once (e.g. in a config dict or constants module) and import into both the eval script and the inference script. Never copy-paste generation params between files. After changing any generation param in the README or inference path, grep the entire repo for `repetition_penalty`, `no_repeat_ngram_size`, `temperature`, and `do_sample` and audit every occurrence.

**Diagnostic signature:** `avg_quality=0.0` across all experiments when the model is otherwise known-good. Quality collapse on every sample simultaneously points to a systemic eval bug, not model regression.

### 9. SPO reward scoring must be binary for evidence tags — partial credit is a trap

A tempting refinement is to give partial credit when evidence tags are absent (say 0.5 × weight) and penalise mixed tags (say 0.3 × weight) rather than using a hard binary. This sounds principled but introduces a training anti-incentive: the model can now earn 50% of the tag reward by simply omitting tags entirely, which is lower-effort than producing correct ones. Mixed-tag outputs that contain both `observed` and `inferred` in the same parenthetical still earn 30% — another free-point leak.

**Canonical scoring:** `+0.15` if `re.search(r"\b(observed|inferred)\b", output)` matches anywhere in the output, else `0.0`. Binary. Any presence of a valid tag token is rewarded; total absence is not. This preserves the incentive to always include at least one tag while keeping the scorer fast and auditable.

**Rule:** when a reward component is binary in the training data (the gold outputs either have the feature or they don't), keep the reward binary. Introduce graded scoring only when the training data itself exhibits a natural gradient.

### 10. One epoch of SPO is rarely sufficient — watch for quality collapse on hard inputs

After 1 epoch of SPO (436 steps), holdout `avg_correctness=0.958` looks healthy. On easy, well-represented inputs it is. On harder or more abstract quotes the model exhibits: repetition spirals in the Entailed Premises section (15+ near-duplicate triplets), garbled tag variants (`obsined`, `obsed`, `observed/derived`), and bullet-point format leakage (`*   Subject | pred | obj`) from the base model's instruction-tuned prior.

These failure modes do not appear in the holdout metrics because the holdout set is drawn from the same distribution as the training data. They appear on out-of-distribution inputs.

**Fix:** Run SPO for at least 2–3 epochs. Monitor the *variance* of per-step reward across training batches, not just the mean — if variance collapses early, the model has found a local optimum that scores well on template compliance but hasn't generalised. Manually run inference on 3–5 diverse held-out quotes (not from the training distribution) at the end of each epoch before declaring done.

### 11. Confidence scores are only valid when the scorer is frozen and external to the policy

Allowing the model being trained to assign confidence to its own outputs is circular: the model learns to output whatever confidence value maximises reward regardless of actual extraction quality. Early experiments showed the model converging to a bimodal strategy — outputting `confidence=0.4` or `confidence=0.6` almost exclusively — because those values happened to be over-represented in the synthetic training data. The model was optimising the label, not the reasoning.

**Fix:** Never use confidence scores produced by the policy model as a training signal. Confidence is only meaningful when it comes from a frozen external judge — a model whose weights are not updated during the training run. The frozen copy sees the same vocabulary and structure as the policy but has no gradient incentive to inflate its scores.

**Corollary:** Drop `confidence=X` annotations from the output format entirely. The model should learn to produce correct premises and a sound throughline; confidence quantification is a post-hoc evaluation concern, not a generation target.

### 12. A single confidence probe is a point estimate — sample a distribution instead

A single frozen-judge scoring pass returns one number. That number is subject to prompt sensitivity, temperature, and the judge's own uncertainty at the margin. A completion that scores 0.62 on one pass might score 0.48 on another; acting on the point estimate means training on noise.

**Fix:** Generate K confidence ratings per completion from the frozen judge at temperature > 0. The resulting distribution yields two signals that a single pass cannot: (1) `conf_mean` — the judge's central estimate of quality, and (2) `conf_std` — the judge's uncertainty. High mean with low std means the judge consistently agrees this completion is good. High std means the completion sits at the quality margin and the reward signal should be discounted.

The combined signal `conf_mean × (1 − conf_std)` simultaneously rewards quality and penalises uncertain assessments.

### 13. Multi-level sampling multiplies the effective signal per quote

GRPO operates on groups of G completions per quote. If each completion is scored by a single judge pass, you have G reward signals to normalise across. If each completion instead receives K confidence samples, the reward estimate for that completion is the mean of K draws — much lower variance — and the total information gathered per quote scales as G × K.

In practice this means you can reduce G (fewer generations per quote, less VRAM) while keeping or improving signal quality by increasing K (more judge samples per completion, which is cheaper since it requires no gradient computation). The two dimensions are independently controllable.

**Rule:** treat G and K as separate budget dials. G controls coverage across the output space for a given quote; K controls estimation quality per point in that space.

### 14. Empty entailed premises must score zero, not neutral

A completion with no entailed premises is not a borderline case — it is a complete extraction failure. The task is specifically to derive premises from memorable quotes; any non-blank quote has derivable premises by definition. Scoring empty-entailed completions at 0.5 (neutral) allows the model to learn that producing nothing is safe, which is the opposite of the intended incentive.

**Fix:** Return 0.0 immediately when the entailed premises list is empty, regardless of what the non-entailed or conclusion sections contain. Apply the same logic to the conclusion coherence probe: an empty premise list as input makes the coherence question undefined, not neutral.

**Data filter corollary:** Do not filter training quotes based on whether the gold output has entailed premises — that would remove hard quotes and make the training distribution easier than inference. Instead, filter on whether the *input quote* is blank. Any non-blank quote should be trained on; the model must learn to extract premises even from difficult inputs.

### 15. Confirm unproductive quotes are consistently unproductive before pruning

A single all-zero reward group for a quote might be a bad generation day — the model can fail to extract on any given forward pass. Pruning the quote immediately discards potentially useful training signal. Conversely, allowing the model to waste G × K inference passes on quotes that will never produce entailed premises wastes compute each epoch.

**Fix:** Track a consecutive zero-group streak per quote across epochs. Only add the quote to the dead set after it has produced all-zero groups for K consecutive epochs. A single recovery epoch resets the streak. This ensures pruning is consensus-based (multiple independent generation attempts) rather than single-sample noise.

### Takeaway

For any RL-from-feedback training loop: (1) generation controls must prevent degenerate outputs before rewards are ever computed, (2) reward functions must explicitly penalise known failure modes rather than only rewarding the ideal case, (3) reward signals computed from gold data are always suspect — check that the distribution of rewards across your training set actually varies before assuming SPO is doing anything useful, (4) offline preference optimisation cannot correct a habit the model has never been penalised for producing — if the failure mode is a specific generated token sequence, only online generation-and-penalise RL can reliably fix it, (5) for small models, multi-regimen training on semantically overlapping formats requires stratified sampling across (regimen × prompt-length) strata — without it, whichever regimen dominates the data mix will corrupt the shared header vocabulary for the other regimens, (6) post-training evaluation scripts must mirror the exact prompt-format pipeline used during training — a chat-template mismatch produces NO_HEADER silently, (7) `repetition_penalty` corrupts structured headers whenever those headers appear verbatim in the prompt instruction list — use `no_repeat_ngram_size=6` instead, (8) eval pipeline generation params must be identical to inference params — a README fix that never propagates to the eval script produces `avg_quality=0.0` silently, (9) one epoch of SPO is not enough for quality generalisation — validate on out-of-distribution inputs at each epoch boundary, not just on holdout metrics, (10) confidence scores must come from a frozen external judge — never from the policy being trained, (11) sample a distribution of K confidence scores per completion rather than a single probe — the mean and std together provide a richer signal than any point estimate, (12) G (completions per quote) and K (confidence samples per completion) are independent budget dials — reduce G and increase K to improve signal quality at fixed compute.

---

**Status:** Production-ready  
**Repository:** https://github.com/thistleknot/spo-reasoning-training-regimen
