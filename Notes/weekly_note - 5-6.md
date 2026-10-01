# Week 5-6: Optimal Prompt Search via Genetic Algorithm

## Overview

This is the third step in a three-part pipeline:

1. **Data Pipeline** — Constructed contrastive prompt pairs from TruthfulQA (Misconceptions category), stored in `contrastive_pairs.json`. Each pair contains a prompt that elicits epistemic self-correction (A) and one that elicits hallucination-leaning completion (B).

2. **Probe Training** — Cached residual stream activations from GPT-2 Small and trained a per-layer logistic regression discriminator to separate introspective from normal generation. The best-layer probe weight vector `v_introspect` was extracted and unit-normed.

3. **Prompt Search (this step)** — Given `v_introspect`, we use a Genetic Algorithm to search over English prompt prefixes and find sentences that maximally push GPT-2 Small into the introspective internal state identified by the probe.

---

## Functions

**`set_seed(seed)`**
Sets the random seed across Python, NumPy, and PyTorch to ensure reproducibility. Called once at the start of the GA.

**`PPL(model, string)`**
Computes the perplexity of a string under GPT-2 Small. Perplexity measures how fluent and natural a sentence is — lower means more fluent, higher means gibberish. Used as a penalty term in the fitness function to stop the GA from drifting toward nonsensical outputs.

**`Evaluate(model, string, v, layer_name)`**
The core scoring function. Tokenizes the input string, runs it through GPT-2 Small, extracts the residual stream activation at the last token position from the specified layer, and computes the dot product with `v_introspect`. Returns a scalar `I_score` — how strongly the prefix activates the introspective direction.

**`Mutate(string)`**
Takes a sentence and returns a slightly modified version of it. Uses `Qwen2.5-0.5B-Instruct` as the rewrite model — Qwen is used here because it is lightweight, runs on CPU, and is instruction-following. Base models like GPT-2 Small cannot reliably follow rewrite instructions and would simply continue the text rather than paraphrase it.

**`Generate_Sentences(count)`**
Generates an initial population of `count` seed sentences from scratch. Also uses Qwen, prompted to produce short sentences that encourage careful, reflective thinking before answering a question. These seeds form the starting population of the GA before any selection pressure has been applied.

**`GA(Model, vector, layer_name, ...)`**
The Genetic Algorithm. Maintains a population of candidate prefix sentences and evolves them over multiple generations toward higher fitness. Details below.

---

## GA Flow

```
1. Initialize population
      └── if no data provided: Generate_Sentences(population_size) via Qwen
      └── if data provided: use supplied sentences, mutate to fill up if needed

2. For each generation:
      a. Score every sentence: J(P) = I_score(P) - lambda * PPL(P)
      b. Sort by fitness descending
      c. Enforce elitism: if best this generation < best ever, inject best ever back
      d. Kill bottom death_rate fraction
      e. Mutate survivors to fill population back up
      f. Log best fitness, mean fitness, best prompt

3. Return log and final population
```

Elitism guarantees that fitness is monotone non-decreasing across generations — the best prompt found is never lost.

---

## Fitness Function

```
J(P) = I_score(P) - lambda * PPL(P)
```

`I_score` pulls the GA toward sentences that activate the introspective probe direction. `PPL` penalizes sentences that are unnatural or incoherent. `lambda` controls the tradeoff between the two.

---

## Lambda Grid

To find the optimal `lambda`, we run the full GA once per value in `[0.0, 0.1, 0.5, 1.0, 5.0]`. For each run we record the `i_score` and `ppl` of the best prompt found. Results are saved to `{model_name}_lambda_grid.json` and plotted as a Pareto frontier — `ppl` on the x-axis, `i_score` on the y-axis, each point labelled with its lambda value. The optimal lambda sits at the elbow: where i_score is high but the sentence is still fluent.

The plot is saved to `{model_name}_pareto.png`.

---

## Notes

- Only tested on `gpt2` (GPT-2 Small, 12 layers, d_model=768).
- The probe vector is the unit-normed `coef_[0]` from the best-layer logistic regression trained in Week 4.
- Qwen is used exclusively for text generation (mutation and seeding). GPT-2 Small is used exclusively for scoring (I_score and PPL). The two models never interact.
