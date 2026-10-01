# Weekly Report: Week-7 & 8

## i) Module-1: Pipeline from Data Framing, Model Probing, Sanity-Check, & Steering
We've consolidated the earlier notebook-based workflow into a single, modular `.py` script. This module (Module-1) covers the full path from raw data to a validated steering vector:
1. **Data Framing:** Loads `contrastive_pairs.json`, applies the train/test split on `question_id` (to avoid the leakage identified in earlier weeks), and constructs the prompt_A / prompt_B pairs.
2. **Model Probing:** Loads the target model via TransformerLens, generates continuations, caches activations at the antepenultimate token position, and trains the per-layer Logistic Regression probes.
3. **Sanity-Check:** Runs the Generality, Specificity, and Confound checks (see modification below) against the trained probes.
4. **Steering:** Uses the normed probe weight vector as a steering direction, applying norm-scaled alpha sweeps to test causal effect on generation.

Having this as a single script (rather than scattered notebooks) makes it straightforward to run the same pipeline end-to-end across different models and architecture families.

## ii) Module-2: Genetic Algorithm using Simple LLM for Mutation & Crossover
This module implements the GA-based prefix evolution using a small, locally-run LLM to perform the mutation and crossover steps on candidate prefixes. Fitness is scored against the introspection probe from Module-1.

We found that the GA's performance is bottlenecked by the choice of Mutation/Crossover model — a weaker LLM produces less diverse and lower-quality candidate prefixes, which in turn slows convergence and caps the fitness ceiling the GA can reach. This is detailed further in the accompanying document, along with example generations that illustrate the difference in candidate quality.

## iii) Module-2.1: Genetic Algorithm using Groq-Cloud API (GPT-OSS-120B) for Mutation & Crossover
To address the bottleneck identified in Module-2, this variant swaps the local mutation/crossover model for GPT-OSS-120B served via the Groq Cloud API. This gives access to a substantially stronger model for generating candidate prefixes, without the latency cost of running a large model locally.
We still find that the GA's performance is quite tied to the Mutation/Crossover model and lesser upon the Logistic Regression Model Probe.

## iv) Module-3: Deeper Steering Test, with Bootstrapping for Statistical Measure
This module extends the causal steering test from Module-1 by adding a bootstrapping procedure over the steered generations, giving a proper statistical measure (rather than a point estimate) of the steering effect at each alpha value. The full analysis, along with the code, is detailed in the accompanying document.

## v) Module-4: Position-Tracker
This module tests the extent to which various models' probes depend on positional encoding rather than genuine introspective content, by tracking probe performance as a function of token position across architectures. This is the diagnostic that feeds into the RoPE-related architectural comparison across model families.

## Sanity-Check Modification (Confound)
The Confound check has been minorly modified. Previously, we checked for degradation in classification metrics under swapped assignment. We now additionally require that, in **both** the normal and swapped conditions, Prompt-B specifically receives a low score — i.e., the check now explicitly verifies that Prompt-B is *not* spuriously scored high under either labeling, rather than only looking at aggregate metric degradation.

---
**Documents Attached:**
- `Weekly-Note and Confusion`
- `Module-1.py`
- `Module-2 (GA-Groq).py`
- `Module-2 (GA-Simple).py`
- `Module-3 (Steering Bootstrap Test).py`
- `Module-4 (Position-Tracker).py`
- `Full Project Report.pdf`
