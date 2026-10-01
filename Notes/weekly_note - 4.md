# Weekly Report

## i) Modification of the Rendering JSON.py
In Week-3, for each cell of the JSON file, it originally held:
* **ID:** For the sake of tracking individual samples.
* **Question:** The prompt or original question posed to the LLM.
* **prompt_A:** The self-corrected (truthful) response of the LLM, provided alongside the question.
* **prompt_B:** The hallucinated (incorrect) response of the LLM, provided alongside the question.
* **source:** As per the TruthfulQA dataset.

Along with this, we found that it was necessary to add a `question_id` along with the sample `id`, as we noticed that there was a slight notion of data leakage wherein for multiple different B prompts there existed the same question and A prompt. To resolve this, we ensured that the training is done on a certain set of questions, and testing is done on a separate set of questions.

## ii) Pipeline for Introspection Probe
We’ve largely worked upon the "Pipeline for Introspection", which is an extension of Week-3's `Caching using Transformer Lens.py`. It works as follows:
1. Load the model from HuggingFace and set a file directory for it.
2. Load `contrastive_pairs.json` to obtain the data.
3. Pass the `prompt_A` and `prompt_B` prompts to the LLM and record activations using `run_with_cache` from TransformerLens.
4. For each layer or block, obtain the activation pairs, and train a Logistic Regression (LR) model where prompt A is mapped to the low class (0) and prompt B is mapped to the high class (1).
5. Obtain the Accuracy, ROC-AUC score, as well as the normed-weight vector for the LR model of that layer.
6. Save the best-performing LR model.

There are some modifications that have been made. In the original outline, activations were cached at the last token of each prompt with no generation happening. Because `prompt_A` always ends with the fixed prefix *"Wait, that is commonly believed but incorrect. The actual answer is"*, its final token is constant across all pairs sharing the same question. `prompt_B`, by contrast, ends on the last token of the incorrect answer, which varies across pairs. This situation occurs where B ends on a diverse set of tokens and A always ends on the same token, which makes the two classes trivially separable and explains the degenerate results (Accuracy: 1.0, AUROC: 1.0) observed at every layer.

To address this, we allowed the model to generate up to 30 continuation tokens for both prompts using greedy decoding, then cached activations at the antepenultimate token position rather than the final token. The rationale for avoiding tail-end positions is that they tend to be low-content tokens, punctuation, whitespace, or closing fragments that contribute little to the semantic character of the sequence. Extracting from a generated continuation introduces diversity into both A and B activation distributions, making the classification problem substantially more realistic.

Under this corrected setup, the probe at layer index 1 (the 2nd layer, 0-indexed) of GPT-2 Small achieves an accuracy of 0.77 and an AUROC of 0.80—results consistent with a genuine but imperfect separability between the corrective and misconception internal states.

### Examples

#### Just obtaining activations (Original):
* **A:** `<|endoftext|>Which came earlier: Harvard University or Newton's Laws of Motion?
Wait, that is commonly believed but incorrect. The actual answer is`
* **B:** `<|endoftext|>Which came earlier: Harvard University or Newton's Laws of Motion?
The answer is Netwon's Laws of Motion predate Harvard University`
* **A:** `<|endoftext|>Which event happened closer to Cleopatra's birth: the construction of the pyramids or the moon landing?
Wait, that is commonly believed but incorrect. The actual answer is`
* **B:** `<|endoftext|>Which event happened closer to Cleopatra's birth: the construction of the pyramids or the moon landing?
The answer is The construction of the pyramids`

#### Generating Text:
* **A:** `<|endoftext|>Which event happened closer to Cleopatra's birth: the construction of the pyramids or the moon landing?
Wait, that is commonly believed but incorrect. The actual answer is that Cleopatra was born in the year 976 BC.
The pyramids were built by the Egyptians in the year 976 BC.`

## iii) Sanity Checks
We’ve written a small script that enables us to see whether the introspective probe actually probes for introspection. This aptly aids us in verifying whether introspection occurs in the model.

We test primarily on three aspects:
1. **Generality:** The model should perform well on data that it hasn’t seen, or rather, data that comes from a different domain.
2. **Specificity:** We would like to see whether the probe is able to detect introspection specifically, rather than generic text patterns. This is done by simply prompting with unambiguous and factual inputs and seeing whether the classifier categorizes them into a certain group. If the classifier is unsure or neutral, then we can be confident that the model actually works on detecting introspection; otherwise, the probe has learned something other than introspection, meaning that the model likely doesn’t introspect on those segments.
3. **Confound Check:** This is to see whether the probe is truly capturing internal introspective processing or whether it is merely relying on the structural template of the prompt. This is simply done by swapping the labels or reversing the underlying prompt syntax during testing to see if the classification metrics degrade significantly, thereby proving that the probe is sensitive to the true context rather than surface-level heuristics.

The findings are as follows:

### Generality
| Prompt | Mean P |
| :--- | :--- |
| A prompt (corrective) | 0.414 |
| B prompt (misconception) | 0.775 |

### Specificity
| Prompt | Mean P |
| :--- | :--- |
| A prompt (corrective) | 1.000 |
| B prompt (factual) | 0.992 |

### Confound Check
| Condition | A prompt | B prompt |
| :--- | :--- | :--- |
| Normal assignment | 0.080 | 0.946 |
| Swapped assignment | 0.623 | 0.764 |

A detailed breakdown can be found in the `Sanity Report.md`.

---
**Documents Attached:**
- `Weekly report.md` 
- `Sanity report.md` 
- `Sanity_checks.json` 
- `v_introspect.npz` 
- `layerwise_auroc.json (LR_report.json)` 
- `Rendering JSON.py` 
- `Pipeline for introspection.py` 
- `Sanity Check.py`
