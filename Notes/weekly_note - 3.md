# Week 3 — Weekly Note

## Overview

The primary work this week focused on setting up the data that will be used for further analysis, as well as establishing the basic pipeline for analysis to be built upon in subsequent weeks.

The dataset utilised is a processed dataset derived from the [TruthfulQA dataset](https://huggingface.co/datasets/domenicrosati/TruthfulQA), originally designed to evaluate whether a language model can produce reliable, truthful answers in a human-like manner. The processed dataset is set up to study how the model functions and understand its internal activations for mechanistic interpretability studies, particularly in probing and identifying internal representations that distinguish truthful self-correction from hallucination.

---

## I) Rendering JSON

This step sets up the relevant data structure and file. The script loads the data from HuggingFace and packages the dataset per data unit as follows:

- **ID:** For the sake of tracking individual samples.
- **Question:** The prompt or original question posed to the LLM.
- **prompt_A:** The self-corrected (truthful) response of the LLM, provided alongside the question.
- **prompt_B:** The hallucinated (incorrect) response of the LLM, provided alongside the question.
- **source:** As per the TruthfulQA dataset.

This generates the required `contrastive_pairs.jsonl` and `contrastive_pairs.json` files.

Two important notes:
1. Only data belonging to the **Misconceptions** category has been considered.
2. A given question may have multiple hallucination prompts associated with it. The dataset is structured such that each hallucination prompt occupies its own row, paired with the appropriate question and self-corrected prompt.

---

## II) Caching

With the processed data in place, the LLMs are loaded from HuggingFace to verify data compatibility and confirm the relevant framework is set up correctly. The process is as follows:

1. Load the model from HuggingFace and set a file directory for it.
2. Load `contrastive_pairs.json` to obtain the data.
3. Pass the `prompt_A` and `prompt_B` prompts to the LLM and record activations using `run_with_cache` from TransformerLens.
4. Save the Key-Value Cache Dictionary to a `.pt` file. During this step, log time, memory usage, and other relevant details such as sub-module names for verification and future analysis.

**Note:** A `.pt` file has been used in place of a `.json` file due to file size constraints and the limitations of the JSON format for this type of data. This change is not expected to significantly affect the overall pipeline going forward.

File included:
- contrastive_pairs.json
- contrastive_pairs.jsonl
- dataset_card.md
- Caching using TransformerLens.py
- Rendering JSON.py