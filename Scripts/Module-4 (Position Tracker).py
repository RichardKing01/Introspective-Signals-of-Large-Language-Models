###### IMPORTS ######
from transformer_lens.model_bridge import TransformerBridge
from datasets import load_dataset

import os
import time
import json
import re
from tqdm import tqdm
import torch
import gc
from scipy.stats import spearmanr

import pandas as pd
import numpy as np

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import random

###### Config ######
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
os.environ["HF_TOKEN"] = ## Fill
os.environ["GROQ_API_KEY"] = ## Fill
Lambda = 0.3
path = 'Log'
seed = 67

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(67)

###### DATA ######
print('Loading Data')
with open('contrastive_pairs.json', 'r') as f:
    data = json.load(f)

with open('contrastive_pairs_held_out.json', 'r') as f:
    held_out_data = json.load(f)

print('Data Loaded')

FILLER_POOL = [
    "The library was quiet on Tuesday afternoon.",
    "She adjusted the thermostat before leaving for work.",
    "The recipe called for two cups of flour and a pinch of salt.",
    "Traffic was lighter than usual on the highway this morning.",
    "He spent the weekend reorganizing his bookshelf.",
    "The garden needed watering after the dry week.",
    "They watched a documentary about deep sea creatures.",
    "The meeting was rescheduled to next Thursday.",
    "A light breeze moved through the open window.",
    "The train arrived exactly on time for once.",
]


###### POSITION TEST ######
def build_padded_prompt(target_string, n_filler_sentences):
    if n_filler_sentences == 0:
        return target_string
    filler = " ".join(FILLER_POOL[i % len(FILLER_POOL)] for i in range(n_filler_sentences))
    return filler + " " + target_string

def position_confound_test(Model, v_introspect, layer_name, target_string, filler_counts = [0, 2, 4, 8, 16, 32, 64], tpos = -1):
    positions, scores = [], []

    for n_filler in filler_counts:
        prompt = build_padded_prompt(target_string, n_filler)
        tokens = Model.to_tokens(prompt)
        seq_len = tokens.shape[1]
        abs_position = seq_len + tpos

        with torch.no_grad():
            _, cache = Model.run_with_cache(tokens, names_filter=layer_name)

        activation = cache[layer_name][0, tpos, :].cpu().numpy()
        score = float(np.dot(activation, v_introspect))

        positions.append(abs_position)
        scores.append(score)

    corr, p_value = spearmanr(positions, scores)
    return {'positions': positions, 'scores': scores, 'spearman_corr': round(float(corr), 4), 'p_value': round(float(p_value), 5)}

###### RUNNING ######
Model_List = ['gpt2', 'EleutherAI/pythia-410m', 'facebook/opt-1.3b']

for model_name in Model_List:
    print(f'Current model to load: {model_name}')
    Model = TransformerBridge.boot_transformers(model_name, device = device)
    safe_name = model_name.replace('/', '--')

    print(f'{model_name} loaded')
    safe_path = os.path.join(path, safe_name)
    os.makedirs(safe_path, exist_ok = True)

    LR_Dir = os.path.join(path, safe_name)
    LR_Path = None
    Layer_Name = None

    for x in os.listdir(LR_Dir):
        if x.endswith('.pkl'):
            Layer_Name = x[:-4]
            LR_Path = os.path.join(LR_Dir, x)
            break

    if LR_Path is None:
        raise FileNotFoundError(f"No .pkl probe found in {LR_Dir}")
    LR = joblib.load(LR_Path)

    vect = LR.coef_[0]
    vect = vect / np.linalg.norm(vect)

    target_strings = [item['prompt_A'] for item in data[:10]]

    all_corrs = []
    for target in target_strings:
        result = position_confound_test(Model, vect, Layer_Name, target)
        all_corrs.append(result['spearman_corr'])

    print(f"Mean correlation across examples: {np.mean(all_corrs):.4f}")
    print(f"Individual correlations: {all_corrs}")

    Model.to("cpu")
    del Model
    gc.collect()
    torch.cuda.empty_cache()

    print("------------------------------------------------\n")