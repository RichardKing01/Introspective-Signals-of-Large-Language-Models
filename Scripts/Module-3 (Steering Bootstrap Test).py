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

###### STEERING TEST ######
def Steering_Test(Model, LR, layer_name, safe_name, data, alphas = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.25, 1.5, 1.75, 2], title = None):

    data = data[:1000]

    def CreateVector(alpha: float, v):
        v = torch.tensor(v, dtype=torch.float32, device=device)

        def Hook_Func(activation, hook):
            norm = activation.norm(dim=-1, keepdim=True)
            return activation + alpha * norm * v
        return Hook_Func

    def Inject_And_Run(Model, Prompt, Complete_Prompt, Hook_Name, Hook_Func):
        Tokenized = Model.to_tokens(Prompt)
        Complete_Tokenized = Model.to_tokens(Complete_Prompt)

        n_prompt = Tokenized.shape[1]

        with torch.no_grad():
            logits = Model.run_with_hooks(Complete_Tokenized, fwd_hooks=[(Hook_Name, Hook_Func)])
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        total_lp = 0.0
        n_continuation = Complete_Tokenized.shape[1] - n_prompt
        for i in range(n_prompt, Complete_Tokenized.shape[1]):
            target_id = Complete_Tokenized[0, i]
            total_lp += log_probs[0, i - 1, target_id].item()

        return total_lp / n_continuation

    vect = LR.coef_[0]
    vect = vect / np.linalg.norm(vect)

    results = {}
    for alpha in alphas:
        Hook_Func = CreateVector(alpha, vect)
        Hook_Name = layer_name

        hits = []
        for item in tqdm(data, desc='Processed Files for Steering: '):
            prompt = f"{item['question']}\nThe answer is"
            A_complete = prompt + " " + item['correct_answer']
            B_complete = prompt + " " + item['incorrect_answer']

            LP_A = Inject_And_Run(Model, prompt, A_complete, Hook_Name, Hook_Func)
            LP_B = Inject_And_Run(Model, prompt, B_complete, Hook_Name, Hook_Func)

            hits.append(1 if LP_A > LP_B else 0)

        accuracy = sum(hits) / len(hits)
        results[alpha] = {'accuracy': accuracy, 'hits': hits}   # NEW: hits list saved alongside accuracy
        print(f'Alpha value: {alpha} | Accuracy: {accuracy}')

    save_path = os.path.join(path, safe_name, f'{safe_name}_{title or "alpha_results"}.json')
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved {save_path}')
    return results

###### Bootstrapping Difference Test ######
def paired_bootstrap_diff(hits_a, hits_b, boot = 10000, seed = seed):

    rand = np.random.default_rng(seed)
    hits_a = np.array(hits_a)
    hits_b = np.array(hits_b)
    n = len(hits_a)

    observed_diff = hits_a.mean() - hits_b.mean()

    boot_diffs = []
    for _ in range(boot):
        index = rand.integers(0, n, size=n)
        boot_diffs.append(hits_a[index].mean() - hits_b[index].mean())

    boot_diffs = np.array(boot_diffs)
    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
    significant = not (ci_low <= 0 <= ci_high)   # 0 outside the interval = real difference

    return {'observed_diff': round(float(observed_diff), 4), 'ci_95': [round(float(ci_low), 4), round(float(ci_high), 4)], 'significant': bool(significant)}

def CreateVector(alpha: float, v):
    v = torch.tensor(v, dtype=torch.float32, device=device)

    def Hook_Func(activation, hook):
        norm = activation.norm(dim=-1, keepdim=True)
        return activation + alpha * norm * v
    return Hook_Func

###### DATA ######
print('Loading Data')
with open('contrastive_pairs.json', 'r') as f:
    data = json.load(f)

with open('contrastive_pairs_held_out.json', 'r') as f:
    held_out_data = json.load(f)
print('Data Loaded')

###### RUNNING ######
Model_List = ['EleutherAI/pythia-1b', 'EleutherAI/pythia-410m', 'EleutherAI/pythia-2.8b']
alphas=[-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.25, 1.5, 1.75, 2]
for model_name in Model_List:
    print(f'Current model to load: {model_name}')
    Model = TransformerBridge.boot_transformers(model_name, device=device)
    safe_name = model_name.replace('/', '--')
    print(f'{model_name} loaded')
    safe_path = os.path.join(path, safe_name)
    os.makedirs(safe_path, exist_ok=True)

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

    print("HELD OUT")
    result_1 = Steering_Test(Model, LR, Layer_Name, safe_name, held_out_data,
                          alphas=[-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.25, 1.5, 1.75, 2],
                          title='Held_Out_finegrained')

    for x, y in zip(alphas[:-1], alphas[1:]):
        comparison = paired_bootstrap_diff(result_1[x]['hits'], result_1[y]['hits'])
        print(f'({x},{y}): {comparison}')

    comparison = paired_bootstrap_diff(result_1[-2]['hits'], result_1[-1]['hits'])
    print(f'({-2},{-1}): {comparison}')

    comparison = paired_bootstrap_diff(result_1[-2]['hits'], result_1[-1]['hits'])
    print(f'({1},{2}): {comparison}')


    print('\n\n TRUTHFULQA')
    result_1 = Steering_Test(Model, LR, Layer_Name, safe_name, data,
                          alphas=[-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.25, 1.5, 1.75, 2],
                          title='TruthfulQA')

    for x, y in zip(alphas[:-1], alphas[1:]):
        comparison = paired_bootstrap_diff(result_1[x]['hits'], result_1[y]['hits'])
        print(f'({x},{y}): {comparison}')

    comparison = paired_bootstrap_diff(result_1[-2]['hits'], result_1[-1]['hits'])
    print(f'({-2},{-1}): {comparison}')

    comparison = paired_bootstrap_diff(result_1[-2]['hits'], result_1[-1]['hits'])
    print(f'({1},{2}): {comparison}')

    Model.to("cpu")
    del Model
    gc.collect()
    torch.cuda.empty_cache()

    print("------------------------------------------------\n")