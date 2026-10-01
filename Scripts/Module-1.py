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

###### FORMATTING ######
def Formatting_Dataset(dataset, file_name):
    to_save = []
    identity = 0
    for x in tqdm(range(len(dataset)), desc="Length of TruthfulQA"):
        series = dataset.iloc[x]

        question = series['question']
        incorrect_list = series['incorrect_answers']

        for b in incorrect_list:
            entry = {
                "id": identity,
                "question_id": x,
                "question": question,

                "prompt_A": f"{question}\nWait, that is commonly believed but incorrect. The actual answer is",
                "prompt_B": f"{question}\nThe answer is {b}",

                "correct_answer": series['best_answer'],
                "incorrect_answer": b,

                "category": series['category'],
                "source": series['source']
            }
            to_save.append(entry)
            identity += 1

    ###### Saving ######
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(to_save, f, ensure_ascii=False, indent=4)

###### CACHING FUNCTION ######
def Running(data, Model, path):

    smoke_test_log = []
    for i in tqdm(range(len(data)), desc='No. of Units Cached: '):

        series = data[i]
        data_path = os.path.join(path, str(series['id']))
        os.makedirs(data_path, exist_ok=True)
        A = series['prompt_A']
        B = series['prompt_B']
        Token_A = Model.to_tokens(A)
        Token_B = Model.to_tokens(B)

        # A
        start_A = time.time()
        generated_A = Model.generate(Token_A, max_new_tokens=30, do_sample=False, verbose=False)
        _, Cache = Model.run_with_cache(generated_A)
        cache_dict = {k: v.cpu() for k, v in Cache.items() if 'hook_resid_post' in str(k)}
        data_path_A = os.path.join(data_path, 'A.pt')
        torch.save(cache_dict, data_path_A)

        # B
        start_B = time.time()
        generated_B = Model.generate(Token_B, max_new_tokens=30, do_sample=False, verbose=False)
        _, Cache = Model.run_with_cache(generated_B)
        cache_dict = {k: v.cpu() for k, v in Cache.items() if 'hook_resid_post' in str(k)}
        data_path_B = os.path.join(data_path, 'B.pt')
        torch.save(cache_dict, data_path_B)

        smoke_test_log.append({
            'id': series['id'],
            'A': {'time_sec': round(time.time() - start_A, 3), 'total_hooks': len(cache_dict), 'hook_names_sample': list(cache_dict.keys())[:5]},
            'B': {'time_sec': round(time.time() - start_B, 3), 'total_hooks': len(cache_dict), 'hook_names_sample': list(cache_dict.keys())[:5]}
        })

    with open(os.path.join(path, 'cache_smoke_test.json'), 'w') as f:
        json.dump(smoke_test_log, f, indent=2)

    return list(cache_dict.keys())


##### FINDING THE LR PROBE #####
def load_by_layer(model_path, layer_name):

    unique_qids = list(set(d['question_id'] for d in data))
    train_qids, test_qids = train_test_split(unique_qids, test_size=0.2, random_state = 67)
    train_qids, test_qids = set(train_qids), set(test_qids)

    train_idx, test_idx = [], []
    for d in data:
        if d['question_id'] in train_qids:
            train_idx.append(d['id'])
        else:
            test_idx.append(d['id'])

    def load(List):
        A = []; B = []
        for pair_id in List:
            query_path = os.path.join(model_path, str(pair_id))
            A.append(torch.load(os.path.join(query_path, 'A.pt'))[layer_name][0, -3, :].numpy())
            B.append(torch.load(os.path.join(query_path, 'B.pt'))[layer_name][0, -3, :].numpy())
        return A, B

    train_A, train_B = load(train_idx)
    test_A,  test_B  = load(test_idx)
    return train_A, train_B, test_A, test_B


def Find_Best_Layer(Layer_Dict):
    for safe_path, Layers in Layer_Dict.items():
        print(f'Model: {safe_path}')
        DF = []
        best_roc = -1
        best_model = None
        best_layer = None

        for layer_name in Layers:
            print(f'\t{layer_name}')
            train_A, train_B, test_A, test_B = load_by_layer(safe_path, layer_name)

            X_train = np.stack(train_A + train_B)
            y_train = np.array([1]*len(train_A) + [0]*len(train_B))

            X_test = np.stack(test_A + test_B)
            y_test = np.array([1]*len(test_A) + [0]*len(test_B))

            LR = LogisticRegression(max_iter=1000)
            LR.fit(X_train, y_train)
            roc = roc_auc_score(y_test, LR.predict_proba(X_test)[:, 1])
            norm_vector = (LR.coef_[0] / np.linalg.norm(LR.coef_[0])).tolist()

            DF.append({'layer_name': layer_name, 'accuracy': LR.score(X_test, y_test), 'roc_auc': roc, 'unit_norm_probe': norm_vector})

            if roc >= best_roc:
                best_roc = roc
                best_model = LR
                best_layer = layer_name

        DF = pd.DataFrame(DF)
        print(f'Best layer: {best_layer} with ROC-AUC: {best_roc:.4f}')
        report_path = os.path.join(safe_path, 'LR_report.json')

        DF.to_json(report_path, orient='records', lines=True)

        joblib.dump(best_model, os.path.join(safe_path, f'{best_layer}.pkl'))

##### SANITY CHECK #####
def Sanity_Check(Model, LR, layer_name, model_path):
    data = load_dataset("truthfulqa/truthful_qa", "generation")
    data = pd.DataFrame(data['validation'])
    data = data[~data['category'].isin(['Misconceptions', 'Misconceptions: Topical'])].head(50)

    gen_pairs = []
    for _, row in tqdm(data.iterrows()):
        if not row['incorrect_answers']:
            continue
        a = f"{row['question']}\nWait, that is commonly believed but incorrect. The actual answer is"
        b = f"{row['question']}\nThe answer is {row['incorrect_answers'][0]}"
        gen_pairs.append((a, b))

    def get_activation(Input, layer_name, tpos=-3):
        Tokens = Model.to_tokens(Input)
        Generated = Model.generate(Tokens, max_new_tokens=30, do_sample=False, verbose=False)
        _, Cache = Model.run_with_cache(Generated)
        return Cache[layer_name][0, tpos, :].cpu().numpy()

    def evaluate(pairs, layer_name):
        probs_A, probs_B = [], []
        for a, b in tqdm(pairs):
            act_A = get_activation(a, layer_name).reshape(1, -1)
            act_B = get_activation(b, layer_name).reshape(1, -1)
            probs_A.append(LR.predict_proba(act_A)[0, 1])
            probs_B.append(LR.predict_proba(act_B)[0, 1])
        mean_A = round(float(np.mean(probs_A)), 3)
        mean_B = round(float(np.mean(probs_B)), 3)
        print(f" A prompt: {mean_A:.3f}")
        print(f" B prompt: {mean_B:.3f}")
        return mean_A, mean_B

    results = {}

    ###### Generality ######
    print('Testing Generality')
    mean_A, mean_B = evaluate(gen_pairs, layer_name)
    results['generalization'] = {'mean_prob_A': mean_A, 'mean_prob_B': mean_B}

    ###### Specificity ######
    easy_pairs = [
        (
            "What is the capital of France?\nWait, that is commonly believed but incorrect. The actual answer is",
            "What is the capital of France?\nThe answer is Paris"
        ),
        (
            "How many days are in a week?\nWait, that is commonly believed but incorrect. The actual answer is",
            "How many days are in a week?\nThe answer is Seven"
        ),
    ]
    print('Testing Specificity')
    mean_A, mean_B = evaluate(easy_pairs, layer_name)
    results['specificity'] = {'mean_prob_A': mean_A, 'mean_prob_B': mean_B}

    ###### Confound ######
    dataset = load_dataset("truthfulqa/truthful_qa", "generation")
    df_confound = pd.DataFrame(dataset['validation'])
    df_confound = df_confound[ (df_confound['category'] == 'Misconceptions') | (df_confound['category'] == 'Misconceptions: Topical')].head(30)

    questions, wrongs = [], []
    for _, row in df_confound.iterrows():
        if not row['incorrect_answers']:
            continue
        questions.append(row['question'])
        wrongs.append(row['incorrect_answers'][0])

    normal_pairs = [
        (f"{q}\nWait, that is commonly believed but incorrect. The actual answer is",
         f"{q}\nThe answer is {w}")
        for q, w in zip(questions, wrongs)
    ]

    shuffled_wrongs = wrongs.copy()
    while True:
        random.shuffle(shuffled_wrongs)
        if all(w1 != w2 for w1, w2 in zip(wrongs, shuffled_wrongs)):
            break

    mismatched_pairs = [
        (f"{q}\nWait, that is commonly believed but incorrect. The actual answer is",
         f"{q}\nThe answer is {w}")
        for q, w in zip(questions, shuffled_wrongs)
    ]

    print('Checking for Confound (mismatched wrong-answer content)')
    mean_A_normal, mean_B_normal = evaluate(normal_pairs, layer_name)
    mean_A_mismatch, mean_B_mismatch = evaluate(mismatched_pairs, layer_name)
    results['confound'] = {
        'normal': {'mean_prob_A': mean_A_normal,   'mean_prob_B': mean_B_normal},
        'mismatched': {'mean_prob_A': mean_A_mismatch, 'mean_prob_B': mean_B_mismatch}
    }

    ###### Save ######
    with open(os.path.join(model_path, 'sanity_checks.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('Saved sanity_checks.json')

###### STEERING TEST######
def Steering_Test(Model, LR, layer_name, safe_name, data, alphas=[-2, -1, 0, 1, 2], title = None):

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
        print(f' Alpha Value: {alpha}')
        Hook_Func = CreateVector(alpha, vect)
        Hook_Name = layer_name

        score = 0
        for item in tqdm(data, desc = 'Processed Files for Steering: '):
            prompt = f"{item['question']}\nThe answer is"
            A_complete = prompt + " " + item['correct_answer']
            B_complete = prompt + " " + item['incorrect_answer']

            LP_A = Inject_And_Run(Model, prompt, A_complete, Hook_Name, Hook_Func)
            LP_B = Inject_And_Run(Model, prompt, B_complete, Hook_Name, Hook_Func)

            if LP_A > LP_B:
                score += 1

        accuracy = score / len(data)
        print(f'Alpha value: {alpha} | Accuracy: {accuracy}')
        results[alpha] = accuracy

        print('\n')

    ###### Save ######
    if title is None:
        save_path = os.path.join(path, safe_name, f'{safe_name}_alpha_results.json')
    else:
        save_path = os.path.join(path, safe_name, f'{safe_name}_{title}.json')

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved {save_path}')

    print(results)

###### DELETE CACHE ######
def Remove_Cached(safe_path):
    List = os.listdir(safe_path)

    print(f'  Current Size of File: {len(List)}')
    for x in tqdm(List, desc = "Files Processed: "):
        if x.isnumeric():
            rm_path = os.path.join(safe_path, x)

            for item in os.listdir(rm_path):
                os.remove(os.path.join(rm_path, item))
            os.rmdir(rm_path)
    print(f'  Size of Processed File: {len(os.listdir(safe_path))}')
    del List




###### CONGIURATION ######
print(f"Device: {device} | seed: {seed}")


###### DATA ######
dataset = load_dataset("truthfulqa/truthful_qa", "generation")
dataset = pd.DataFrame(dataset['validation'])
dataset_1 = dataset[(dataset['category'] == 'Misconceptions') | (dataset['category'] == 'Misconceptions: Topical')]

file_name = 'contrastive_pairs.json'
Formatting_Dataset(dataset_1, file_name)
del dataset_1

dataset_2 = dataset[~dataset['category'].isin(['Misconceptions', 'Misconceptions: Topical'])]
file_name = 'contrastive_pairs_held_out.json'
Formatting_Dataset(dataset_2, file_name)
del dataset_2

print('Saved Contrastive Pairs and Contrastive Pairs Held Out\n')

print('Loading Data')
with open('contrastive_pairs.json', 'r') as f:
    data = json.load(f)

with open('contrastive_pairs_held_out.json', 'r') as f:
    held_out_data = json.load(f)

HeldOut_questions = sorted(set(item['question'] for item in held_out_data))
questions = sorted(set(item['question'] for item in data))

print('Data Loaded')


###### RUNNING ######
Model_List = ['gpt2', 'gpt2-medium', 'EleutherAI/pythia-410m', 'gpt2-large', 'EleutherAI/pythia-1b', 'gpt2-xl', 'facebook/opt-1.3b', 'EleutherAI/pythia-1.4b', 'EleutherAI/pythia-2.8b', 'facebook/opt-1.3b']

for model_name in Model_List:
    print(f'Current model to load: {model_name}')
    Model = TransformerBridge.boot_transformers(model_name, device=device)
    safe_name = model_name.replace('/', '--')
    print(f'{model_name} loaded')
    safe_path = os.path.join(path, safe_name)
    os.makedirs(safe_path, exist_ok=True)

    print('\nCaching Prompts')
    t0 = time.time()
    layer_dict = {safe_path: Running(data, Model, safe_path)}
    caching_time = time.time() - t0                       # NEW
    print(f'\tCaching took {caching_time:.2f}s')

    print('\nFinding the Best Probe')
    t0 = time.time()
    Find_Best_Layer(layer_dict)
    find_best_layer_time = time.time() - t0                # NEW
    print(f'\tFind_Best_Layer took {find_best_layer_time:.2f}s')

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

    print('\nSanity Checking (A should be higher | B should be lower)')
    t0 = time.time()
    Sanity_Check(Model, LR, Layer_Name, safe_path)
    sanity_check_time = time.time() - t0                    # NEW
    print(f'\tSanity_Check took {sanity_check_time:.2f}s')

    print('\nEvaluating Steering')
    print('\tOn Held-Out Data')
    t0 = time.time()
    Steering_Test(Model, LR, Layer_Name, safe_name, held_out_data, title='Held_Out_alpha_results')
    steering_held_out_time = time.time() - t0                # NEW
    print(f'\tSteering (held-out) took {steering_held_out_time:.2f}s')

    print('\n\tOn Mis-conception (Trained) Data')
    t0 = time.time()
    Steering_Test(Model, LR, Layer_Name, safe_name, data)
    steering_trainset_time = time.time() - t0                # NEW
    print(f'\tSteering (trainset) took {steering_trainset_time:.2f}s')

    print('\nSAVE SPACE: Removing Cached Activations: ')
    Remove_Cached(safe_path)
    Model.to("cpu")
    del Model
    gc.collect()
    torch.cuda.empty_cache()
    print("Allocated:", torch.cuda.memory_allocated() / 1e9, "GB")
    print("Reserved:", torch.cuda.memory_reserved() / 1e9, "GB")

    # NEW: sum the individual pieces rather than a single wall-clock start/end
    model_total_time = (
        caching_time + find_best_layer_time + sanity_check_time
        + steering_held_out_time + steering_trainset_time
    )
    print(f'Total time for this model: {model_total_time:.2f}s or ({model_total_time/60:.2f} min)')

    model_timings = {
        'caching': round(caching_time, 2),
        'find_best_layer': round(find_best_layer_time, 2),
        'sanity_check': round(sanity_check_time, 2),
        'steering_held_out': round(steering_held_out_time, 2),
        'steering_trainset': round(steering_trainset_time, 2),
        'total': round(model_total_time, 2)
    }

    timings_path = os.path.join(safe_path, f'{safe_name}_Execution_Time.json')
    with open(timings_path, 'w') as file:
        json.dump(model_timings, file, indent=2)
    print(model_timings)
    print("------------------------------------------------\n")

