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

from groq import Groq, RateLimitError

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

print('Loading Cloud Model')
client = Groq()

MIN_INTERVAL = 2.0
_last_call_time = 0.0
BASE_BACKOFF = 5.0
MAX_BACKOFF = 60.0

def Call_LLM(prompt, model='openai/gpt-oss-20b', temperature=0.7):
    global _last_call_time

    elapsed = time.monotonic() - _last_call_time
    if elapsed < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - elapsed)

    attempt = 0
    while True:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            _last_call_time = time.monotonic()
            return response.choices[0].message.content

        except RateLimitError:
            wait = min(BASE_BACKOFF * (2 ** attempt), MAX_BACKOFF) + random.uniform(0, 1)
            print(f"\tRate limited, backing off {wait:.1f}s (attempt {attempt+1})")
            time.sleep(wait)
            _last_call_time = time.monotonic()
            attempt += 1

###### PPL Function ######
def PPL(model, string: str) -> float:
    tokens = model.to_tokens(string)
    with torch.no_grad():
        logits = model(tokens)

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    target_tokens = tokens[0, 1:]
    pred_log_probs = log_probs[0, :-1, :]

    token_log_probs = pred_log_probs[
        torch.arange(len(target_tokens)), target_tokens
    ]

    mean_nll = -token_log_probs.mean().item()
    return float(np.exp(mean_nll))

###### Evaluate ######
def Evaluate(model, string: str, v: np.ndarray, layer_name: str) -> float:

    tokens = model.to_tokens(string)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=layer_name)

    activation = cache[layer_name][0, -1, :].cpu().numpy()
    i_score = float(np.dot(activation, v))

    return i_score

###### Mutate ######
def Mutate(prompt):
    Input = f"""
You are improving a prompt.

Original prompt:
"{prompt}"

Rewrite it to be slightly better, clearer, or more effective.
Keep the meaning similar.
Return ONLY the new prompt.
"""
    return Call_LLM(Input).strip().replace("\n", " ")

###### Crossover ######
def Crossover(P1, P2):
    Input = f"""
You are combining two Prompts into a better prompt.

Prompt-1: {P1}
Prompt-2: {P2}

Create a new prompt such that it would keep the best parts of both prompts
RETURN ONLY THE NEW PROMPT.
    """
    return Call_LLM(Input).strip().replace("\n", " ")


###### Seed Generator ######
def Generate_Sentences(count):
    prompt = f"""
Generate {count} distinct prompts.

Each prompt should encourage careful, step-by-step reasoning before answering a question.

Requirements:
- Each prompt must be a single sentence
- No numbering
- No explanations
- No extra text
- Each on a new line

Output only the prompts.
"""
    response = Call_LLM(prompt)
    response = response.split("\n")
    response = [r.strip() for r in response]
    return response

###### Roulette #######
def Roulette(population, scores, k):
    scores = np.array(scores)

    min_score = scores.min()
    if min_score < 0:
        scores = scores - min_score + 1e-6

    probs = scores / scores.sum()
    indices = np.random.choice(len(population), size=k, p=probs)

    return [population[i] for i in indices]

###### Genetic Algorithm ######
def GA(Model, vector: np.ndarray, layer_name: str, Population_size: int = 20, Generations: int = 5, death_rate: float = 0.5, mutation_rate: float = 0.75, Data=None):
    print(f'\nRunning Genetic Algorithm (Population: {Population_size} | Generations: {Generations})')

    print('\tObtaining Data')
    if Data is None:
        print('\tGenerating Sentences')
        Population = Generate_Sentences(Population_size)
        print(f"\tUnique sentences generated: {len(set(Population))} / {Population_size}")
    else:
        Population = Data[:Population_size]
        while len(Population) < Population_size:
            Population.append(Mutate(random.choice(Data)))

    Log = []
    best_ever_fitness = -np.inf
    best_ever_prompt = None
    Population_sorted = []

    for generation in range(Generations):
        print(f'\n\tGeneration: {generation + 1}')

        scores = [
            Evaluate(Model, string, vector, layer_name) - Lambda * PPL(Model, string)
            for string in Population
        ]

        Pairs = sorted(zip(scores, Population), key=lambda x: x[0], reverse=True)
        scores_sorted = [s for s, _ in Pairs]
        Population_sorted = [p for _, p in Pairs]

        if scores_sorted[0] > best_ever_fitness:
            best_ever_fitness = scores_sorted[0]
            best_ever_prompt = Population_sorted[0]

        best_fitness = scores_sorted[0]
        mean_fitness = float(np.mean(scores_sorted))
        Log.append({
            "generation": generation + 1,
            "best_fitness": best_fitness,
            "mean_fitness": mean_fitness,
            "best_prompt": Population_sorted[0]
        })

        print(f"\t-Gen {generation + 1} | best = {best_fitness:.4f} | mean = {mean_fitness:.4f} | '{Population_sorted[0]}'")
        for x in range(int(len(Population_sorted)*0.1)):
            print(f'\t\t{x}: {Pairs[x]}')

        Parents = Roulette(Population_sorted, scores_sorted, Population_size)

        NewPopulation = []

        while len(NewPopulation) < Population_size:
            p1, p2 = random.sample(Parents, 2)

            child = Crossover(p1, p2)

            if random.random() < mutation_rate:
                child = Mutate(child)

            NewPopulation.append(child)

        Population = NewPopulation

        if best_ever_prompt not in Population:
            Population[0] = best_ever_prompt

    return Log, Population, Population_sorted

###### COMPARING GA-GENERATED WITH CoT ######
def Comparing_Prompts(Model, LR, layer_name, safe_name, questions, n_ga_prompts=2):

    CoT_Generic = [
        "Let's think step by step",
        "Let's work through this carefully before answering"
    ]

    sentences_path = os.path.join(path, safe_name, f'{safe_name}_Sentences.txt')
    with open(sentences_path, 'r') as f:
        all_lines = f.read().split("\n")


    real_lines = [line.strip() for line in all_lines if line.strip()]
    GA_prompts = real_lines[-n_ga_prompts:]
    print(f'Using last {len(GA_prompts)} non-blank lines from {sentences_path} as GA prompts:')
    for p in GA_prompts:
        print(f'  - {p}')

    def get_activation(Model, Input, layer_name):
        Tokens = Model.to_tokens(Input)
        with torch.no_grad():
            _, Cache = Model.run_with_cache(Tokens, names_filter=layer_name)
        return Cache[layer_name][0, -1, :].cpu().numpy()

    # FIX: parameters now match what the body actually uses -- `prefixes` and `questions`
    # were referenced before but never existed as real names in scope.
    def evaluate_prefixes(Model, prefixes, questions, LR, layer_name):
        results = {}
        for i, prefix in enumerate(prefixes):
            hits = []
            for question in tqdm(questions):
                prompt = f"{prefix}\n{question}"
                activation = get_activation(Model, prompt, layer_name).reshape(1, -1)
                prob = LR.predict_proba(activation)[0, 1]
                hits.append(1 if prob > 0.5 else 0)
            rate = round(float(np.mean(hits)), 3)
            results[f'prompt_{i+1}'] = {'prefix': prefix, 'probe_activation_rate': rate}
            print(f"  prompt {i+1}: rate={rate:.3f} | '{prefix}'")
        return results

    results = {}

    print('Testing Genetic Algorithm')
    results['Genetic Algorithm'] = evaluate_prefixes(Model, GA_prompts, questions, LR, layer_name)

    print('Testing Chain-Of-Thought')
    results['Chain of Thought'] = evaluate_prefixes(Model, CoT_Generic, questions, LR, layer_name)

    print(results)

    ###### Save ######
    folder_path = os.path.join(path, safe_name)
    os.makedirs(folder_path, exist_ok=True)
    save_path = os.path.join(folder_path, f'{safe_name}_Comparing_Prompt.json')
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved {save_path}')

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
Model_List = ['gpt2', 'EleutherAI/pythia-1.4b']

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

    seed_prompts = [
    "Before answering, pause and check whether your first instinct might be a common misconception.",
    "Walk through your reasoning step by step, and reconsider any assumption that feels too automatic.",
    "Think carefully about this question, and if your initial answer seems obvious, double-check it for errors.",
    "Take a moment to reflect on whether the popular answer to this question is actually correct.",
    "Reason through each step methodically, watching for places where a common belief might be wrong.",
    "Before giving your final answer, ask yourself whether you are repeating a widely believed but false claim.",
    "Slow down and examine your reasoning carefully, especially if the answer seems like common knowledge.",
    "Consider the question step by step, and be willing to revise your answer if it turns out to be a misconception.",
    "Pause here, reconsider the question from scratch, and check your answer against what is actually true.",
    "Work through this logically, and flag any point where you might be relying on a false popular belief.",
    "Take your time to think this through, questioning any answer that feels too familiar or too easy.",
    "Carefully reconsider whether your first answer reflects a genuine fact or just a widespread myth.",
    "Break this down step by step, and correct course if you notice you are echoing a common misconception.",
    "Before responding, reflect on whether this question is designed to test a well-known but incorrect belief.",
    "Think step by step, and remain skeptical of any answer that matches a popular but unverified claim.",
    "Examine your own reasoning as you go, and be ready to catch and fix any mistaken assumption.",
    "Consider this carefully, pausing to check if the obvious answer might actually be a common falsehood.",
    "Reason through the problem in order, staying alert to any step where a misconception could creep in.",
    "Take a step back and verify your reasoning before committing to an answer that might be widely misunderstood.",
    "Go through this methodically, and revise your answer if you realize it echoes a mistaken common belief."
    ]

    t0 = time.time()
    Log, Final_Population, _ = GA(Model, vector = vect, layer_name = Layer_Name, Population_size = 20, Generations = 5, mutation_rate = 0.75, Data=seed_prompts.copy())


    report_path = os.path.join(path, safe_name, f'{safe_name}_GA_report.json')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    Log_df = pd.DataFrame(Log)
    Log_df.to_json(report_path, orient='records', lines=True)
    print(f'\nLog saved to {report_path}')

    file_name = os.path.join(path, safe_name, f'{safe_name}_Sentences.txt')
    os.makedirs(os.path.dirname(file_name), exist_ok = True)
    with open(file_name, 'a') as file:
        for sentence in Final_Population:
            file.write(sentence + "\n")

    Comparing_Prompts(Model, LR, Layer_Name, safe_name, HeldOut_questions, n_ga_prompts=2)
    Comparing_Prompts(Model, LR, Layer_Name, safe_name, questions, n_ga_prompts=2)

    total = time.time() - t0

    print(f'total time by {safe_name}: {total}')

    Model.to("cpu")
    del Model
    gc.collect()
    torch.cuda.empty_cache()

    print("------------------------------------------------\n")