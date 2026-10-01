###### Imports ######
import os
import re
import time
import json
import random

from tqdm import tqdm
import numpy as np
import pandas as pd
import joblib
import torch

from transformer_lens import HookedTransformer
from transformer_lens.model_bridge import TransformerBridge
from transformers import AutoModelForCausalLM, AutoTokenizer

###### Device ######
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
os.environ["HF_TOKEN"] = "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"  # Note: remove before pushing to GitHub

###### Config ######
Lambda = 0.3
path = 'Log'

###### Mutator — loaded once at startup ######
print('Loading Mutator Models')
Mutate_Tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
Mutate_Model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",torch_dtype=torch.float32)
Mutate_Model.eval()
print('Models Loaded\n')

###### Reproducibility ######
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

###### PPL ######
def PPL(model, string: str) -> float:
    tokens = model.to_tokens(string)
    with torch.no_grad():
        logits = model(tokens)

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    target_tokens = tokens[0, 1:] # shape: (seq_len - 1,)
    pred_log_probs = log_probs[0, :-1, :] # shape: (seq_len - 1, vocab_size)

    # Gather log prob of each actual next token
    token_log_probs = pred_log_probs[
        torch.arange(len(target_tokens)), target_tokens
    ]

    # Perplexity = exp(mean negative log prob)
    mean_nll = -token_log_probs.mean().item()
    return float(np.exp(mean_nll))

###### Evaluate ######
def Evaluate(model, string: str, v: np.ndarray, layer_name: str) -> float:

    tokens = model.to_tokens(string)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter = layer_name)

    activation = cache[layer_name][0, -1, :].cpu().numpy()   # shape: (d_model,)
    i_score = float(np.dot(activation, v))

    return i_score

###### Mutate ######
def Mutate(string: str) -> str:

    Message = [
        {
            "role": "user",
            "content": f"Rewrite this sentence with a small change. Return only the sentence, nothing else.\n{string}"
        }
    ]

    Text = Mutate_Tokenizer.apply_chat_template(Message, tokenize = False, add_generation_prompt = True)
    Input = Mutate_Tokenizer(Text, return_tensors="pt")

    with torch.no_grad():
        generated = Mutate_Model.generate(**Input, max_new_tokens = 40, do_sample = True, temperature = 0.7, pad_token_id = Mutate_Tokenizer.eos_token_id)

    new_tokens = generated[0][Input["input_ids"].shape[1]:]
    Output = Mutate_Tokenizer.decode(new_tokens, skip_special_tokens = True)

    return Output.strip()

###### Seed Generator ######
def Generate_Sentences(count: int):
    Message = [
        {
            "role": "user",
            "content": (
                f"Generate {count} short sentences that encourage careful, reflective thinking "
                f"before answering a question. One sentence per line, there should be no numbering, no explanation."
            )
        }
    ]

    Text = Mutate_Tokenizer.apply_chat_template(Message, tokenize = False, add_generation_prompt = True)
    Input = Mutate_Tokenizer(Text, return_tensors="pt")

    with torch.no_grad():
        generated = Mutate_Model.generate(**Input, max_new_tokens = count * 20, do_sample = True, temperature = 0.9, pad_token_id=Mutate_Tokenizer.eos_token_id)

    new_tokens = generated[0][Input["input_ids"].shape[1]:]
    Output = Mutate_Tokenizer.decode(new_tokens, skip_special_tokens=True)

    sentences = [re.sub(r'^\d+[\.\)]\s*', '', s).strip() for s in Output.split('\n') if s.strip()]

    # If model gave fewer than count, fill by repeating randomly
    while len(sentences) < count:
        sentences.append(random.choice(sentences))

    return sentences[:count]

###### Genetic Algorithm ######
def GA(Model, vector: np.ndarray, layer_name: str, Population_size: int = 20, Generations: int = 5, death_rate: float = 0.5, mutation_rate: float = 0.75, Data  = None, seed: int = 67):
    print(f'\nRunning Genetic Algorithm (Population: {Population_size} | Generations: {Generations})')
    print('\tSetting Seed')
    set_seed(seed)

    # --- Initialize population ---
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

    for generation in range(Generations):
        print(f'\n\tGeneration: {generation + 1}')

        # --- Score ---
        scores = [
            Evaluate(Model, string, vector, layer_name) - Lambda * PPL(Model, string)
            for string in Population
        ]

        # --- Sort descending by fitness ---
        Pairs = sorted(zip(scores, Population), key=lambda x: x[0], reverse=True)
        scores_sorted = [s for s, _ in Pairs]
        Population_sorted = [p for _, p in Pairs]

        # --- Log ---
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

        # --- Select survivors ---
        number_of_survivors = int(Population_size * (1 - death_rate))
        Survivors = Population_sorted[:number_of_survivors]

        # --- Mutate survivors to fill population back up ---
        Candidates = []
        for i in range(Population_size - number_of_survivors):
            parent = Survivors[i % number_of_survivors]
            if random.random() < mutation_rate:
                Candidates.append(Mutate(parent))
            else:
                Candidates.append(parent)

        Population = Survivors + Candidates

    return Log, Population



###### Running ######
Model_List = ['gpt2', 'gpt2-medium', 'EleutherAI/pythia-410m', 'gpt2-large', 'EleutherAI/pythia-1b', 'gpt2-xl', 'facebook/opt-1.3b', 'EleutherAI/pythia-1.4b', 'EleutherAI/pythia-2.8b', 'facebook/opt-1.3b']

seed = int(input('Enter a seed number: '))
Population_size = int(input('Enter population size (default 20, press enter to skip): ') or 20)
Generations = int(input('Enter number of generations (default 5, press enter to skip): ') or 5)

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

    # --- Load TransformerLens model ---
    print(f'\nLoading model: {model_name}')
    Model = HookedTransformer.from_pretrained(model_name, device=device)

    # --- Run GA ---
    Log, Final_Population = GA(Model, vect, Layer_Name, Population_size=Population_size, Generations=Generations, seed=seed)

    # --- Save log ---
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

    Lambda_Values = [0.0, 0.1, 0.5, 1.0, 5.0]
    Grid_Results = []

    for lam in Lambda_Values:
        print(f'\nLambda Grid: running with Lambda={lam}')
        Lambda = lam
        Grid_Log, Grid_Population = GA(Model, vect, Layer_Name, Population_size=Population_size, Generations=Generations, seed=seed)

        file_name = os.path.join(path, safe_name, f'{safe_name}_{lam}_Sentences.txt')
        os.makedirs(os.path.dirname(file_name), exist_ok = True)
        with open(file_name, 'a') as file:
            for sentence in Final_Population:
                file.write(sentence + "\n")

        best_prompt = Grid_Log[-1]['best_prompt']
        best_i = Evaluate(Model, best_prompt, vect, Layer_Name)
        best_ppl = PPL(Model, best_prompt)
        Grid_Results.append({
            'lambda': lam,
            'best_fitness': Grid_Log[-1]['best_fitness'],
            'i_score': best_i,
            'ppl': best_ppl,
            'best_prompt': best_prompt
        })
        print(f'Lambda={lam} | i_score={best_i:.4f} | ppl={best_ppl:.4f} | fitness={Grid_Log[-1]["best_fitness"]:.4f}')

    # save grid results
    grid_path = os.path.join(path, safe_name, f'{safe_name}_lambda_grid.json')
    pd.DataFrame(Grid_Results).to_json(grid_path, orient='records', lines=True)
    print(f'\nLambda grid saved to {grid_path}')

    # plot: i_score vs ppl across lambda values (Pareto frontier)
    fig, ax = plt.subplots()
    ppls = [result['ppl'] for result in Grid_Results]
    iscores = [result['i_score'] for result in Grid_Results]
    lams = [result['lambda'] for result in Grid_Results]

    ax.scatter(ppls, iscores)

    for i, lam in enumerate(lams):
        ax.annotate(f'λ={lam}', (ppls[i], iscores[i]), textcoords='offset points', xytext=(5, 5))
    ax.set_xlabel('PPL (lower = more fluent)')
    ax.set_ylabel('I_score (higher = more introspective)')
    ax.set_title('Pareto Frontier: I_score vs PPL across Lambda values')
    plot_path = os.path.join(path, safe_name, f'{safe_name}_pareto.png')
    plt.savefig(plot_path)
    plt.close()
    print(f'Pareto plot saved to {plot_path}')