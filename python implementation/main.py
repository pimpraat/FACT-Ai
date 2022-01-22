# %%
import random
import numpy as np
from itertools import repeat
from data import get_synthetic_data, get_bank_data, get_pokec_data
from secretary import secretary_algorithm, one_color_secretary_algorithm, multiple_color_secretary_algorithm, multiple_color_thresholds, shuffle_input
from secretary_evaluation import evaluation

# %%
def shuffle_within_group(candidates_dict):

    for v in candidates_dict.values():
        random.shuffle(v[0])
        
    return candidates_dict

# %% [markdown]
# def prepare_multicolor_input(candidates_scores, candidates_dict):

# %% [markdown]
#     color_per_candidate = []

# %% [markdown]
#     for color in list(candidates_dict.keys()):
#         color_per_candidate.extend(repeat(color, len(candidates_dict[color][0])))

# %% [markdown]
#     color_matching = list(zip(candidates_scores, color_per_candidate))
#     random.shuffle(color_matching)

# %% [markdown]
#     candidates, color_per_candidate = zip(*color_matching)

# %% [markdown]
#     return candidates, color_per_candidate

# %%
def SecretaryExperiment(candidates_dict, n):
    
    colors = list(candidates_dict.keys())
    max_colors = {color: None for color in colors}
    candidates_scores = np.concatenate([v[0] for v in candidates_dict.values()])
    candidates_probabilities = [v[1] for v in candidates_dict.values()]
    
    results_SA, results_SCSA, results_MCSA = [], [], []
    
    thresholds = multiple_color_thresholds(colors, candidates_probabilities)
    
    color_per_candidate = []
    
    for color in list(candidates_dict.keys()):
        color_per_candidate.extend(repeat(color, len(candidates_dict[color][0])))
        max_colors[color] = max(candidates_dict[color][0])
        
    print("HERE: ", candidates_scores)
    print(color_per_candidate)
    print(max_colors)

    for i in range(200):
        candidates_scores, color_per_candidate = shuffle_input(candidates_scores, color_per_candidate)
        result_SA = secretary_algorithm(candidates_scores, color_per_candidate, max_colors)
        # print("Best candidate in SA: ", result_SA)
        results_SA.append(result_SA)
        
        shuffle_within_group(candidates_dict)
        result_SCSA = one_color_secretary_algorithm(candidates_dict, color_per_candidate, max_colors)
        # print("Best candidate in SCSA: ", result_SCSA)
        results_SCSA.append(result_SCSA)
    
        result_MCSA = multiple_color_secretary_algorithm(colors, candidates_scores, color_per_candidate, thresholds, n, max_colors)
        # print("Best candidate in MCSA: ", result_MCSA, '\n')
        results_MCSA.append(result_MCSA)

    print("HERE: ", candidates_scores)
    print(color_per_candidate)
    
    evaluation("SA", results_SA, colors, n)
    evaluation("SCSA", results_SCSA, colors, n)
    evaluation("MCSA", results_MCSA, colors, n)

# %%
def SyntheticExperiment():
    
    colors = ['red', 'green', 'blue', 'yellow']
    n = [10, 100, 1000, 10000]
    probabilities = [0.27, 0.26, 0.25, 0.24]
    
    synthetic_data = get_synthetic_data(colors, n, probabilities)

    SecretaryExperiment(synthetic_data, n)

# %%
def UnbalancedSyntheticExperiment():

    colors = ['red', 'green', 'blue', 'yellow']
    n = [10, 100, 1000, 10000]
    probabilities = [0.3, 0.25, 0.25, 0.2]
    
    synthetic_data = get_synthetic_data(colors, n, probabilities)

    SecretaryExperiment(synthetic_data, n)

# %%
def BankExperiment():
    
    n = []
    probabilities = [0.2, 0.21, 0.22, 0.23, 0.24]
    path = 'data/bank_raw.csv'
    bank_data = get_bank_data(path, probabilities)
    
    for values in bank_data.values():
        n.append(len(values[0]))
        
    SecretaryExperiment(bank_data, n)

# %% [markdown]
# def InfMaxExperiment():
#     return 0


# %%
if __name__ == "__main__":
    
    SyntheticExperiment()
    # UnbalancedSyntheticExperiment()
    # BankExperiment()
    # InfMaxExperiment()
