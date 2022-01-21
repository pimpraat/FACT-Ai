import random
import pickle
import numpy as np
from itertools import repeat
from data import get_synthetic_data, get_bank_data
from secretary import secretary_algorithm, one_color_secretary_algorithm, multiple_color_secretary_algorithm, multiple_color_thresholds


def shuffle_input(candidates, color_per_candidate):
    
    color_matching = list(zip(candidates, color_per_candidate))
    random.shuffle(color_matching)
    
    return zip(*color_matching)

def shuffle_within_group(candidates_per_group):

    for v in candidates_per_group.values():
        random.shuffle(v[0])
        
    return candidates_per_group
    
def secretary_experiment(candidates_per_group, *args):
    
    colors, probabilities, n = args[0], args[1], args[2]
    
    all_candidates = np.concatenate([v[0] for v in candidates_per_group.values()])
    thresholds = multiple_color_thresholds(probabilities)

    results_SA, results_SCSA, results_MCSA = [], [], []
    
    candidates_color = []
    max_colors = {color: None for color in colors}

    for color in list(candidates_per_group.keys()):
        candidates_color.extend(repeat(color, len(candidates_per_group[color][0])))
        max_colors[color] = max(candidates_per_group[color][0])

    for _ in range(20000):
        all_candidates, candidates_color = shuffle_input(all_candidates, candidates_color)
        result_SA = secretary_algorithm(all_candidates, candidates_color, max_colors)
        results_SA.append(result_SA)
        
        shuffle_within_group(candidates_per_group)
        result_SCSA = one_color_secretary_algorithm(candidates_per_group, candidates_color, max_colors)
        results_SCSA.append(result_SCSA)
    
        result_MCSA = multiple_color_secretary_algorithm(all_candidates, candidates_color, max_colors, colors, thresholds, n)
        results_MCSA.append(result_MCSA)

    # evaluation("SA", results_SA, colors, n)
    # evaluation("SCSA", results_SCSA, colors, n)
    # evaluation("MCSA", results_MCSA, colors, n)

    return results_SA, results_SCSA, results_MCSA

def synthetic_experiment():
    
    colors = ['red', 'green', 'blue', 'yellow']
    probabilities = [0.25, 0.25, 0.25, 0.25]
    n = [10, 100, 1000, 10000]
    
    synthetic_data = get_synthetic_data(colors, probabilities, n)
    results = secretary_experiment(synthetic_data, colors, probabilities, n)
    
    return results

def unbalanced_synthetic_experiment():

    colors = ['red', 'green', 'blue', 'yellow']
    probabilities = [0.3, 0.25, 0.25, 0.2]
    n = [10, 100, 1000, 10000]
    
    synthetic_data = get_synthetic_data(colors, probabilities, n)
    results = secretary_experiment(synthetic_data, colors, probabilities, n)
    
    return results
    
def bank_experiment():
    
    probabilities = [0.2, 0.21, 0.22, 0.23, 0.24]
    path = 'bank_raw.csv'
    n = []
    bank_data = get_bank_data(path, probabilities)

    for values in bank_data.values():
        n.append(len(values[0]))
        
    results = secretary_experiment(bank_data, _, probabilities, n)
    
    return results

# def InfMaxExperiment():
#     return 0


def run_experiments():
    
    results = synthetic_experiment()
    results2 = unbalanced_synthetic_experiment()
    
    with open('results_synthetic1.pickle', 'wb') as f:
        pickle.dump(results, f)
        
    with open('results_synthetic2.pickle', 'wb') as f:
        pickle.dump(results2, f)
