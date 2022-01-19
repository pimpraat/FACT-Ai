import numpy as np
from itertools import repeat
import random
from operator import itemgetter

def secretary_algorithm(candidates):

    stop_rule = round(len(candidates)/np.e)

    # print("First fraction of candidates: ", candidates[:stop_rule])
    # print("The remaining candidates: ", candidates[stop_rule:])

    # Catch error if best candidate is already in [:1/e]. Return last
    try:
        best_candidate = next(x for x in candidates[stop_rule:] if x>max(candidates[:stop_rule]))
    except StopIteration:
        best_candidate = candidates[-1]

    return best_candidate

def one_color_secretary_algorithm(colored_candidates):
    max_probability = (max(np.array(list(colored_candidates.values()), dtype=object)[:,1]))

    for values in colored_candidates.values():
        if values[1] == max_probability: # Overwrites for all p are equal, but that's the heuristic they set
            single_color = values[0]
            # print(single_color)
            
    best_candidate = secretary_algorithm(single_color)
    
    return best_candidate

def multiple_color_secretary_algorithm(colors, candidates, color_per_candidate, thresholds):

    n = 7

    print(colors)
    print(candidates)
    print(color_per_candidate)
    print(thresholds, '\n')
    
    max_C_j = [0, 0, 0]
    
    stop_rule = [round(n*thresholds[i]) for i in range(len(thresholds))]
    
    for i in range(len(candidates)):
        print(candidates[i])
        current_color = color_per_candidate[i]
        print(current_color)
        
        if stop_rule[colors.index(current_color)] == 0 and candidates[i] > max_C_j[colors.index(current_color)]:
            print("FINAL CANDIDATE: ", candidates[i])
            return candidates[i]
        
        elif candidates[i] > max_C_j[colors.index(current_color)]:
            print("new max: ", candidates[i])
            max_C_j[colors.index(current_color)] = candidates[i]
            stop_rule[colors.index(current_color)] -= 1
            
        elif stop_rule[colors.index(current_color)] != 0:
            print("minus 1: ")
            stop_rule[colors.index(current_color)] -= 1
           
        print(stop_rule) 
        print("\n")
    
    
def create_data(n, colors, probabilities):
    
    candidates = np.arange(0, n, 5)
    np.random.shuffle(candidates)

    colored_candidates = {'red': [], 'green': [], 'blue': []}

    i=0
    for j in range(len(colors)):
        colored_candidates[colors[j]] = [candidates[i:i+7], probabilities[j]]
        i = i+7
        
    return candidates, colored_candidates

def multiple_color_thresholds(colors, probabilities):
    
    # print(colors)
    # print(probabilities)
    
    k = len(colors)
    thresholds = []
    
    sort_index = sorted(enumerate(probabilities), reverse=True, key = itemgetter(1))
    probabilities.sort(reverse = True)
    
    thresholds.insert(0, np.power((1 - (k - 1) * probabilities[-1]), 1 / (k - 1)))
    
    for j in range(k-1, 1, -1):

        dividend = [probabilities[r-1]/(j-1) - probabilities[j-1] for r in range(1, j+1)]
        divisor = [probabilities[r-1]/(j-1) - probabilities[j] for r in range(1, j+1)]
        thresholds.insert(0, thresholds[0] * np.power((sum(dividend) / sum(divisor)), 1 / (j - 1)))
    
    thresholds.insert(0, thresholds[0] * np.power(np.e, probabilities[1] / probabilities[0] - 1))

    unsort_index = [tuple[0] for tuple in sort_index]
    thresholds = [thresholds[i] for i in unsort_index]
    
    print(thresholds)
    return thresholds
    
if __name__ == "__main__":

    n = 105
    colors = ['red', 'green', 'blue']
    probabilities = [0.3, 0.5, 0.2]
    candidates, colored_candidates = create_data(n, colors, probabilities)
    print(colored_candidates)
    
    best_SA = secretary_algorithm(candidates)
    best_SCSA = one_color_secretary_algorithm(colored_candidates)
    
    print("Best candidate in SA: ", best_SA)
    print("Best candidate in SCSA: ", best_SCSA)
    
    
    print(candidates)
    
    color_per_candidate = []
    color_per_candidate.extend(repeat("red", len(colored_candidates['red'][0])))
    color_per_candidate.extend(repeat("green", len(colored_candidates['green'][0])))
    color_per_candidate.extend(repeat("blue", len(colored_candidates['blue'][0])))
    
    color_matching = list(zip(candidates, color_per_candidate))
    random.shuffle(color_matching)
    
    candidates, color_per_candidate = zip(*color_matching)
    
    print(candidates)
    print(color_per_candidate)
    
    thresholds = multiple_color_thresholds(colors, probabilities)
    best_MCSA = multiple_color_secretary_algorithm(colors, candidates, color_per_candidate, thresholds)
    
    print("Best candidate in MCSA: ", best_MCSA)
