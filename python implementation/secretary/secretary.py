import numpy as np


def secretary_algorithm(all_candidates, candidates_color, max_colors):

    stop_rule = round(len(all_candidates)/np.e)
    max_value = max(all_candidates[:stop_rule])

    try:
        best_candidate = next(x for x in all_candidates[stop_rule:] if x > max_value)
    except StopIteration:
        best_candidate = all_candidates[-1]

    winning_color = candidates_color[list(all_candidates).index(best_candidate)]

    return best_candidate, winning_color, best_candidate == max_colors[winning_color]

def one_color_secretary_algorithm(candidates_per_group, candidates_color, max_colors):
    
    rand_balanced = np.random.rand()

    for color in candidates_per_group.keys():
        
        current_group = candidates_per_group[color]
        if rand_balanced <= current_group[1]:
            winning_color = color
            break
        
        rand_balanced -= current_group[1]
            
    best_candidate, _, _ = secretary_algorithm(current_group[0], candidates_color, max_colors)
    
    return best_candidate, winning_color, best_candidate == max_colors[winning_color]

def multiple_color_secretary_algorithm(all_candidates, candidates_color, max_colors, *args):
    
    colors, thresholds, n = args[0], args[1], args[2]
    
#     for i in range(len(thresholds)):
#         thresholds[i] = thresholds[i] * n[i]
        
    max_until_threshold = [0] * len(colors)
    current_color_index = [0] * len(colors)
    
    for i in range(0, len(all_candidates)):

        current_color = candidates_color[i]
        if current_color_index[colors.index(current_color)] < thresholds[colors.index(current_color)]:
            max_until_threshold[colors.index(current_color)] = max(max_until_threshold[colors.index(current_color)], all_candidates[i])
            current_color_index[colors.index(current_color)] += 1

    for i in range(0, len(all_candidates)):

        current_color = candidates_color[i]
        if current_color_index[colors.index(current_color)] >= thresholds[colors.index(current_color)]:

            if all_candidates[i] >= max_until_threshold[colors.index(current_color)]:
                return all_candidates[i], current_color, all_candidates[i] == max_colors[current_color]

    return all_candidates[-1], current_color, all_candidates[-1] == max_colors[current_color]

def multiple_color_thresholds(p):
    
    t = [0] * len(p)
    k = len(p)
    
    t[k-1] = np.power((1 - (k - 1) * p[k - 1]), (1 / (k - 1)))
    
    for j in range(k-2, 0, -1): #this is equivalent to 2 <= i <= k-1
        
        sum = np.sum([p[r] for r in range(0, j+1)])
        sum /= j
        t[j] = t[j+1] * np.power((sum - p[j]) / (sum - p[j+1]), 1 / j)

    t[0] = t[1] * np.exp(p[1] / p[0] - 1)

    return t
