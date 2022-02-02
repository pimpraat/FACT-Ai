import typing
import numpy as np
from data import SecretaryInstance


def secretary_algorithm(all_candidates, max_colors) -> SecretaryInstance:
    """This method runs the first baseline: the Secretary Algorithm

    Args:
        all_candidates ([SecretaryInstance]): List of all candidates
        max_colors ([string]): List of names of all groups

    Returns:
        SecretaryInstance: The selected candidate
    """

    stop_rule = round(len(all_candidates) / np.e)
    max_value = np.max([item.score for item in all_candidates[:stop_rule]])

    try:
        best_candidate = next(x for x in all_candidates[stop_rule:] if x.score > max_value)
        best_candidate.ismax = best_candidate.score == max_colors[best_candidate.color]
    except StopIteration:
        best_candidate = SecretaryInstance(-1, -1, None)

    return best_candidate

def one_color_secretary_algorithm(candidates, max_colors, *args) -> SecretaryInstance:
    """This method runs the second baseline: the One Color Secretary Algorithm

    Args:
        all_candidates ([SecretaryInstance]): List of all candidates
        max_colors ([string]): List of names of all groups
        args (tuple): Necessary arguments, namely the list of colors, probabilities and size of groups

    Returns:
        SecretaryInstance: The selected candidate
    """
    
    colors, probabilities = args[0], args[1]
    rand_balanced = np.random.rand()

    for i in range(len(probabilities)):
        
        if rand_balanced <= probabilities[i]:
            winning_group = [x for x in candidates if x.color == colors[i]]
            break
        
        rand_balanced -= probabilities[i]
            
    best_candidate = secretary_algorithm(winning_group, max_colors)
    
    try:
        best_candidate.ismax = best_candidate.score == max_colors[best_candidate.color]
    except KeyError:
        best_candidate.ismax = False

    return best_candidate

def multiple_color_secretary_algorithm(candidates, max_colors, *args) -> SecretaryInstance:
    """This method runs the fair opt algorithm: the Multiple Color Secretary Algorithm

    Args:
        all_candidates ([SecretaryInstance]): List of all candidates
        max_colors ([string]): List of names of all groups
        args (tuple): Necessary arguments, namely the list of colors, probabilities and size of groups

    Returns:
        SecretaryInstance: The selected candidate
    """
    
    colors, thresholds = args[0], args[1]
    max_until_threshold = [0] * len(colors)
    
    for i in range(len(candidates)):

        color_index = colors.index(candidates[i].color)
        if i < thresholds[color_index]:
            max_until_threshold[color_index] = max(max_until_threshold[color_index], candidates[i].score)

        if i >= thresholds[color_index] and candidates[i].score >= max_until_threshold[color_index]:
            candidates[i].ismax = candidates[i].score == max_colors[candidates[i].color]
            return candidates[i]

    return SecretaryInstance(-1, -1, None)

def multiple_color_thresholds(p) -> typing.List[float]:
    """Helper function for the fair opt algorithm. Receives probabilities and converts them to threshold

    Args:
        p ([float]): The groups probability of being selected

    Returns:
        t ([float]): A percentage threshold to be used in the main algorithm
    """
    
    t = [0.0] * len(p)
    k = len(p)
    
    t[k-1] = np.power((1 - (k - 1) * p[k - 1]), (1 / (k - 1)))
    
    for j in range(k-2, 0, -1):
        
        sum = np.sum([p[r] for r in range(0, j+1)])
        sum /= j
        t[j] = t[j+1] * np.power((sum - p[j]) / (sum - p[j+1]), 1 / j)

    t[0] = t[1] * np.exp(p[1] / p[0] - 1)

    return t
