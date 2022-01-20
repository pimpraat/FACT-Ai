import numpy as np
from collections import Counter

def evaluation(algorithm_name, results_SA, colors, n):
    
    chosen_scores = [item[0] for item in results_SA]
    chosen_colors = [item[1] for item in results_SA]
    is_max = [item[2] for item in results_SA]
    
    print('\n', 'Algorithm', algorithm_name, '; Group sizes:', n)
    print(Counter(chosen_colors))
    print(Counter(is_max))