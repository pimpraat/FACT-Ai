import numpy as np
import pandas as pd
from scipy.io import mmread

def create_data(colors, n, probabilities):

    colored_candidates = {color: None for color in colors}

    for i in range(len(n)):
        colored_candidates[colors[i]] = [np.random.uniform(0, 1, n[i]), probabilities[i]]
        
    return colored_candidates

def get_synthetic_data():
    
    colors = ['red', 'green', 'blue', 'yellow']
    n = [10, 100, 1000, 10000]
    
    probabilities = [0.25, 0.25, 0.25, 0.25]
    synthetic_data_equal_p = create_data(colors, n, probabilities)
    print(synthetic_data_equal_p)
    
    probabilities = [0.3, 0.25, 0.25, 0.2]
    synthetic_data_diff_p = create_data(colors, n, probabilities)
    print(synthetic_data_diff_p)
    
    return synthetic_data_equal_p, synthetic_data_diff_p

def get_bank_data(path):
    
    bank_data = pd.read_csv(path, sep = ';')
    feedback_max = bank_data[['age', 'duration']]

    colors = ['under 30', '31-40', '41-50', '51-60', 'over 60']
    probability = 0.2

    colored_candidates = {
        "under 30": [feedback_max.loc[feedback_max['age'] <= 30]['duration'].values, probability],
        "31-40": [feedback_max.loc[(feedback_max['age'] > 30) & (feedback_max['age'] <= 40)]['duration'].values, probability],
        "41-50": [feedback_max.loc[(feedback_max['age'] > 40) & (feedback_max['age'] <= 50)]['duration'].values, probability],
        "51-60": [feedback_max.loc[(feedback_max['age'] > 50) & (feedback_max['age'] <= 60)]['duration'].values, probability],
        "over 60": [feedback_max.loc[feedback_max['age'] > 60]['duration'].values, probability]
    }
    
    return colored_candidates

def get_pokec_data(path):
    
    print(path)
    with open(path) as f:
        pokec_data = f.readlines()

    # print(len(pokec_data))
    print(type(pokec_data))
    
    smallerlist = [l.split(',') for l in ','.join(pokec_data).split('\n\'',)]
    
    print(smallerlist[:100])
    
if __name__ == "__main__":
    
    # synthetic_data_equal_p, synthetic_data_diff_p = get_synthetic_data()

    # bank_data = get_bank_data(path = 'data/bank_raw.csv')
    get_pokec_data(path = 'data/soc-pokec-profiles.txt')
    
    