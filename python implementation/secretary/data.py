import numpy as np
import pandas as pd


def get_synthetic_data(*args):
    
    colors, probabilities, n = args[0], args[1], args[2]   
    synthetic_data = {color: None for color in colors}

    for i in range(len(n)):
        synthetic_data[colors[i]] = [np.random.uniform(0, 1, n[i]), probabilities[i]]

    return synthetic_data

def get_bank_data(path, probabilities):
    
    bank_data = pd.read_csv(path, sep = ';')
    call_duration = bank_data[['age', 'duration']]

    colored_candidates = {
        "under 30": [call_duration.loc[call_duration['age'] <= 30]['duration'].values, probabilities[0]],
        "31-40": [call_duration.loc[(call_duration['age'] > 30) & (call_duration['age'] <= 40)]['duration'].values, probabilities[1]],
        "41-50": [call_duration.loc[(call_duration['age'] > 40) & (call_duration['age'] <= 50)]['duration'].values, probabilities[2]],
        "51-60": [call_duration.loc[(call_duration['age'] > 50) & (call_duration['age'] <= 60)]['duration'].values, probabilities[3]],
        "over 60": [call_duration.loc[call_duration['age'] > 60]['duration'].values, probabilities[4]]
    }
    
    return colored_candidates

def get_pokec_data(path):
    
    with open(path) as f:
        pokec_data = f.readlines()
    
    smallerlist = [l.split(',') for l in ','.join(pokec_data).split('\n\'',)]
    
    return 0
