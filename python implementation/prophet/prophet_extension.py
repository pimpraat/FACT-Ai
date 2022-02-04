import matplotlib.pyplot as plt
import dataframe_image as dfi
import pandas as pd
from ipynb.fs.defs.prophet import generate_distribution, finv
from statistics import mean
from tqdm import tqdm

def fair_general_prophet_extended(q, V, distribution_type, epsilon):
    """The Fair General Prophet algorithm

    Args:
        q (list): probability of picking a candidate within that group
        V (np.ndarray): list of values for each candidate
        distribution_type (string): string to indicate the distribution type
        epsilon (float): hyperparameter

    Returns:
        int: The index of the candidate
    """

    s = 0.0
    for i in range(len(V)):
        p = (1 - (q[i] / 2) / (epsilon - (s / 2)))
        if V[i] >= finv(distribution_type, p):
            return i
        s += q[i]

def fair_IID_prophet_extended(V, distribution_type, epsilon):
    """The Fair IID Prophet algorithm

    Args:
        values (np.ndarray): list of values for each candidate
        distribution_type (string): string to indicate the distribution type
        epsilon (float): hyperparameter

    Returns:
        int: The index of the candidate
    """   

    n = len(V)
    for i in range(n):
        p = 1 - (2 / (3 * n)) / (epsilon - 2 * (i - 1) / (3 * n))
        if V[i] >= finv(distribution_type, p):
                 return i
        
def run_experiment_extended(algorithm, n_experiment_reps, distribution_type, n_candidates, epsilon):
    """Runs the experiments with the algorithm specified

    Args:
        algorithm (string): either "FairGeneralProphet", "FairIIDProphet", "SC", "EHKS", "CFHOV", "DP"
        n_experiment_reps (int): the number of times the algorithm needs to run
        distribution_type (string): either "uniform" or "binomial"
        n_candidates (int): the number of candidates in each experiment
        epsilon (float): hyperparameter

    Returns:
        list: array containing which candidate position was chosen
        list: array contraining the values of each picked/selected candidate
    """    

    arrival_position, chosen_values, chosen_values_exclude_None = [0] * n_candidates, [], []
    nones = 0

    for _ in tqdm(range(n_experiment_reps)):

        q, values = generate_distribution(distribution_type, n_candidates)

        if algorithm == "FairGeneralProphet":
            result = fair_general_prophet_extended(q, values, distribution_type, epsilon)
        elif algorithm == "FairIIDProphet":
            result = fair_IID_prophet_extended(values, distribution_type, epsilon)
        if result != None:
            arrival_position[result] += 1
            chosen_values.append(values[result])
            chosen_values_exclude_None.append(values[result])
            
        if result == None: 
            chosen_values.append(0)
            nones += 1     
        
    none_rate = nones / n_experiment_reps
        
    return none_rate, mean(chosen_values), mean(chosen_values_exclude_None), arrival_position

def grid_search(algorithm, n_experiment_reps, distribution_type, n_candidates, parameters, path):
    """Runs a grid search for the optimal parameter epsilon

    Args:
        algorithm (string): either "FairGeneralProphet", "FairIIDProphet", "SC", "EHKS", "CFHOV", "DP"
        n_experiment_reps (int): the number of times the algorithm needs to run
        distribution_type (string): either "uniform" or "binomial"
        n_candidates (int): the number of candidates in each experiment
        parameters (list): list of values for the epsilon hyperparameter
        path (string): current directory
    """    

    df = pd.DataFrame(columns=['epsilon', 'None rate', "Mean value (None=0)", "Mean value (excluding None)"])
    print(algorithm, distribution_type)

    for param in parameters:
        if algorithm == 'FairIIDProphet':
            param = round(param, 1) # round epsilon in order to deal with float mistake in np.arange generation
        if algorithm == 'FairIIDProphet' and param in [0.6,0.8,1.1]:
            continue # For clarity in the plots, we epsilon values 0.6, 0.8, or 1.1 from the grid search results

        none_rate, avg_include, avg_exclude, chosen_positions = run_experiment_extended(algorithm=algorithm, 
                                                                                       n_experiment_reps=n_experiment_reps,
                                                                                       distribution_type=distribution_type, 
                                                                                       n_candidates=n_candidates, 
                                                                                       epsilon=param)

        a_series = pd.Series([param, none_rate, avg_include, avg_exclude], index = df.columns)
        df = df.append(a_series, ignore_index=True)
        plt.plot(range(n_candidates), chosen_positions, label= str("γ = " + str(param)))

    plt.xlabel("Arrival position")
    plt.ylabel("Num Picked")
    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=2)
    plt.tight_layout()
    plt.savefig("images/extensionFairPA_uniform.png")
    dfi.export(df, 'images/extensionFairPA_table_uniform.png')
    plt.show()
    print(df)
