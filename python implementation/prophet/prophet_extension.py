# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Prophet extension

# %%
import matplotlib.pyplot as plt
import dataframe_image as dfi
import pandas as pd
import numpy as np
import datetime
import os
from ipynb.fs.defs.prophet import generate_distribution, finv
from scipy.stats import ttest_ind
from statistics import mean
from tqdm import tqdm


# %%
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



# %% [markdown]
# ## Running experiments

# %%
# Creating directory to save to
date_time = datetime.datetime.now().strftime("%d_%m_%H.%M")
path = os.path.join('images','extension',date_time)
if not os.path.exists(path):
    os.makedirs(path)

parameters_general_prophet = np.arange(0.25, 1.5, .25)
n_experiment_reps = 50000

# Fair general prophet Uniform distribution
grid_search(algorithm='FairGeneralProphet',
            n_experiment_reps = n_experiment_reps,
            distribution_type = 'uniform',
            n_candidates = 50,
            parameters = parameters_general_prophet,
            path = path)

# Fair general prophet Binomial distribution
grid_search(algorithm='FairGeneralProphet',
            n_experiment_reps = n_experiment_reps,
            distribution_type = 'binomial',
            n_candidates = 1000,
            parameters = parameters_general_prophet,
            path = path)

# Fair IID prophet Uniform distribution
parameters_fair_iid = np.arange(0.5, 1.3, 0.10)
## For clarity in the plots, we epsilon values 0.6, 0.8, or 1.1 from the grid search results
excluded_parameters = np.isin(parameters_fair_iid, [0.6, 0.8, 1.1], invert=True)
parameters_fair_iid = parameters_fair_iid[excluded_parameters]

# Fair IID prophet Uniform distribution
grid_search(algorithm='FairIIDProphet',
            n_experiment_reps = n_experiment_reps,
            distribution_type = 'uniform',
            n_candidates = 50,
            parameters = parameters_fair_iid,
            path = path)

# Fair IID prophet Binomial distribution
grid_search(algorithm='FairIIDProphet',
            n_experiment_reps = n_experiment_reps,
            distribution_type = 'binomial',
            n_candidates = 1000,
            parameters = parameters_fair_iid,
            path = path)

# %% [markdown]
# ## Running the significance test for the extension, uniform distribution

# %% [markdown]
# _Output is two lists of 10x the avgInclude for the two groups. Group 1 is PaperValue, Group 2 is our ExtensionValue, for both FairProphet and FairIID._

# %%
fair_general_prophet_paper, fair_general_prophet_extension = [], []
fair_IID_prophet_paper, fair_IID_prophet_extension = [], []

fair_general_prophet_paper_exclude, fair_general_prophet_extension_exclude = [], []
fair_IID_prophet_paper_exclude, fair_IID_prophet_extension_exclude = [], []

fair_general_prophet_paper_parameter = 1
fair_IID_prophet_paper_parameter = 1
fair_general_prophet_extension_parameter = .5
fair_IID_prophet_extension_parameter = .7

for i in range(10):
    _, avg_include, avg_exclude, _ = run_experiment_extended(algorithm="FairGeneralProphet", 
                                                             n_experiment_reps=50000,
                                                             distribution_type="uniform", 
                                                             n_candidates=50, 
                                                             epsilon=fair_general_prophet_paper_parameter)
    fair_general_prophet_paper.append(avg_include)
    fair_general_prophet_paper_exclude.append(avg_exclude)
    
    _, avg_include, avg_exclude, _ = run_experiment_extended(algorithm="FairGeneralProphet", 
                                                             n_experiment_reps=50000,
                                                             distribution_type="uniform", 
                                                             n_candidates=50, 
                                                             epsilon=fair_general_prophet_extension_parameter)
    fair_general_prophet_extension.append(avg_include)
    fair_general_prophet_extension_exclude.append(avg_exclude)
    
    
for i in range(10):
    _, avg_include, avg_exclude, _ = run_experiment_extended(algorithm="FairIIDProphet", 
                                                             n_experiment_reps=50000,
                                                             distribution_type="uniform", 
                                                             n_candidates=50, 
                                                             epsilon=fair_IID_prophet_paper_parameter)
    fair_IID_prophet_paper.append(avg_include)
    fair_IID_prophet_paper_exclude.append(avg_exclude)
    
    _, avg_include, avg_exclude, _ = run_experiment_extended(algorithm="FairIIDProphet", 
                                                             n_experiment_reps=50000,
                                                             distribution_type="uniform", 
                                                             n_candidates=50, 
                                                             epsilon=fair_IID_prophet_extension_parameter)
    fair_IID_prophet_extension.append(avg_include)
    fair_IID_prophet_extension_exclude.append(avg_exclude)

# %%
print(ttest_ind(fair_general_prophet_paper, fair_general_prophet_extension))
print(ttest_ind(fair_IID_prophet_paper, fair_IID_prophet_extension))

# %%
print(ttest_ind(fair_general_prophet_paper_exclude, fair_general_prophet_extension_exclude))
print(ttest_ind(fair_IID_prophet_paper_exclude, fair_IID_prophet_extension_exclude))
