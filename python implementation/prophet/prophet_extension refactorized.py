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
from ipynb.fs.defs.prophet import generateDistribution, Finv
import pandas as pd
import numpy as np
from tqdm import tqdm # for the progress bar
from statistics import mean
import matplotlib.pyplot as plt
import dataframe_image as dfi
from scipy.stats import ttest_ind


# %%
def FairGeneralProphetExtended(q, V, distribution_type, epsilon):
    s,n = 0.0, len(V)
    for i in range(0,n): #value < 1 reaches a drop!
        p = (1- (q[i]/2)/(epsilon-(s/2)))
        if V[i] >= Finv(distribution_type, p):
            return i
        s += q[i]

def FairIIDProphetExtended(V, distribution_type, epsilon):
    n = len(V)
    for i in range(0, n):
        p = 1 - (2/(3*n)) / (epsilon - 2*(i-1)/(3*n))
        if V[i] >= Finv(distribution_type, p):
                 return i
        
def runExperimentExtended(algorithm, N_experimentReps, distribution_type, n_candidates, epsilon):
    arrivalPositionsChosen, chosenValues, chosenValuesExcludeNone = [0]*n_candidates, [], []
    nones = 0
    for _ in tqdm(range(0, N_experimentReps)):
        q, Values = generateDistribution(distribution_type, n_candidates)
        
        if algorithm == "FairGeneralProphet":
                result = FairGeneralProphetExtended(q, Values, distribution_type, epsilon)
        elif algorithm == "FairIIDProphet":
                result = FairIIDProphetExtended(Values, distribution_type, epsilon)
        if result != None:
            arrivalPositionsChosen[result] += 1
            chosenValues.append(Values[result])
            chosenValuesExcludeNone.append(Values[result])
            
        if result == None: 
            chosenValues.append(0)
            nones += 1     
        
    noneRate = nones/N_experimentReps
        
    return noneRate, mean(chosenValues), mean(chosenValuesExcludeNone), arrivalPositionsChosen

def grid_search(algorithm, N_experimentReps, distribution_type,n_candidates, parameters):
    df = pd.DataFrame(columns=['epsilon', 'None rate', "Mean value (None=0)", "Mean value (excluding None)"])
    
    print(algorithm,distribution_type)
    for param in parameters:
        if algorithm == 'FairIIDProphet':
            param = round(param,1) # round epsilon in order to deal with float mistake in np.arange generation
        
        nonerate, avg_include, avg_exclude, chosen_positions = runExperimentExtended(algorithm=algorithm, 
                                                                                     N_experimentReps=N_experimentReps,
                                                                                     distribution_type=distribution_type, 
                                                                                     n_candidates=50, 
                                                                                     epsilon=param
                                                                                    )

        df = df.append(pd.Series([param,nonerate,avg_include,avg_exclude], index = df.columns), ignore_index=True)

        plt.plot(range(0,50), chosen_positions, label= str("γ = " + str(param)))
    plt.xlabel("Arrival position")
    plt.ylabel("Num Picked")
    plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                    mode="expand", borderaxespad=0, ncol=2)
    plt.savefig("images/extensionFairPA_uniform.png")
    dfi.export(df, 'images/extensionFairPA_table_uniform.png')
    plt.show()
    return df



# %% [markdown]
# ## Running experiments

# %%
# Fair general prophet Uniform distribution
parameters_general_prophet = np.arange(0.25, 1.5, .25)

grid_search(algorithm='FairGeneralProphet',
            N_experimentReps = 5000,
            distribution_type = 'uniform',
            n_candidates = 50,
            parameters = parameters_general_prophet)

# Fair general prophet Binomial distribution
grid_search(algorithm='FairGeneralProphet',
            N_experimentReps = 50,
            distribution_type = 'binomial',
            n_candidates = 1000,
            parameters = parameters_general_prophet)


# Fair IID prophet Uniform distribution
parameters_fair_iid = np.arange(0.5, 1.3, 0.10)
## For clarity in the plots, we epsilon values 0.6, 0.8, or 1.1 from the grid search results
excluded_parameters = np.isin(parameters_fair_iid, [0.6, 0.8, 1.1], invert=True)
parameters_fair_iid = parameters_fair_iid[excluded_parameters]

# Fair IID prophet Uniform distribution
grid_search(algorithm='FairIIDProphet',
            N_experimentReps = 5000,
            distribution_type = 'uniform',
            n_candidates = 50,
            parameters = parameters_fair_iid)

# Fair IID prophet Binomial distribution
grid_search(algorithm='FairIIDProphet',
            N_experimentReps = 50,
            distribution_type = 'binomial',
            n_candidates = 1000,
            parameters = parameters_fair_iid)

# %% [markdown]
# ## Running the significance test for the extension, uniform distribution

# %% [markdown]
# _Output is two lists of 10x the avgInclude for the two groups. Group 1 is PaperValue, Group 2 is ourExtensionValue, for both FairProphet and FairIID._

# %%
FairProphetPA_PaperValue, FairProphetPA_ExtensionValue = [], []
FairProphetIID_PaperValue, FairProphetIID_ExtensionValue = [], []

FairProphetPA_PaperValue_exclude, FairProphetPA_ExtensionValue_exclude = [], []
FairProphetIID_PaperValue_exclude, FairProphetIID_ExtensionValue_exclude = [], []

FairProphetPA_PaperParameter = 1
FairProphetIID_PaperParameter = 1
FairProphetPA_ExtensionParameter = .5
FairProphetIID_ExtensionParameter = .7

for i in range(0, 10):
    _, avg_include, avg_exclude, _ = runExperimentExtended(algorithm="FairGeneralProphet", 
                                                                                     N_experimentReps=50000,
                                                                                     distribution_type="uniform", 
                                                                                     n_candidates=50, 
                                                                                     epsilon=FairProphetPA_PaperParameter
                                                                                    )
    FairProphetPA_PaperValue.append(avg_include)
    FairProphetPA_PaperValue_exclude.append(avg_exclude)
    
    _, avg_include, avg_exclude, _ = runExperimentExtended(algorithm="FairGeneralProphet", 
                                                                                     N_experimentReps=50000,
                                                                                     distribution_type="uniform", 
                                                                                     n_candidates=50, 
                                                                                     epsilon=FairProphetPA_ExtensionParameter
                                                                                    )
    FairProphetPA_ExtensionValue.append(avg_include)
    FairProphetPA_ExtensionValue_exclude.append(avg_exclude)
    
    
for i in range(0, 10):
    _, avg_include, avg_exclude, _ = runExperimentExtended(algorithm="FairIIDProphet", 
                                                                                     N_experimentReps=50000,
                                                                                     distribution_type="uniform", 
                                                                                     n_candidates=50, 
                                                                                     epsilon=FairProphetIID_PaperParameter
                                                                                    )
    FairProphetIID_PaperValue.append(avg_include)
    FairProphetIID_PaperValue_exclude.append(avg_exclude)
    
    _, avg_include, avg_exclude, _ = runExperimentExtended(algorithm="FairIIDProphet", 
                                                                                     N_experimentReps=50000,
                                                                                     distribution_type="uniform", 
                                                                                     n_candidates=50, 
                                                                                     epsilon=FairProphetIID_ExtensionParameter
                                                                                    )
    FairProphetIID_ExtensionValue.append(avg_include)
    FairProphetIID_ExtensionValue_exclude.append(avg_exclude)

# %%
print(ttest_ind(FairProphetPA_PaperValue, FairProphetPA_ExtensionValue))
print(ttest_ind(FairProphetIID_PaperValue, FairProphetIID_ExtensionValue))

# %%
print(ttest_ind(FairProphetPA_PaperValue_exclude, FairProphetPA_ExtensionValue_exclude))
print(ttest_ind(FairProphetIID_PaperValue_exclude, FairProphetIID_ExtensionValue_exclude))

# %%
