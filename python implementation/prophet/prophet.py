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
# # The Multi-Color Prophet Problem notebook

# %%
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.stats import uniform
from numpy.random import default_rng
from collections import Counter
import matplotlib.pyplot as plt
from statistics import mean
import pandas as pd
import math

from numpy import save
from numpy import load

from tqdm import tqdm # for the progress bar
import dataframe_image as dfi

# %% [markdown]
# ## Helper functions

# %%
"""
:returns: x_cum, probability, r_probability, choose
"""
def calculatePreMiddleBinomial():
    n = 1000
    p = 0.5 #probability of coin succeeding
    choose = np.zeros((n+1,n+1)) # creating 'choose' variable -> number of combinations per number of successes
    for i in range(n+1):
        choose[i,0] = 1

    for i in range(1,n+1):
        for j in range(1,n+1):
            choose[i,j] = choose[i-1,j-1] + choose[i-1,j]

    n_combinations = choose[-1]
    probability = np.ones((n+1)) #probability of n successes
    r_probability = np.ones((n+1)) #reverse probability of n failures

    for i in range(1,n+1):
        probability[i] = probability[i - 1] * p
        r_probability[i] = r_probability[i - 1] * (1 - p);
        
    #max dist --> chance of getting at least certain amount of successes after all candidates
    x = n_combinations * probability * np.flip(r_probability) #calculating p of i successes in one try, by multiplying the p of this many successes, this many failures and all combinations in which they could have occurred
    x_cum = np.flip(np.cumsum(np.flip(x))) #calculating cumulative probability (p of getting at least i successes in one try)
    return np.flip(np.cumsum(np.flip(x))), [probability, r_probability, choose]

"""
:param n: number of candidates
:param x_cum_prepared: probability
:returns: inverse of the given distribution given its probability
"""
def middleBinomial(n, x_cum_prepared):
    n_candidates = n
    max_dist = 1 - pow(1-x_cum_prepared, n_candidates) #p of getting i successes after certain amount of tries

    #middle --> find highest number of successes where probability of reaching at least that is more than 0.5
    for i in np.arange(len(max_dist)-1, -1,-1):
        if max_dist[i] >= 0.5:
            #TODO: figure out why in the code they do divde this by n!
            middle = i #/ n ### question: binomial data comes in absolute n successes or fraction of total tries ??? ###
            break
        if i == 0.0:
            middle = 0
    return middle


"""
:param distribution_type: either "uniform" or "binomial"
:param prob: probability
:returns: inverse of the given distribution given its probability
"""
def Finv(distribution, prob):
    lower, upper = 0.0,1.0
    if distribution == "uniform":
        return prob * (upper-lower)
    if distribution == "binomial":
        return scipy.stats.binom.ppf(prob, n=1000, p=0.5)
        
"""
:param distribution_type: either "uniform" or "binomial"
:param n_candidates: number of candidates
:returns x_cum: 
:returns: the middle of the specified distribution
"""
def Middle(distribution_type, n, x_cum):
    if distribution_type == "uniform":
        rrange = 1.0
        return rrange * np.power(1.0 / 2, 1.0 / n)
    if distribution_type == "binomial":
        return middleBinomial(n, x_cum)
        
"""
:param distribution_type: either "uniform" or "binomial"
:param n: number of candidates
:returns : 
"""
def Expected(lower_bound, precalculated):
    ans, rangge = 0.0 , 0.0
    n = 1000-1 
    probability_ , r_probability_ , choose_= precalculated

    for i in range(math.ceil(lower_bound * n), n-1, 1):
        ans += (probability_[i] * r_probability_[n - i] * choose_[n][i] * i) / n;
        rangge += probability_[i] * r_probability_[n - i] * choose_[n][i];
    return ans / rangge

"""
:param distribution_type: either "uniform" or "binomial"
:param n_candidates: number of candidates
:returns precalculated: 
:returns p_th_:
"""
def PThreshold(distribution_type, n_candidates, precalculated):
    if distribution_type == "uniform":
        V = [0.0]*n_candidates
        V[n_candidates - 1] = .5
        p_th_ = [0.0]*n_candidates
        for i in range(n_candidates-2, -1, -1):
            p_th_[i] = V[i+1]
            V[i] = (1 + p_th_[i]) / 2
        return p_th_
    if distribution_type == "binomial":
        V = [0.0]*n_candidates
        V[n_candidates - 1] = Expected(0, precalculated)
        p_th_ = [0.0]*n_candidates
        for i in range(n_candidates-2, -1, -1):
            p_th_[i] = V[i+1] #TOOD NOTE
            V[i] = Expected(p_th_[i], precalculated)
        return p_th_
    


# %%
def FairGeneralProphet (q, V, distribution_type):
    summ = 0.0
    for i in range(0,len(V)): 
        if V[i] >= Finv(distribution_type, (1.0 - (q[i] / (2.0 - summ)))):
            return i
        summ += q[i]

def FairIIDProphet(Values, distribution_type):
    for i in range(0, len(Values)):
        p = (2.0 / 3.0) / len(Values)
        if Values[i] >= Finv(distribution_type, (1.0 - p / (1.0 - p * i))):
            return i



# %%
# Implemented according to the function “ComputeSolutionOneHalf” in unfair-prophet.cc
def SC_algorithm(Values, distribution_type, x_cum):
    middleValue = Middle(distribution_type, len(Values), x_cum)
    for i in range(0, len(Values)):
        if Values[i] >= middleValue:
            return i


# Implemented according to the function “ComputeSolutionOneMinusOneE” in unfair-prophet.cc
def EHKS_algorithm(Values, distribution_type):
    threshold = Finv(distribution_type, (1.0 - (1.0 / len(Values))))
    for i in range(0, len(Values)):
        if Values[i] >= threshold:
            return i

# Implemented according to the function “ComputeSolutionDiffEq” in unfair-prophet.cc
def CFHOV_algorithm(Values, distribution_type):

    # These precomputed threshold originate from the original paper (Correa et al., 2021)
    diff_solution_50 = np.loadtxt("diff_solution_50.txt", delimiter=", ")
    diff_solution_1000 = np.loadtxt("diff_solution_1000.txt", delimiter=", ")
    diff_solution = diff_solution_50 if len(Values) == 50 else diff_solution_1000
        
    for i in range(0, len(Values)):
        if Values[i] >= Finv(distribution_type, np.power(diff_solution[i], (1.0 / (len(Values) - 1)))):
            return i
        
        
def DP_algorithm(Values, distribution_type, precalculated):
    pttth = PThreshold(distribution_type, len(Values), precalculated)
    for i in range(0, len(Values)):
        if Values[i] >= (pttth[i])*1000:
            return i


# %% [markdown]
# ## The experiments

# %%
"""
:param distribution_type: either "uniform" or "binomial"
:param size: number of candidates
:returns q: 
:returns V:
"""
def generateDistribution(distribution_type, n):
    rng = default_rng()
    if distribution_type == "uniform":
        q, V = [1/n] * n , rng.uniform(low=0.0, high=1.0, size=n)
    elif distribution_type == "binomial":
        q, V = [1/n] * n , rng.binomial(n=1000, p=.5, size=n)
    return q,V

"""
:param algorithm: string either "FairGeneralProphet", "FairIIDProphet", "SC", "EHKS", "CFHOV", "DP"
:param N_experimentReps: the number of times the algorithm needs to run
:param distribution_type: either "uniform" or "binomial"
:param n_candidates: interger with the number of candidates in each experiment
:returns arrivalPositionsChosen: array containing which candidate position was chosen
:returns chosenValues: array contraining the values of each picked/selected candidate
"""
def runExperiment(algorithm, N_experimentReps, distribution_type, n_candidates):
    arrivalPositionsChosen, chosenValues = [0]*n_candidates, []
#     precalculated = None
    
    if (algorithm == "SC" or algorithm == "DP") and (distribution_type == "binomial"):
        x_cummm, precalculated = calculatePreMiddleBinomial()
    else:
        x_cummm, precalculated = None, None
        
    for _ in tqdm(range(0, N_experimentReps)):
        q, Values = generateDistribution(distribution_type, n_candidates)
        
        if algorithm == "FairGeneralProphet":
            result = FairGeneralProphet(q, Values, distribution_type)
        elif algorithm == "FairIIDProphet":
            result = FairIIDProphet(Values, distribution_type)
        elif algorithm == "SC":
            result = SC_algorithm(Values, distribution_type, x_cummm)
        elif algorithm =="EHKS":
            result = EHKS_algorithm(Values, distribution_type)
        elif algorithm == "CFHOV":
            result = CFHOV_algorithm(Values, distribution_type)
        elif algorithm == "DP":
            result = DP_algorithm(Values, distribution_type, precalculated)
                
                
        if result != None:
            arrivalPositionsChosen[result] += 1
            chosenValues.append(Values[result])
            
        if result == None: chosenValues.append(0)
    return arrivalPositionsChosen, chosenValues

# %%
# Local tests -> remove before upload

runExperiment(algorithm="FairGeneralProphet", N_experimentReps=50, distribution_type="uniform", n_candidates=100)
runExperiment(algorithm="FairIIDProphet", N_experimentReps=50, distribution_type="uniform", n_candidates=100)
runExperiment(algorithm="SC", N_experimentReps=50, distribution_type="uniform", n_candidates=100)
runExperiment(algorithm="EHKS", N_experimentReps=50, distribution_type="uniform", n_candidates=100)
runExperiment(algorithm="CFHOV", N_experimentReps=50, distribution_type="uniform", n_candidates=100)
runExperiment(algorithm="DP", N_experimentReps=50, distribution_type="uniform", n_candidates=100)

runExperiment(algorithm="FairGeneralProphet", N_experimentReps=50, distribution_type="binomial", n_candidates=100)
runExperiment(algorithm="FairIIDProphet", N_experimentReps=50, distribution_type="binomial", n_candidates=100)
runExperiment(algorithm="SC", N_experimentReps=50, distribution_type="binomial", n_candidates=100)
runExperiment(algorithm="EHKS", N_experimentReps=50, distribution_type="binomial", n_candidates=100)
runExperiment(algorithm="CFHOV", N_experimentReps=50, distribution_type="binomial", n_candidates=100)
runExperiment(algorithm="DP", N_experimentReps=50, distribution_type="binomial", n_candidates=100)

# %%

# %%
