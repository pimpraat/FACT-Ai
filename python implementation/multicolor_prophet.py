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

rng = default_rng()


# %% [markdown]
# _We next consider the following multi-color prophet problem. In this model n candidates arrive in uniform random order. Candidates are partitioned into k groups $C = {C_1,···,C_k}$. We write $n=(n1,...,nk)$ for the vector of groupsizes, i.e., $|C_j| = n_j$ ,for all 1 ≤ j ≤ k. We identify each of the groups with a distinct color and let c(i), vi denote the color and value of candidate i, respectively. The value vi that is revealed upon arrival of i, and is drawn independently from a given distribution Fi. We use $F = (F_1, . . . , Fn)$ to refer to the vector of distributions. We are also given a probability vector $p = (p_1, . . . , p_k)$. The goal is to select a candidate in an online manner in order to maximize the expectation of the value of the selected candidate, while selecting from each color with probability proportional to p. We distinguish between the basic setting in which $p_j$ is the proportion of candidates that belong to group j, i.e., $p_j = n_j/n$, and the general setting in which $p$ is arbitrary. We compare ourselves with the fair optimum, the optimal offline algorithm that respects the $p_j$ ’s._

# %% [markdown]
# ## The two algorithms presented in the paper

# %%
def Finv(distribution, prob):
    lower, upper = 0.0,1.0
    if distribution == "uniform":
        return prob * (upper-lower)
    if distribution == "binomial":
        return scipy.stats.binom.ppf(prob, n=1000, p=0.5)
        

## Is different for the Binomial!
def MiddleUniform(n):
    rrange = 1.0
    return rrange * np.power(1.0 / 2, 1.0 / n)
    
def MiddleBinomialDistribution(n):
    

#TODO: make class for the distributions!!
        
def FairGeneralProphet (q, V, distribution_type):
    summ = 0.0
    for i in range(0,len(V)):
        if V[i] >= Finv(distribution_type, (1.0 - (q[i] / (2 - summ)))):
#         if V[i] >= Finv(distribution_type, (1- (q[i]/2)/(1-(summ/2)))):
            return i
        summ += q[i]

def FairIIDProphet(Values, distribution_type):
    """
    :param F:
    :returns: 
    """
    for i in range(0, len(Values)):
        p = (2.0 / 3.0) / len(Values)
        if Values[i] >= Finv(distribution_type, (1.0 - p / (1.0 - p * i))):
            return i
#         n = len(Values)
#         deler = 2/(3*n)
#         noemer = 1 - 2*(i - 2) / (3*n)
#         if Values[i] >= Finv(distribution_type, (1 - deler/noemer)):
#             return i


# SC: second method “ComputeSolutionOneHalf”
def SC_algorithm(Values, distribution_type):
    for i in range(0, len(Values)):
        # distributions[elements[i].type].get().Middle(elements.size())
        if Values[i] >= MiddleUniform(len(Values)):
            return i


# EHKS: first method “ComputeSolutionOneMinusOneE”
def EHKS_algorithm(Values, distribution_type):
    for i in range(0, len(Values)):
        if Values[i] >= Finv(distribution_type, (1.0 - (1.0 / len(Values)))):
            return i



# %%
# This cell will contain both the SC algorithm and the EHKS algorithms as supporting baselines

# %% [markdown]
# ## The experiments

# %% [markdown]
# _"We focus on the case, where values are distributed i.i.d. and each candidate is a group on its own. We consider two settings. In the first one the input stream consists of 50 samples from the uniform distribution in range [0, 1], and in the second one the input consists of 1000 samples from the binomial distribution with 1000 trials and 1/2 probability of success of a single trial. For better comparability with existing algorithms, in both cases we assume each candidate is a group on its own. We run each algorithm 50, 000 times."_

# %% [markdown]
# ### Uniform Distribution

# %%
# input stream consists of 50 samples from the uniform distribution in range [0, 1]
n = 50
q, V = [1/n] * n , rng.uniform(low=0.0, high=1.0, size=n)

print(SC_algorithm(V, "uniform"))
print(EHKS_algorithm(V, "uniform"))

N_experimentReps = 50000*2
arrivalPositionsChosenFairPA, arrivalPositionsChosenFairIID = [0]*n, [0]*n
arrivalPositionsChosenSC, arrivalPositionsChosenEHKS = [0]*n, [0]*n
a,b,c,d = [],[],[],[]
# Running experiment for the FairPA
for _ in range(N_experimentReps):
    result = None
#     while (result == None):
    q, V = [1/n] * n , rng.uniform(low=0.0, high=1.0, size=n)
    result = FairGeneralProphet(q, V, "uniform")
    if result != None:
        arrivalPositionsChosenFairPA[FairGeneralProphet(q, V, "uniform")] += 1
        a.append(V[result])

# Running experiment for the FairIID
for _ in range(N_experimentReps):
    result = None
#     while (result == None):
    q, V = [1/n] * n , rng.uniform(low=0.0, high=1.0, size=n)
    result = FairIIDProphet(V, "uniform")
    if result != None:
        arrivalPositionsChosenFairIID[FairIIDProphet(V, "uniform")] += 1
        b.append(V[FairIIDProphet(V, "uniform")])
        
# Running experiment for the SC
for _ in range(N_experimentReps):
    result = None
#     while (result == None):
    q, V = [1/n] * n , rng.uniform(low=0.0, high=1.0, size=n)
    result = SC_algorithm(V, "uniform")
    if result != None:
        arrivalPositionsChosenSC[SC_algorithm(V, "uniform")] += 1
        c.append(V[SC_algorithm(V, "uniform")])
        
# Running experiment for the EHKS
for _ in range(N_experimentReps):
    result = None
#     while (result == None):
    q, V = [1/n] * n , rng.uniform(low=0.0, high=1.0, size=n)
    result = EHKS_algorithm(V, "uniform")
    if result != None:
        arrivalPositionsChosenEHKS[EHKS_algorithm(V, "uniform")] += 1
        d.append(V[EHKS_algorithm(V, "uniform")])
            

plt.plot(range(0,50), arrivalPositionsChosenFairPA, label="Fair PA")
plt.plot(range(0,50), arrivalPositionsChosenFairIID, label="Fair IID")
plt.plot(range(0,50), arrivalPositionsChosenEHKS, label="EHKS")
plt.plot(range(0,50), arrivalPositionsChosenSC, label="SC")
plt.plot(range(0,50), range(0,4000,80), label="replicate CFHOV for scale")
plt.title("100k experiments, discarding None results")
plt.xlabel("Arrival position")
plt.ylabel("Num Picked")
plt.legend(loc="upper right")

print("The average value of the chosen candidate in the uniform distribution: \n)")
print("FairPA: ", mean(a), "(should be 0.501)")
print("FairIID: ", mean(b), "(should be 0.661)")



# %% [markdown]
# This led us to examining two approaches.
#
# It is also worth noting that just skipping over None results does not lead to the same results, only increasing the number of experiments. As shown here

# %% [markdown]
# ## Binomial distribution

# %%
# input consists of 1000 samples from the binomial distribution with 1000 trials 
# and 1/2 probability of success of a single trial
n = 200
k = n # since each candidate is a group on its own
q, V = [1/n] * n , rng.binomial(n=1000, p=.5, size=n)

N_experimentReps = 2000
arrivalPositionsChosenFairPA, arrivalPositionsChosenFairIID = [0]*n, [0]*n

# Running experiment for the FairPA
for cnt in range(N_experimentReps):
#     print(cnt)
    if (cnt % 50 == 0) : print(cnt)
    q, V = [1/n] * n , rng.binomial(n=1000, p=.5, size=n)
    result = FairGeneralProphet(q, V, "binomial")
    if result != None:
        arrivalPositionsChosenFairPA[FairGeneralProphet(q, V, "binomial")] += 1
# print("Ran FairPA")

# Running experiment for the FairIID
for cnt in range(N_experimentReps):
#     print("2: ", cnt)
    if cnt % 50 : print(cnt)
    q, V = [1/n] * n , rng.binomial(n=1000, p=.5, size=n)
    if FairIIDProphet(V, "binomial") != None:
        arrivalPositionsChosenFairIID[FairIIDProphet(V, "binomial")] += 1
            
plt.plot(range(0,n), arrivalPositionsChosenFairPA, label="Fair PA")
plt.plot(range(0,n), arrivalPositionsChosenFairIID, label="Fair IID")
plt.xlabel("Arrival position")
plt.ylabel("Num Picked")
plt.legend(loc="upper right")

# %%
