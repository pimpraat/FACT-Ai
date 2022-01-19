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
import matplotlib as plt
from numpy.random import default_rng


# %% [markdown]
# We next consider the following multi-color prophet prob- lem. In this model n candidates arrive in uniform random order. Candidates are partitioned into k groups $C = {C_1,···,C_k}$. We write $n=(n1,...,nk)$ for the vector of groupsizes, i.e., $|C_j| = n_j$ ,for all 1 ≤ j ≤ k. We identify each of the groups with a distinct color and let c(i), vi denote the color and value of candidate i, respectively. The value vi that is revealed upon arrival of i, and is drawn independently from a given distribution Fi. We use $F = (F_1, . . . , Fn)$ to refer to the vector of distributions. We are also given a probability vector $p = (p_1, . . . , p_k)$. The goal is to select a candidate in an online manner in order to maximize the expectation of the value of the selected candidate, while selecting from each color with probability proportional to p. We distinguish between the basic setting in which $p_j$ is the proportion of candidates that belong to group j, i.e., $p_j = n_j/n$, and the general setting in which $p$ is arbitrary. We compare ourselves with the fair optimum, the optimal offline algorithm that respects the $p_j$ ’s.

# %% [markdown]
# ## The two algorithms presented in the paper

# %%
def Finv(prob):
    lower, upper = 0,1
    if distribution == "uniform":
        return lower + (upper - lower) * prob
    else:
        return scipy.stats.binom.ppf(prob, n=1000, p=0.5)
        

def FairGeneralProphet (F, q, V, distribution_type):
    """
    :param F: vector consisting of n distributions
    :param q: vector containing probabilities q1, . . . , qn with which FAIROPT accepts candidate 1, ... , n
    :param V: vector containing values for all candidates i, called from their F[i] distribution
    :returns: 
    """
    s = 0
    for i in range(0,len(V)):
        if V[i] > Finv(Vlower, Vupper, (1- (q[i]/2)/(1-(s/2)))
            return i
        s += q[i]



# %%
def FairIIDProphet(F, Values):
    """
    :param F:
    :returns: 
    """
    for i in range(0, len(Values)):
        if Values[i] >= 1 - ((2/3*len(Values))/(1-2*(i-1)/3*len(Values))):
            return i


# %% [markdown]
# ## The mentioned baseline algorithms

# %% [markdown]
# * SC algorithm (Samuel-Cahn, 1984)
#     * This algorithm sets a single threshold so that the maximum is above this threshold with probability exactly 1/2. 
# * EHKS algorithm (Ehsani et al., 2018)
#     * This algorithm sets a single threshold so that an individual candidate is accepted with probability 1/n
# * CFHOV algorithm (Correa et al., 2021)
#     * This algorithm sets a sequence of thresholds based on acceptance probabilities that result from solving a differential equation.
# * DP algorithm (e.g. Chow et al., 1971)
#     * This algorithm is the optimal threshold algorithm for the prophet problem, where thresholds are obtained by backward induction.

# %%
## TODO: Implement based on prepared pseudocode => ComputeSolutionOneHalf
def SamualCahn(n_candidates):
    """
    :param F:
    :returns: 
    """
    
    for i in range (0,n):
        
        
    
    pass


# %%
#TODO: Implement based on prepared pseudocode => ComputeSolutionOneMinusOneE
def EHKS():
    """
    :param F:
    :returns: 
    """
    pass


# %%
#TODO: Implement based on prepared pseudocode => ComputeSolutionThreeForth
def CFHOV():
    """
    :param F:
    :returns: 
    """
    pass


# %%
#TODO: Implement based on prepared pseudocode => ComputeSolutionDiffEq
def DP_algorithm():
    """
    :param F:
    :returns: 
    """
    pass


# %% [markdown]
# ## Some additional needed functions

# %% [markdown]
# We distinguish between the basic setting in which $p_j$ is the proportion of candidates that belong to group j, i.e., $p_j = n_j/n$, and the general setting in which p is arbitrary.

# %%
def generateProbabilityVector():
    """
    :returns p: probability vector with length j
    """
    return p


# Don't to use this?? Ask TA
def EFairOpt(n, C, F, p):
    """
    :param n: integer for the number of candidates
    :param C: list of groups
    :param F: vector of distribution (objects)
    :param p: probability vector
    :returns: 
    """
    pass


# %% [markdown]
# ## The experiments

# %% [markdown]
# *"We focus on the case, where values are distributed i.i.d. and each candidate is a group on its own. We consider two settings. In the first one the input stream consists of 50 samples from the uniform distribution in range [0, 1], and in the second one the input consists of 1000 samples from the binomial distribution with 1000 trials and 1/2 probability of success of a single trial. For better comparability with existing algorithms, in both cases we assume each candidate is a group on its own. We run each algorithm 50, 000 times."*

# %% [markdown]
# ### Uniform Distribution

# %%
rng = default_rng()
import matplotlib.pyplot as plt

# %%
# input stream consists of 50 samples from the uniform distribution in range [0, 1]
n = 50
k = n # since each candidate is a group on its own
C = range(0, n)
n_vector = [1] * n
q = [1/n] * n
gen_input = rng.uniform(low=0.0, high=1.0, size=n)



plt.scatter(gen_input)

# %% [markdown]
# ### Biniomial Distribution

# %%
# input consists of 1000 samples from the binomial distribution with 1000 trials 
# and 1/2 probability of success of a single trial
n = 1000
k = n # since each candidate is a group on its own

gen_input = rng.binomial(n=1000, p=.5, size=1000)
gen_input
plt.plot(

# %%
