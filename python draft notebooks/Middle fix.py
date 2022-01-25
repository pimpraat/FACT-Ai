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

# %%
import numpy as np

# %%
# n = 7  #number of 'coin tosses' 
# p = 0.5 #probability of coin succeeding


# # creating 'choose' variable -> number of combinations per number of successes
# choose = np.zeros((n+1,n+1))
# for i in range(n+1):
#     choose[i,0] = 1

# for i in range(1,n+1):
#     for j in range(1,n+1):
#         choose[i,j] = choose[i-1,j-1] + choose[i-1,j]
        
# n_combinations = choose[-1]

# # print('total number of options:',np.sum(n_combinations))
# # print('number of options choose per success score:',n_combinations)

# probability = np.ones((n+1)) #probability of n successes
# r_probability = np.ones((n+1)) #reverse probability of n failures

# for i in range(1,n+1):
#     probability[i] = probability[i - 1] * p
#     r_probability[i] = r_probability[i - 1] * (1 - p);

# print('probability of n successes:',probability)
# print('probability of n failures:',r_probability)

# %%
elements = [0.1,0.2,0.3,0.4,0.5,0.6,0.8] #candidates/elements/tries


# print(middle)


def middleBinomial(n):
    # n = number of 'coin tosses' 
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
    max_dist = 1 - pow(1-x_cum, len(elements)) #p of getting i successes after certain amount of tries

    #middle --> find highest number of successes where probability of reaching at least that is more than 0.5
    for i in np.arange(len(max_dist)-1, -1,-1):
        if max_dist[i] >= 0.5:
            middle = i / n ### question: binomial data comes in absolute n successes or fraction of total tries ??? ###
            break
        if i == 0.0:
            middle = 0
    return middle

    
    

def compute_solution_onehalf(elements):
    for i,element in enumerate(elements):
        if elements[i] >= middleBinomial(len(elements)):
            return element
    return None
        

compute_solution_onehalf(elements)

# %%
