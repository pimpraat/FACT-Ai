import numpy as np
import scipy, scipy.stats
import typing
import math
from numpy.random import default_rng
from statistics import mean
from tqdm import tqdm

def calculate_premiddle_binomial() -> typing.Tuple[np.ndarray, typing.List[typing.Tuple]]:
    """Calculates the premiddle of a binomial distribution

    Returns:
        np.ndarray: precalculated cumulative distribution
        np.ndarray: the probability
        np.ndarray: r_probability
        np.ndarray: choose
    """    

    n, p = 1000, 0.5
    choose = np.zeros((n+1, n+1)) # creating 'choose' variable -> number of combinations per number of successes
    for i in range(n+1):
        choose[i, 0] = 1

    for i in range(1, n+1):
        for j in range(1, n+1):
            choose[i, j] = choose[i-1, j-1] + choose[i-1, j]

    n_combinations = choose[-1]
    probability, r_probability = np.ones((n+1)), np.ones((n+1))

    for i in range(1, n+1):
        probability[i] = probability[i - 1] * p
        r_probability[i] = r_probability[i - 1] * (1 - p)
        
    # max dist --> chance of getting at least certain amount of successes after all candidates
    x = n_combinations * probability * np.flip(r_probability) # calculating p of i successes in one try, by multiplying the p of this many successes, this many failures and all combinations in which they could have occurred
    
    return np.flip(np.cumsum(np.flip(x))), [probability, r_probability, choose]

def middle_binomial(n, x_cum_prepared) -> int:
    """Calculating the middle of a binomial distribution

    Args:
        n (int): number of candidates
        x_cum_prepared (float): probability

    Returns:
        int: inverse of the given distribution given its probability
    """    

    n_candidates = n
    max_dist = 1 - pow(1-x_cum_prepared, n_candidates) 

    # middle --> find highest number of successes where probability of reaching at least that is more than 0.5
    for i in np.arange(len(max_dist)-1, -1,-1):
        if max_dist[i] >= 0.5:
            middle = i
            break
        if i == 0.0:
            middle = 0
 
    return middle

def finv(distribution, prob) -> float:
    """Calculating the inverse of the given distribution using its probability

    Args:
        distribution (string): either "uniform" or "binomial"
        prob (float): probability

    Returns:
        float: inverse of the given distribution given its probability
    """
  
    lower, upper = 0.0, 1.0

    if distribution == "uniform":
        return prob * (upper-lower)
    if distribution == "binomial":
        return scipy.stats.binom.ppf(prob, n=1000, p=0.5)
        
def middle(distribution_type, n, x_cum) -> float:
    """Finding the middle of the given distribution

    Args:
        distribution_type (string): either "uniform" or "binomial"
        n (int): number of candidates
        x_cum (float): precalculated cumulative distribution

    Returns:
        float: the middle of the specified distribution
    """    

    if distribution_type == "uniform":
        rrange = 1.0
        return rrange * np.power(1.0 / 2, 1.0 / n)
    if distribution_type == "binomial":
        return middle_binomial(n, x_cum)

def expected(lower_bound, precalculated) -> float:
    """The expected (value) probability of the binomial

    Args:
        lower_bound (float): the lower bound
        precalculated (list): precalculated premiddle binomial

    Returns:
        float: the expected value of the probability
    """    
    
    ans, rangge = 0.0 , 0.0
    n = 1000 - 1 
    probability_ , r_probability_ , choose_= precalculated

    for i in range(math.ceil(lower_bound * n), n-1):
        ans += (probability_[i] * r_probability_[n - i] * choose_[n][i] * i) / n
        rangge += probability_[i] * r_probability_[n - i] * choose_[n][i]

    return ans / rangge

def p_threshold(distribution_type, n_candidates, precalculated) -> typing.List[float]:
    """Calculates the threshold for the distributions

    Args:
        distribution_type (string): either "uniform" or "binomial"
        n_candidates (int): number of candidates
        precalculated (list): precalculated premiddle binomial

    Returns:
        list: threshold pth
    """    
    
    if distribution_type == "uniform":
        
        V = [0.0] * n_candidates
        V[n_candidates - 1] = .5
        p_th_ = [0.0] * n_candidates
        for i in range(n_candidates - 2, -1, -1):
            p_th_[i] = V[i+1]
            V[i] = (1.0 + p_th_[i]) / 2
            
        return p_th_
    
    if distribution_type == "binomial":

        V = [0.0] * n_candidates
        V[n_candidates - 1] = expected(0, precalculated)
        p_th_ = [0.0]*n_candidates
        for i in range(n_candidates - 2, -1, -1):
            p_th_[i] = V[i+1]
            V[i] = expected(p_th_[i], precalculated)

        return p_th_
    
def fair_general_prophet(q, V, distribution_type) -> int:
    """The Fair General Prophet algorithm

    Args:
        q (list): probability of picking a candidate within that group
        V (np.ndarray): list of values for each candidate
        distribution_type (string): string to indicate the distribution type

    Returns:
        int: The index of the candidate
    """

    summ = 0.0

    for i in range(len(V)): 
        if V[i] >= finv(distribution_type, (1.0 - (q[i] / (2.0 - summ)))):
            return i
        summ += q[i]

def fair_IID_prophet(values, distribution_type) -> int:
    """The Fair IID Prophet algorithm

    Args:
        values (np.ndarray): list of values for each candidate
        distribution_type (string): string to indicate the distribution type

    Returns:
        int: The index of the candidate
    """    

    for i in range(len(values)):
        p = (2.0 / 3.0) / len(values)
        if values[i] >= finv(distribution_type, (1.0 - p / (1.0 - p * i))):
            return i

def SC_algorithm(values, distribution_type, x_cum) -> int:
    """The SC algorithm

    Args:
        values (np.ndarray): list of values for each candidate
        distribution_type (string): string to indicate the distribution type
        x_cum (float): probability

    Returns:
        int: The index of the candidate
    """     

    middle_value = middle(distribution_type, len(values), x_cum)

    for i in range(len(values)):
        if values[i] >= middle_value:
            return i

def EHKS_algorithm(values, distribution_type) -> int:
    """The EHKS algorithm

    Args:
        values (np.ndarray): list of values for each candidate
        distribution_type (string): string to indicate the distribution type

    Returns:
        int: The index of the candidate
    """   
    
    threshold = finv(distribution_type, (1.0 - (1.0 / len(values))))

    for i in range(len(values)):
        if values[i] >= threshold:
            return i

def CFHOV_algorithm(values, distribution_type) -> int:
    """The CFHOV algorithm

    Args:
        values (np.ndarray): list of values for each candidate
        distribution_type (string): string to indicate the distribution type

    Returns:
        int: The index of the candidate
    """  
   
    # These precomputed threshold originate from the original paper (Correa et al., 2021)
    diff_solution_50 = np.loadtxt("diff_solution_50.txt", delimiter=", ")
    diff_solution_1000 = np.loadtxt("diff_solution_1000.txt", delimiter=", ")
    diff_solution = diff_solution_50 if len(values) == 50 else diff_solution_1000
        
    for i in range(len(values)):
        if values[i] >= finv(distribution_type, np.power(diff_solution[i], (1.0 / (len(values) - 1)))):
            return i
        
def DP_algorithm(values, distribution_type, precalculated) -> int:
    """The DP algorithm

    Args:
        values (np.ndarray): list of values for each candidate
        distribution_type (string): string to indicate the distribution type
        precalculated (NoneType): value of premiddle binomial

    Returns:
        int: The index of the candidate
    """  

    pth = p_threshold(distribution_type, len(values), precalculated)

    for i in range(len(values)):
        if distribution_type == "uniform":
            if values[i] >= (pth[i]):
                return i
        if distribution_type == "binomial":
            if values[i] >= (pth[i]) * 1000:
                return i

def generate_distribution(distribution_type, n) -> typing.Tuple[typing.List, np.ndarray]:
    """[summary]

    Args:
        distribution_type (string): either "uniform" or "binomial"
        n (int): number of candidates

    Returns:
        list: list of probabilities
        np.ndarray: list of values for each candidate
    """    
    
    rng = default_rng()

    if distribution_type == "uniform":
        q, V = [1 / n] * n , rng.uniform(low=0.0, high=1.0, size=n)
    elif distribution_type == "binomial":
        q, V = [1 / n] * n , rng.binomial(n=1000, p=.5, size=n)

    return q, V

def run_experiment(algorithm, n_experiment_reps, distribution_type, n_candidates) -> typing.Tuple[typing.List, typing.List]:
    """Runs the experiments with the algorithm specified

    Args:
        algorithm (string): either "FairGeneralProphet", "FairIIDProphet", "SC", "EHKS", "CFHOV", "DP"
        n_experiment_reps (int): the number of times the algorithm needs to run
        distribution_type (string): either "uniform" or "binomial"
        n_candidates (int): the number of candidates in each experiment

    Returns:
        list: array containing which candidate position was chosen
        list: array contraining the values of each picked/selected candidate
    """    
    
    arrival_position, chosen_values = [0] * n_candidates, []
    
    if (algorithm == "SC" or algorithm == "DP") and (distribution_type == "binomial"):
        x_cum, precalculated = calculate_premiddle_binomial()
    else:
        x_cum, precalculated = None, None
        
    for _ in tqdm(range(n_experiment_reps)):

        q, values = generate_distribution(distribution_type, n_candidates)
        if algorithm == "FairGeneralProphet":
            result = fair_general_prophet(q, values, distribution_type)
        elif algorithm == "FairIIDProphet":
            result = fair_IID_prophet(values, distribution_type)
        elif algorithm == "SC":
            result = SC_algorithm(values, distribution_type, x_cum)
        elif algorithm =="EHKS":
            result = EHKS_algorithm(values, distribution_type)
        elif algorithm == "CFHOV":
            result = CFHOV_algorithm(values, distribution_type)
        elif algorithm == "DP":
            result = DP_algorithm(values, distribution_type, precalculated)
                
        if result != None:
            arrival_position[result] += 1
            chosen_values.append(values[result])
            
        if result == None: chosen_values.append(0)
        
    return arrival_position, chosen_values


# Local tests -> remove before upload

run_experiment(algorithm="FairGeneralProphet", n_experiment_reps=50, distribution_type="uniform", n_candidates=100)
run_experiment(algorithm="FairIIDProphet", n_experiment_reps=50, distribution_type="uniform", n_candidates=100)
run_experiment(algorithm="SC", n_experiment_reps=50, distribution_type="uniform", n_candidates=100)
run_experiment(algorithm="EHKS", n_experiment_reps=50, distribution_type="uniform", n_candidates=100)
run_experiment(algorithm="CFHOV", n_experiment_reps=50, distribution_type="uniform", n_candidates=100)
run_experiment(algorithm="DP", n_experiment_reps=50, distribution_type="uniform", n_candidates=100)

run_experiment(algorithm="FairGeneralProphet", n_experiment_reps=50, distribution_type="binomial", n_candidates=100)
run_experiment(algorithm="FairIIDProphet", n_experiment_reps=50, distribution_type="binomial", n_candidates=100)
run_experiment(algorithm="SC", n_experiment_reps=50, distribution_type="binomial", n_candidates=100)
run_experiment(algorithm="EHKS", n_experiment_reps=50, distribution_type="binomial", n_candidates=100)
run_experiment(algorithm="CFHOV", n_experiment_reps=50, distribution_type="binomial", n_candidates=100)
run_experiment(algorithm="DP", n_experiment_reps=50, distribution_type="binomial", n_candidates=100)

# Temporary, for fixing the DP
position_DP_100k_uniform, DP_100k_uniform = run_experiment(algorithm="DP", n_experiment_reps=50000*2, 
                                                distribution_type="uniform", n_candidates=50)
mean(DP_100k_uniform)
