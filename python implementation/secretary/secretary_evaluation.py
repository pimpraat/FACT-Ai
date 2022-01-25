import pickle
from secretary_experiments import synthetic_experiment, unbalanced_synthetic_experiment

if __name__ == "__main__":
    
    results = synthetic_experiment()
    results2 = unbalanced_synthetic_experiment()
    
    with open('python implementation/results_synthetic1.pickle', 'wb') as f:
        pickle.dump(results, f)
        
    with open('python implementation/results_synthetic2.pickle', 'wb') as f:
        pickle.dump(results2, f)
