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
# # Generating binomial experiment data

# %% [markdown]
# _This notebook has been used to generate the data-files for all binomial distribution experiments. We save them because creating them on the fly (in for example the prophet_results notebook) takes too long._
#
# _Please comment the lowest two blocks out to run the whole generation script. Please keep in mind that this can take > 20 hours_

# %%
from ipynb.fs.defs.prophet import generateDistribution, Finv
from numpy import save, load
from tqdm import tqdm # for the progress bar

# %%
# arrivalPositionsChosenFairPA, FairPA_values = runExperiment(algorithm="FairGeneralProphet", N_experimentReps=50000, 
#                                                 distribution_type="binomial", n_candidates=1000)
# save('data/FairPA_positions.npy', arrivalPositionsChosenFairPA)
# save('data/FairPA_values.npy', FairPA_values)

# arrivalPositionsChosenFairIID, FairIID_values = runExperiment(algorithm="FairIIDProphet", N_experimentReps=50000, 
#                                                 distribution_type="binomial", n_candidates=1000)
# save('data/FairIID_positions.npy', arrivalPositionsChosenFairIID)
# save('data/FairIID_values.npy', FairIID_values)

# arrivalPositionsChosenSC, SC_values = runExperiment(algorithm="SC", N_experimentReps=50000, 
#                                                distribution_type="binomial", n_candidates=1000)

# save('data/SC_positions.npy', arrivalPositionsChosenSC)
# save('data/SC_values.npy', SC_values)

# arrivalPositionsChosenEHKS, EHKS_values = runExperiment(algorithm="EHKS", N_experimentReps=50000, 
#                                                 distribution_type="binomial", n_candidates=1000)

# save('data/EHKS_positions.npy', arrivalPositionsChosenEHKS)
# save('data/EHKS_values.npy', EHKS_values)

# arrivalPositionsChosenCFHOV, CFHOV_values = runExperiment(algorithm="CFHOV", N_experimentReps=50000, 
#                                                 distribution_type="binomial", n_candidates=1000)

# save('data/CFHOV_positions.npy', arrivalPositionsChosenCFHOV)
# save('data/CFHOV_values.npy', CFHOV_values)

# arrivalPositionsChosenDP, DP_values = runExperiment(algorithm="DP", N_experimentReps=50000, 
#                                                 distribution_type="binomial", n_candidates=1000)

# save('data/DP_positions.npy', arrivalPositionsChosenDP)
# save('data/DP_values.npy', DP_values)



# %%
# arrivalPositionsChosenFairPA, FairPA_values = runExperiment(algorithm="FairGeneralProphet", N_experimentReps=50000*2, 
#                                                 distribution_type="binomial", n_candidates=1000)
# save('data/FairPA_positions100k.npy', arrivalPositionsChosenFairPA)
# save('data/FairPA_values100k.npy', FairPA_values)

# arrivalPositionsChosenFairIID, FairIID_values = runExperiment(algorithm="FairIIDProphet", N_experimentReps=50000*2, 
#                                                 distribution_type="binomial", n_candidates=1000)
# save('data/FairIID_positions100k.npy', arrivalPositionsChosenFairIID)
# save('data/FairIID_values100k.npy', FairIID_values)

# arrivalPositionsChosenSC, SC_values = runExperiment(algorithm="SC", N_experimentReps=50000*2, 
#                                                distribution_type="binomial", n_candidates=1000)

# save('data/SC_positions100k.npy', arrivalPositionsChosenSC)
# save('data/SC_values100k.npy', SC_values)

# arrivalPositionsChosenEHKS, EHKS_values = runExperiment(algorithm="EHKS", N_experimentReps=50000*2, 
#                                                 distribution_type="binomial", n_candidates=1000)

# save('data/EHKS_positions100k.npy', arrivalPositionsChosenEHKS)
# save('data/EHKS_values100k.npy', EHKS_values)

# arrivalPositionsChosenCFHOV, CFHOV_values = runExperiment(algorithm="CFHOV", N_experimentReps=50000*2, 
#                                                 distribution_type="binomial", n_candidates=1000)

# save('data/CFHOV_positions100k.npy', arrivalPositionsChosenCFHOV)
# save('data/CFHOV_values100k.npy', CFHOV_values)

# arrivalPositionsChosenDP, DP_values = runExperiment(algorithm="DP", N_experimentReps=50000*2, 
#                                                 distribution_type="binomial", n_candidates=1000)

# save('data/DP_positions100k.npy', arrivalPositionsChosenDP)
# save('data/DP_values100k.npy', DP_values)
