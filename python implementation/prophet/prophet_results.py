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
# # Prophet results

# %% [markdown]
# _This notebook contains all the results used in the paper. For their imlplementation please refer to the prophet notebook itself. Furthermore please make sure to install ipynb by running:_ `pip install ipynb dataframe_image tqdm`
#

# %%
from ipynb.fs.defs.prophet import runExperiment
import matplotlib.pyplot as plt
from statistics import mean
from numpy import load, save
import dataframe_image as dfi
import matplotlib.pyplot as plt
import pandas as pd

# %%
# Generating and importing all required data

arrivalPositionsChosenFairPA_50k_uniform, FairPA_50k_uniform = runExperiment(algorithm="FairGeneralProphet", N_experimentReps=50000, 
                                                distribution_type="uniform", n_candidates=50)
    
arrivalPositionsChosenFairIID_50k_uniform, FairIID_50k_uniform = runExperiment(algorithm="FairIIDProphet", N_experimentReps=50000, 
                                                distribution_type="uniform", n_candidates=50)
    
arrivalPositionsChosenSC_50k_uniform, SC_50k_uniform = runExperiment(algorithm="SC", N_experimentReps=50000, 
                                                distribution_type="uniform", n_candidates=50)
    
arrivalPositionsChosenEHKS_50k_uniform, EHKS_50k_uniform = runExperiment(algorithm="EHKS", N_experimentReps=50000, 
                                                distribution_type="uniform", n_candidates=50)

arrivalPositionsChosenCFHOV_50k_uniform, CFHOV_50k_uniform = runExperiment(algorithm="CFHOV", N_experimentReps=50000, 
                                                distribution_type="uniform", n_candidates=50)

arrivalPositionsChosenDP_50k_uniform, DP_50k_uniform = runExperiment(algorithm="DP", N_experimentReps=50000, 
                                                distribution_type="uniform", n_candidates=50)

## Now 100k experiments

arrivalPositionsChosenFairPA_100k_uniform, FairPA_100k_uniform = runExperiment(algorithm="FairGeneralProphet", N_experimentReps=50000*2, 
                                                distribution_type="uniform", n_candidates=50)
    
arrivalPositionsChosenFairIID_100k_uniform, FairIID_100k_uniform = runExperiment(algorithm="FairIIDProphet", N_experimentReps=50000*2, 
                                                distribution_type="uniform", n_candidates=50)
    
arrivalPositionsChosenSC_100k_uniform, SC_100k_uniform = runExperiment(algorithm="SC", N_experimentReps=50000*2, 
                                                distribution_type="uniform", n_candidates=50)
    
arrivalPositionsChosenEHKS_100k_uniform, EHKS_100k_uniform = runExperiment(algorithm="EHKS", N_experimentReps=50000*2, 
                                                distribution_type="uniform", n_candidates=50)

arrivalPositionsChosenCFHOV_100k_uniform, CFHOV_100k_uniform = runExperiment(algorithm="CFHOV", N_experimentReps=50000*2, 
                                                distribution_type="uniform", n_candidates=50)

arrivalPositionsChosenDP_100k_uniform, DP_100k_uniform = runExperiment(algorithm="DP", N_experimentReps=50000*2, 
                                                distribution_type="uniform", n_candidates=50)

# All binomial cases are imported due to computational time
# please refer to the prohet_generate_data.ipynb for generation of this data

#Now importing the binomial cases 50k
arrivalPositionsChosenFairPA_50k_binomial, FairPA_50k_binomial = load('data/FairPA_positions.npy'), load('data/FairPA_values.npy')
arrivalPositionsChosenFairIID_50k_binomial, FairIID_50k_binomial = load('data/FairIID_positions.npy'), load('data/FairIID_values.npy')
arrivalPositionsChosenSC_50k_binomial, SC_50k_binomial = load('data/SC_positions.npy'), load('data/SC_values.npy')
arrivalPositionsChosenEHKS_50k_binomial, EHKS_50k_binomial = load('data/EHKS_positions.npy'), load('data/EHKS_values.npy')
arrivalPositionsChosenCFHOV_50k_binomial, CFHOV_50k_binomial = load('data/CFHOV_positions.npy'), load('data/CFHOV_values.npy')
arrivalPositionsChosenDP_50k_binomial, DP_50k_binomial = load('data/DP_positions.npy'), load('data/DP_values.npy')

#Now importing the binomial cases 10k
arrivalPositionsChosenFairPA_100k_binomial, FairPA_100k_binomial = load('data/FairPA_positions100k.npy'), load('data/FairPA_values100k.npy')
arrivalPositionsChosenFairIID_100k_binomial, FairIID_100k_binomial = load('data/FairIID_positions100k.npy'), load('data/FairIID_values100k.npy')
arrivalPositionsChosenSC_100k_binomial, SC_100k_binomial = load('data/SC_positions100k.npy'), load('data/SC_values100k.npy')
arrivalPositionsChosenEHKS_100k_binomial, EHKS_100k_binomial = load('data/EHKS_positions100k.npy'), load('data/EHKS_values100k.npy')
arrivalPositionsChosenCFHOV_100k_binomial, CFHOV_100k_binomial = load('data/CFHOV_positions100k.npy'), load('data/CFHOV_values100k.npy')
arrivalPositionsChosenDP_100k_binomial, DP_100k_binomial = load('data/DP_positions100k.npy'), load('data/DP_values100k.npy')


# %% [markdown]
# ## Generating plots for the prophet problem

# %%
plt.rcParams["figure.figsize"] = (8,3)
plt.subplots_adjust(bottom=0.25)
plt.rcParams.update({'font.size': 14})
plt.plot(range(0,50), arrivalPositionsChosenFairPA_50k_uniform, label="Fair PA")
plt.plot(range(0,50), arrivalPositionsChosenFairIID_50k_uniform, label="Fair IID")
plt.plot(range(0,50), arrivalPositionsChosenSC_50k_uniform, label="SC")
plt.plot(range(0,50), arrivalPositionsChosenEHKS_50k_uniform, label="EHKS")
# plt.plot(range(0,50), arrivalPositionsChosenDP_50k_uniform, label="DP")
plt.plot(range(0,50), arrivalPositionsChosenCFHOV_50k_uniform, label="CFHOV")

plt.grid(visible=True, linewidth=1)
plt.xlabel("Arrival position")
plt.ylabel("Num Picked")
plt.rcParams.update({'font.size': 12})
plt.legend(loc="upper left", ncol=5)
plt.savefig("images/uniform_distribution/50kExperiments_uniform.png")

# %%
plt.rcParams["figure.figsize"] = (8,3)
plt.subplots_adjust(bottom=0.25)
plt.rcParams.update({'font.size': 14})
plt.plot(range(0,50), arrivalPositionsChosenFairPA_100k_uniform, label="Fair PA")
plt.plot(range(0,50), arrivalPositionsChosenFairIID_100k_uniform, label="Fair IID")
plt.plot(range(0,50), arrivalPositionsChosenSC_100k_uniform, label="SC")
plt.plot(range(0,50), arrivalPositionsChosenEHKS_100k_uniform, label="EHKS")
plt.plot(range(0,50), arrivalPositionsChosenCFHOV_100k_uniform, label="CFHOV")
plt.grid(visible=True, linewidth=.5)
# plt.title("100k experiments, discarding None results")
plt.xlabel("Arrival position")
plt.ylabel("Num Picked")
plt.rcParams.update({'font.size': 12})
plt.legend(loc="upper left", ncol=5)
plt.savefig("images/uniform_distribution/100kExperiments_uniform.png")

# %%
plt.rcParams["figure.figsize"] = (8,3)
plt.subplots_adjust(bottom=0.25)
plt.rcParams.update({'font.size': 14})

plt.plot(range(0,1000), load('data/FairPA_positions.npy'), label="FairPA")
plt.plot(range(0,1000), load('data/FairIID_positions.npy'), label="Fair IID")
plt.plot(range(0,1000), load('data/SC_positions.npy'), label="SC")
plt.plot(range(0,1000), load('data/EHKS_positions.npy'), label="EHKS")
plt.plot(range(0,1000), load('data/CFHOV_positions.npy'), label="CFHOV")
# plt.plot(range(0,1000), load('data/DP_positions.npy'), label="DP")
plt.grid(visible=True, linewidth=1)
plt.xlabel("Arrival position")
plt.ylabel("Num Picked")
# plt.title("Binomial distribution with 1k candidates, and 50k experiments")
plt.rcParams.update({'font.size': 12})
plt.legend(loc="upper left", ncol=5)
plt.savefig("images/binomial_distribution/50kExperiments_binomial.png")

# %%
plt.rcParams["figure.figsize"] = (8,3)
plt.subplots_adjust(bottom=0.25)
plt.rcParams.update({'font.size': 14})
plt.plot(range(0,1000), load('data/FairPA_positions100k.npy'), label="FairPA")
plt.plot(range(0,1000), load('data/FairIID_positions100k.npy'), label="Fair IID")
plt.plot(range(0,1000), load('data/SC_positions100k.npy'), label="SC")
plt.plot(range(0,1000), load('data/EHKS_positions100k.npy'), label="EHKS")
plt.plot(range(0,1000), load('data/CFHOV_positions100k.npy'), label="CFHOV")
# plt.plot(range(0,1000), load('data/DP_positions100k.npy'), label="DP")
plt.grid(visible=True, linewidth=1)
plt.xlabel("Arrival position")
plt.ylabel("Num Picked")

plt.rcParams.update({'font.size': 12})
plt.legend(loc="upper left", ncol=5)
plt.savefig("images/binomial_distribution/100kExperiments_binomial.png")

# %% [markdown]
# # Analysis of the average values of the chosen candidates

# %%
print("Original : The average value of the chosen candidate for our Algorithm 2 (Fair PA), our Algorithm 3 (Fair IID), SC, EHKS, CFHOV, and DP for the uniform distribution is 0.501, 0.661, 0.499, 0.631, 0.752, 0.751, while for the binomial distribution it is 298.34, 389.24, 277.63, 363.97, 430.08, 513.34, respectively.")
print("Uniform distribution (original, 50k, 100k):")
print("FairPA: 0.501, ", round(mean(FairPA_50k_uniform),3), " , ", round(mean(FairPA_100k_uniform),3))
print("FairIID: 0.661, ", round(mean(FairIID_50k_uniform),3), " , ", round(mean(FairIID_100k_uniform),3))
print("SC: 0.499, ", round(mean(SC_50k_uniform),3), " , ", round(mean(SC_100k_uniform),3))
print("EHKS: 0.631, ", round(mean(EHKS_50k_uniform),3), " , ", round(mean(EHKS_100k_uniform),3))
print("CFHOV: 0.752, ", round(mean(CFHOV_50k_uniform),3), " , ", round(mean(CFHOV_100k_uniform),3))
print("DP: 0.751, ", round(mean(DP_50k_uniform),3), " , ", round(mean(DP_100k_uniform),3))

print("Binomial distribution:")
print("FairPA: 298.34, ", round(mean(FairPA_50k_binomial),3), " , ", round(mean(FairPA_100k_binomial),3))
print("FairIID: 389.24, ", round(mean(FairIID_50k_binomial),3), " , ", round(mean(FairIID_100k_binomial),3))
print("SC: 277.63, ", round(mean(SC_50k_binomial),3), " , ", round(mean(SC_100k_binomial),3))
print("EHKS: 363.97, ", round(mean(EHKS_50k_binomial),3), " , ", round(mean(EHKS_100k_binomial),3))
print("CFHOV: 430.08, ", round(mean(CFHOV_50k_binomial),3), " , ", round(mean(CFHOV_100k_binomial),3))
print("DP: 513.34, ", round(mean(DP_50k_binomial),3), " , ", round(mean(DP_100k_binomial),3))

# %%
print("Uniform distribution:")
df = pd.DataFrame(columns=['Algorithm', 'Reported in the paper', "Reproduced 50k experiments", "Reproduced 100k experiments"])

df = df.append(pd.Series(["FairPA","0.501",round(mean(FairPA_50k_uniform),3),round(mean(FairPA_100k_uniform),3)],
                         index = df.columns), ignore_index=True)
df = df.append(pd.Series(["FairIID","0.661",round(mean(FairIID_50k_uniform),3),round(mean(FairIID_100k_uniform),3)],
                         index = df.columns), ignore_index=True)
df = df.append(pd.Series(["SC","0.499",round(mean(SC_50k_uniform),3),round(mean(SC_100k_uniform),3)],
                         index = df.columns), ignore_index=True)
df = df.append(pd.Series(["EHKS","0.631",round(mean(EHKS_50k_uniform),3),round(mean(EHKS_100k_uniform),3)],
                         index = df.columns), ignore_index=True)
df = df.append(pd.Series(["CFHOV","0.752",round(mean(CFHOV_50k_uniform),3),round(mean(CFHOV_100k_uniform),3)],
                         index = df.columns), ignore_index=True)
df = df.append(pd.Series(["DP","0.751",round(mean(DP_50k_uniform),3),round(mean(DP_100k_uniform),3)],
                         index = df.columns), ignore_index=True)

df


# %%
print("Binomial distribution:")
df = pd.DataFrame(columns=['Algorithm', 'Reported in the paper', "Reproduced 50k experiments", "Reproduced 100k experiments"])

df = df.append(pd.Series(["FairPA","298.34",round(mean(FairPA_50k_binomial),3),round(mean(FairPA_100k_binomial),3)],
                         index = df.columns), ignore_index=True)
df = df.append(pd.Series(["FairIID","389.24",round(mean(FairIID_50k_binomial),3),round(mean(FairIID_100k_binomial),3)],
                         index = df.columns), ignore_index=True)
df = df.append(pd.Series(["SC","277.63",round(mean(SC_50k_binomial),3),round(mean(SC_100k_binomial),3)],
                         index = df.columns), ignore_index=True)
df = df.append(pd.Series(["EHKS","363.97",round(mean(EHKS_50k_binomial),3),round(mean(EHKS_100k_binomial),3)],
                         index = df.columns), ignore_index=True)
df = df.append(pd.Series(["CFHOV","430.08",round(mean(CFHOV_50k_binomial),3),round(mean(CFHOV_100k_binomial),3)],
                         index = df.columns), ignore_index=True)
df = df.append(pd.Series(["DP","513.34",round(mean(DP_50k_binomial),3),round(mean(DP_100k_binomial),3)],
                         index = df.columns), ignore_index=True)

df

# %% [markdown]
# # Percentage value of the optimal, but unfair, online algorithm

# %%
print("Original: In conclusion, for both settings, both our algorithms Algorithm 2 and Algorithm 3 provide perfect fairness, while giving 66.71% and 88.01% (for the uniform case), and 58.12% and 75.82% (for the binomial case), of the value of the optimal, but unfair, online algorithm.")

print("Uniform case, for FairPA")
print("Assuming DP as the 'optimal, but unfair, online algorithm' :", sum(FairPA_100k_uniform) / sum(CFHOV_100k_uniform) *100, "%")

print("\n Uniform case, for FairIID")
print("Assuming DP as the 'optimal, but unfair, online algorithm' :", sum(FairIID_100k_uniform) / sum(CFHOV_100k_uniform) * 100, "%")

print("Binomial case, for FairPA")
print("Assuming DP as the 'optimal, but unfair, online algorithm' :", sum(FairPA_100k_binomial) / sum(DP_100k_binomial) *100, "%")

print("\n Binomial case, for FairIID")
print("Assuming DP as the 'optimal, but unfair, online algorithm' :", sum(FairIID_100k_binomial) / sum(DP_100k_binomial) * 100, "%")

# %%
