# Introduction
The '_python implementation_' folder includes our own Python based implementation in two seperate directories: one for the secretary and one for the prophet implementation.

## Dependencies
In order to run all files use either: 'pip install ipynb tqdm dataframe_image pickle typing' to install the less common libraries or use the supplied fact_ai.yml environment file to install all required dependencies in one go. This will make sure all required dependencies for all steps of this reproducibility study are installed (the actual algorithms, generating data, and analysis).

# Running the code

To run the notebooks containing the results mentioned in the paper, run the following notebooks:
* in 'python implementation/secretary/Secretary_Evalulation.ipynb' jupyter notebook in order to run the results for all secretary experiments used in the paper.
* in 'python implementation/prophet/prophet_results.ipynb' jupyter notebook in order to run general results for the prophet experiments used in the paper.
* in 'python implementation/prophet/prophet_results_extension.ipynb' jupyter notebook in order to run  results for the prophet extension used in the paper.

_All of these notebooks contain only the neccesary calls to produce the results. Further implementations and notebooks to run the experiments can be found in the folders 'secretary' and 'prophet' respectively_.

# For secretary_experiments.py
The notebook runs the two synthetic experiments and retrieves saved results for the real datasets. We do not run the real data experiments (by default) because 1) the data is too large to upload it on github and 2) the experiments take a long time

If you want to run the real data experiments, follow the next steps:
  1. Download the data
    a) For the Bank dataset, download bank_train.csv: https://www.kaggle.com/gauravduttakiit/portuguese-bank-telemarketing-data
    b) For the Pokec dataset, download soc-pokec-profilex.txt and soc-pokec-relationships.txt: https://www.kaggle.com/shakeelsial/pokec-full-data-set?select=soc-pokec-profiles.txt
    2. Rename the files:
      a) bank_raw.csv
      b) soc-pokec-profilex.txt, soc-pokec-relationships.txt
    3. Place the files in FACT-Ai\python implementation\secretary\data
    4. Open secretary_experiments.py and uncomment all experiments from the run_experiments() method (lines 128-129,137-147)
    5. Run the secretary_experiments_results.ipynb notebook
    6. Enjoy our findings!

# Running the C++ code
To run original c++ representation, run the main.cc file after uncommenting the wanted experiments in the main method. This will print the results in the command line.

# Set up
To allow for version control of jupyter notebooks, please run:

pip install jupytext
Having this package installed -- in combination with the jupytext.toml file already present in the master branch of the repo -- every time you save a notebook, a copy will automatically be created in .py format. From every created .py file, an equivalent .ipynb notebook file can be created. In this way, we can push and pull our .py files, and recreate the notebooks locally.

To reconstruct a notebook from a .py file, run:

jupytext --to notebook notebook.py
