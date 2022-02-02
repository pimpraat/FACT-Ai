# Introduction
The code for this project firstly divided in two seperate directories, '_c++_' which contains the runnable code (fixed by us), based on the code by authors of the original paper (Correa et al., 2021). The '_python implementation_' folder includes our own Python based implementation in two seperate directories.

## Dependencies
In order to run all files use either: 'pip install ipynb tqdm dataframe_image pickle typing' to install the less common libraries or use the supplied fact_ai.yml environment file to install all required dependencies in one go. This will make sure all required dependencies for all steps of this reproducibility study are installed (the actual algorithms, generating data, and analysis).

# Running the code 

To run the notebooks containing the results mentioned in the paper, run the following notebooks:
* in 'python implementation/secretary/Secretary_Evalulation.ipynb' jupyter notebook in order to run the results for all secretary experiments used in the paper.
* in 'python implementation/prophet/prophet_results.ipynb' jupyter notebook in order to run general results for the prophet experiments used in the paper.
* in 'python implementation/prophet/prophet_results_extension.ipynb' jupyter notebook in order to run  results for the prophet extension used in the paper.

_All of these notebooks contain only the neccesary calls to produce the results. Further implementations and notebooks to run the experiments can be found in the folders 'secretary' and 'prophet' respectively_.


# Running the C++ code
To run original c++ representation, run the main.cc file after uncommenting the wanted experiments in the main method. This will print the results in the command line.

# Set up
To allow for version control of jupyter notebooks, please run:

pip install jupytext
Having this package installed in -- combination with the jupytext.toml file already present in the master branch of the repo -- every time you save a notebook, automatically a copy will be created in .py format. From every created .py file can be reconstructed its original .ipynb notebook file. So in this way, we can push and pull our .py files, and recreate the notebooks locally.

To reconstruct a notebook from a .py file, run:

jupytext --to notebook notebook.py
