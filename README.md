# Introduction
The files in the C++ folder are authored by authors of the original paper
(Correa et al., 2021). The python folder includes our Python based implementation.

# Set up
To allow for version control of jupyter notebooks, please run:

pip install jupytext
Having this package installed in -- combination with the jupytext.toml file already present in the master branch of the repo -- every time you save a notebook, automatically a copy will be created in .py format. From every created .py file can be reconstructed its original .ipynb notebook file. So in this way, we can push and pull our .py files, and recreate the notebooks locally.

To reconstruct a notebook from a .py file, run:

jupytext --to notebook notebook.py