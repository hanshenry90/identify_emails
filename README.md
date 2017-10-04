Check that you have a working python installation, preferably version 2.6 or 2.7 (that’s the version that we use--other versions may work, but we can’t guarantee it). If you're using Python 3, you may find some more extensive revision of code to get things working at this time, so you may want to set up a Python 2 environment to work through the course materials.

We will use pip to install some packages. First get and install pip from here. Using pip, install a bunch of python packages:

go to your terminal line (don’t open python, just the command prompt)
install sklearn: pip install scikit-learn
-- for your reference, here’s the sklearn installation instructions
install natural language toolkit: pip install nltk
Get the Intro to Machine Learning source code. You will need git to clone the repository: git clone https://github.com/udacity/ud120-projects.git
You only have to do this once, the code base contains the starter code for all the mini-projects. Go into the tools/ directory, and run startup.py. It will first check for the python modules, then download and unzip a large dataset that we will use heavily later. The download/unzipping can take a while, but you don’t have to wait for it to finish in order to get started on Part 1.