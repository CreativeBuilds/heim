# Installation

## Create a virtual environment

It is recommended to install HEIM into a virtual environment with Python version 3.8 to avoid dependency conflicts. 
HEIM requires Python version 3.8. To create, a Python virtual environment with Python version >= 3.8 and activate it, 
follow the instructions below.

Using [virtualenv](https://docs.python.org/3/library/venv.html#creating-virtual-environments):

```
# Create a virtual environment.
# Only run this the first time.
python3 -m pip install virtualenv
python3 -m virtualenv -p python3.8 heim-venv

# Activate the virtual environment.
source heim-venv/bin/activate
```

Using [Anaconda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html):

```
# Create a virtual environment.
# Only run this the first time.
conda create -n crfm-heim python=3.8 pip

# Activate the virtual environment.
conda activate crfm-heim
```

## Install HEIM

Within this virtual environment, check out the [repository](https://github.com/stanford-crfm/heim) and 
run the [install script](https://github.com/stanford-crfm/heim/blob/main/pre-commit.sh).

```
git clone https://github.com/stanford-crfm/heim
cd heim
./pre-commit.sh
```
