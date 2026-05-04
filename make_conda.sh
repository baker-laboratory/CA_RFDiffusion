#!/bin/bash 
set -e
# change to what you want 
ENV_NAME="ca_rfd_release"


# Create new conda env 
#conda create -n $ENV_NAME python=3.9 -y
conda create -n $ENV_NAME -y python=3.9 -c defaults -c conda-forge

# hook and activate 
# eval "$('conda' 'shell.bash' 'hook')"
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME 
conda info --envs
# Verify activation
#CURRENT_ENV=$(conda info --active-env-name)
if [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]; then
    echo "Failed to activate $ENV_NAME. Currently in: $CURRENT_ENV"
    exit 1
fi
#### Install packages #### 
# They are installed in stages

conda install -y \
   -c conda-forge \
   dm-tree=0.1.7 \
   pdbfixer=1.8.1 \
   mdtraj=1.9.7

# pytorch + dependancies
conda install -y \
   -c nvidia \
   -c nvidia/label/cuda-12.1.0 \
   -c pytorch \
   -c pyg \
   -c dglteam/label/cu121 \
   -c anaconda \
   -c conda-forge \
   pip \
   ipython=8.8.0 \
   "ipykernel>=6.22.0" \
   numpy=1.22 \
   pandas=1.5.2 \
   seaborn=0.12.2 \
   matplotlib \
   jupyterlab=3.5.0 \
   pytorch==2.2 \
   pytorch-cuda==12.1 \
   dgl==2.0.0.cu121 \
   pyg

conda install -y \
   -c conda-forge \
   openbabel=3.1.1 \
   einops=0.7.0

# pip extras
pip install \
   e3nn==0.5.1 \
   "hydra-core==1.3.1" \
   pyrsistent==0.19.3 \
   opt_einsum==3.3.0 \
   sympy==1.12 \
   omegaconf==2.3.0 \
   icecream==2.1.3 \
   wandb==0.13.10 \
   deepdiff==6.3.0 \
   assertpy==1.1 \
   biotite==0.36.1 \
   GPUtil==1.4.0 \
   addict==2.4.0 \
   fire==0.5.0 \
   tmtools==0.0.2 \
   plotly==5.16.1 \
   deepspeed==0.8.0 \
   biopython==1.80 \
   ipdb==0.13.11 \
   pytest==7.4.0 \
   mock==5.0.1 \
   openmm \
   colorlog \
   "ml-collections"

# Jax doesn't install nicely with other packages for reasons
pip install \
   jax==0.4.13

# Install git repos
pip install git+https://github.com/RalphMao/PyTimer.git

