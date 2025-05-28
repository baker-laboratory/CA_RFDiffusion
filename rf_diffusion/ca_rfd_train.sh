#!/bin/bash 
# shell script for executing CA RFD diffusion model training
# NOTE: ensure your conda environment is activated 
export PYTHONPATH=$PYTHONPATH:../
export PYTHONPATH="${PYTHONPATH}:/home/davidcj/manuscripts/serine_hydrolase/bakerlab_rfdiffusion/rf_diffusion_repo/lib/se3_flow_matching/"

script='./train_multi_deep.py'

#####################
## Diffusion model ##
#####################
CA_CFG='train_ca_rfd_diffusion_model'
python $script --config-name=$CA_CFG


######################
## Refinement model ##
######################
#
# uncomment stuff below for training refinement model
# CA_CFG_REFINE='train_ca_rfd_refinement_model'
# python $script --config-name $CA_CFG_REFINE
