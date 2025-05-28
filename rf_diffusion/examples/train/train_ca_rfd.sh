#!/bin/bash 
# shell script for executing CA RFD diffusion model training

script='./train_multi_deep.py'

#####################
## Diffusion model ##
#####################
# This config specifically for the *diffusion* model (as opposed to refinement model)
CA_CFG='train_ca_rfd_diffusion_model'
# uncomment for training ca rfdiffusion diffusion model
# apptainer exec --nv exec/bakerlab_rf_diffusion_aa.sif $script --config-name=$CA_CFG


######################
## Refinement model ##
######################
# This config specifically for the *refinement* model (as opposed to the diffusion model)
CA_CFG_REFINE='train_ca_rfd_refinement_model'
apptainer exec --nv exec/bakerlab_rf_diffusion_aa.sif $script --config-name=$CA_CFG_REFINE
