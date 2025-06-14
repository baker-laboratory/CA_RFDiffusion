#!/bin/bash 

# Note 1: rf_diffusion needs to be accessible in your $PYTHONPATH
# Note 2: You will need to change the pdb path to some pdb you want to refine which was generated by ca rfdiffusion diffusion model
# Note 3: In this example, the inference.ligand flag is present to demonstrate that the diffusion output had 'mu2' as the ligand in it

# Hint: Look at the config yaml!

export PYTHONPATH=$PYTHONPATH:../

# Note: you need to define these 
pdb="/path/to/your/diffused/backbone.pdb"
CKPT='/path/to/your/downloaded/ca_rfd_refinement.pt'

# This is an example of supplying extra flags to override the .yaml file. 
# Note that the inference.ligand needs to match your specific ligand in the pdb files.
# 'mu2' is here as an example because that's what the diffusion step example also uses.
python run_inference.py \
    --config-name=test_ca_rfd_refinement \
    inference.num_designs=4 \
    inference.input_pdb=$pdb \
    inference.ligand='mu2' \
    inference.ckpt_path=$CKPT
