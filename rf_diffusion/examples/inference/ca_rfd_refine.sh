#!/bin/bash 

# Note 1: rf_diffusion needs to be accessible in your $PYTHONPATH
# Note 2: You will need to change the pdb path to some pdb you want to refine which was generated by ca rfdiffusion diffusion model
# Note 3: In this example, the inference.ligand flag is present to demonstrate that the diffusion output had 'mu2' as the ligand in it

# Hint: Look at the config yaml!

pdb='path/to/some_pdb_with_trb_file_next_to_it.pdb'
CKPT='/mnt/projects/ml/ca_rfd/BFF_3_w_new_conf.pt'

apptainer exec --nv ./exec/bakerlab_rf_diffusion_aa.sif python run_inference.py \
    --config-name=test_ca_rfd_refinement \
    inference.num_designs=2 \
    inference.input_pdb=$pdb \
    inference.ligand='mu2' \
    inference.ckpt_path=$CKPT
