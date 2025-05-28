#!/bin/bash 

# Note: Need the rf_diffusion/ repo root dir in PYTHONPATH, as well as `rf_diffusion/lib/se3_flow_matching`
export PYTHONPATH="${PYTHONPATH}:../"
export PYTHONPATH="${PYTHONPATH}:../lib/se3_flow_matching/"

###### CA RFdiffusion tests #####
## individual tests
#pytest test_ca_rfd_unconditional_train.py
#pytest test_ca_rfd_sm_train.py
#pytest test_ca_rfd_inference.py
#pytest test_ca_rfd_refinement.py

## run them all
pytest -k "test_ca_rfd"
