#!/net/software/containers/users/altaeth/rf_diffusion_aa_nucleic.sif

"""
Flattens and formats Hydra configurations with overrides into a single YAML file. It initializes Hydra,
composes configurations using specified defaults, and outputs a formatted YAML file with linebreaks between
top-level entries for readability.

Usage:
    ./flatten_config.py --config <config_name> --config_dir <config_directory> --output_file <output_filename>

Arguments:
    --config        Base config filename (without extension).
    --config_dir    Directory with Hydra config files.
    --output_file   Name of the output file for the flattened config.
"""

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import argparse
import os

def format_config_with_linebreaks(cfg):
    # Convert to dictionary for easier manipulation
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    # Format with linebreaks between top-level keys
    formatted_str = ""
    last_item = None
    for key, value in config_dict.items():
        if isinstance(value, dict):
            if isinstance(last_item, dict):
                formatted_str += f"{OmegaConf.to_yaml({key: value})}\n"
            else:
                formatted_str += f"\n{OmegaConf.to_yaml({key: value})}\n"
        else:
            formatted_str += f"{key}: {value}\n"
        
        last_item = value

    return formatted_str

def flatten_hydra_config(config_name, config_dir, output_file):
    # Initialize Hydra and add the configuration directory
    hydra.initialize(config_path=config_dir)
    cfg = hydra.compose(config_name=config_name)

    # Format the configuration with added linebreaks
    flattened_config = format_config_with_linebreaks(cfg)

    # Generate the full path for the output file
    output_path = os.path.join(config_dir, output_file)

    # Save the formatted configuration to the specified output file
    with open(output_path, 'w') as f:
        f.write(flattened_config)

    # Cleanup Hydra to avoid singleton conflict if re-initialized later
    GlobalHydra.instance().clear()

def main():
    parser = argparse.ArgumentParser(description="Flatten and format Hydra configurations with overrides.")
    parser.add_argument('--config', type=str, help='Base config file name')
    parser.add_argument('--config_dir', type=str, help='Directory containing Hydra configuration')
    parser.add_argument('--output_file', type=str, help='Output file name for the formatted configuration')
    
    args = parser.parse_args()

    flatten_hydra_config(args.config, args.config_dir, args.output_file)

if __name__ == '__main__':
    main()
