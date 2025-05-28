from icecream import ic
import subprocess

def get():
    o = subprocess.run('scontrol show hostnames "$SLURM_JOB_NODELIST"', capture_output=True, check=True, shell=True)
    nodes = o.stdout.decode().split()
    master_addr = sorted(nodes)[0]
    return master_addr
    
if __name__ == '__main__':
    ic(get())
