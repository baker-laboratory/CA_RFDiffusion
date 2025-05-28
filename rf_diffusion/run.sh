#source activate /mnt/net/software/conda/envs/SE3nv
crop=384
max_len=384
max_complex_chain=250

export MASTER_PORT=36851

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)


python -u ./train_multi_deep.py -p_drop 0.15 -accum 16 -crop $crop -w_blen 0.1 -w_bang 0.1 -w_lj 0.1 -w_hb 0.0 -w_str 10.0 -maxlat 256 -maxseq 1024 -num_epochs 200 -lr 0.0005 -n_main_block 32 -seed 450 -seqid 150.0 -mintplt 1 -use_H -max_length $max_len -max_complex_chain $max_complex_chain -task_names diff -task_p 1.0 -interactive -diff_T 200
