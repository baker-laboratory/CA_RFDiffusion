Bootstrap: docker
From: ubuntu
IncludeCmd: yes

%setup
rsync -a --no-g --no-o /home/dimaio/RoseTTAFold2/SE3Transformer/ $APPTAINER_ROOTFS/SE3Transformer/

%files
/etc/localtime
/etc/hosts
/etc/apt/sources.list
/home/dimaio/apptainer/Miniconda3-latest-Linux-x86_64.sh /opt/miniconda.sh
/home/dimaio/apptainer/cutlass-3.5.1 /opt/cutlass
#/home/dimaio/apptainer/biotite /opt/biotite

%post
## Switch shell to bash
rm /bin/sh; ln -s /bin/bash /bin/sh

# Common symlinks
ln -s /net/databases /databases
ln -s /net/software /software
ln -s /home /mnt/home
ln -s /projects /mnt/projects
ln -s /net /mnt/net

apt-get update

apt-get install -y g++-11
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 50
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 50
update-alternatives --install /usr/bin/c++ c++ /usr/bin/gcc-11 50

apt-get install -y libx11-6 libxau6 libxext6 libxrender1 libpng-dev libjpeg-dev #libtiff5
apt-get install -y git
apt-get install -y libaio-dev

apt-get install -y software-properties-common
add-apt-repository -y ppa:apptainer/ppa
apt-get update
apt-get install -y apptainer
apt-get clean

# Install conda
bash /opt/miniconda.sh -b -u -p /usr

# install other deps
conda install --yes \
   -c nvidia/label/cuda-12.4.0 \
   -c https://conda.rosettacommons.org \
   -c pytorch \
   -c dglteam/label/th24_cu124 \
   python==3.11 \
   pip \
   numpy"<2" \
   matplotlib \
   jupyterlab \
   dgl \
   conda-forge::openbabel==3.1.1 \
   cuda \
   pytorch==2.4 \
   pytorch-cuda==12.4 \
   pyrosetta \

pip install \
   hydra-core==1.3.1 \
   ml-collections==0.1.1 \
   addict==2.4.0 \
   assertpy==1.1.0 \
   biopython==1.83 \
   colorlog \
   compact-json \
   cython==3.0.0 \
   cytoolz==0.12.3 \
   debugpy==1.8.5 \
   deepdiff==6.3.0 \
   dm-tree==0.1.8 \
   e3nn==0.5.1 \
   einops==0.7.0 \
   executing==2.0.0 \
   fastparquet==2024.5.0 \
   fire==0.6.0 \
   GPUtil==1.4.0 \
   icecream==2.1.3 \
   ipdb==0.13.11 \
   ipykernel==6.29.5 \
   ipython==8.27.0 \
   ipywidgets \
   mdtraj==1.10.0 \
   numba \
   omegaconf==2.3.0 \
   opt_einsum==3.3.0 \
   pandas==2.2.3 \
   plotly==5.16.1 \
   pre-commit==3.7.1 \
   py3Dmol==2.2.1 \
   pyarrow==17.0.0 \
   pydantic \
   pyrsistent==0.19.3 \
   pytest-benchmark \
   pytest-cov==4.1.0 \
   pytest-dotenv==0.5.2 \
   pytest==8.2.0 \
   rdkit==2024.3.5 \
   RestrictedPython \
   ruff==0.6.2 \
   scipy==1.13.1 \
   seaborn==0.13.2 \
   sympy==1.13.2 \
   tmtools \
   tqdm==4.65.0 \
   typer==0.12.5 \
   wandb==0.13.10

# biotite fork
ln -s /usr/include/*.h /usr/local/include/
#pip install /opt/biotite/
pip install git+https://github.com/biotite-dev/biotite.git@fab175e7ba4608d9613f092ad4e080661c6cc816

# pyg
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

# for cuequivariant
pip install -U -i https://pypi.anaconda.org/rapidsai-wheels-nightly/simple "pylibcugraphops-cu12>=24.6.0a24" 

# deepspeed
pip install deepspeed==0.15.1

# Install git repos
pip install git+https://github.com/RalphMao/PyTimer.git
pip install git+https://github.com/baker-laboratory/ipd.git
pip uninstall -y ipd

# Run the biotite setup command
# (Temporary measure until we switch to released Biotite version)
python -m biotite.setup_ccd

# Clean up
conda clean -a -y
apt-get -y autoremove
apt-get clean
# rm /opt/miniconda.sh

%environment
export PATH=$PATH:/usr/local/cuda/bin
export CUTLASS_PATH=/opt/cutlass/
export MKL_SERVICE_FORCE_INTEL=1

%runscript

exec python "$@"

%help
SE3nv environment for running RF-diffusion, etc.
