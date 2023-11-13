# create new conda environment
conda create -n dhcp
conda activate dhcp

# install pytorch
conda install pytorch=1.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia

# install pytorch3d
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d=0.7.2 -c pytorch3d

# install dependencies
pip install pymeshlab tqdm scipy==1.10.1 nibabel==5.0.1 antspyx==0.3.8
