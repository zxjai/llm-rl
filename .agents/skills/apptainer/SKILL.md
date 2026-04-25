# Apptainer


```bash 
# check existing versions
module avail apptainer

# load a version
module load apptainer/1.4.5
apptainer --version

# help page 
apptainer help

apptainer cache list
apptainer cache clean

# Option 1: pull 
# Warning: have enough diskusage before running
# Recommendation: check the image size before building https://hub.docker.com/r/vllm/vllm-openai-rocm/tags
# much faster to build in $SLURM_TMPDIR instead of over network file system
export APPTAINER_CACHEDIR=$PWD/.apptainer_cache
export APPTAINER_TMPDIR=$PWD/.apptainer_tmp
apptainer pull  docker://vllm/vllm-openai-rocm:nightly # vllm
apptainer pull docker://rocm/pytorch:latest  # torch

# Option 2 build (usually not needed, just pull)
apptainer build --sandbox rocm_torch/ docker://rocm/pytorch:latest # torch
apptainer shell --writable rocm_torch/

# Optinal but recommended: create a symlink of the apptainer image to your working directory 
ln -s <apptainer_path> <working_path>

# run image, add --rocm flag to use AMD GPUs
apptainer shell --rocm rocm_torch.sif

# execute command
apptainer exec --rocm rocm_torch.sif <command>

# to attach dirs use the flag
--bind <host_path>:<container_path>
```


For more options, see [apptainern docs](https://apptainer.org/docs/user/latest/quick_start.html).
