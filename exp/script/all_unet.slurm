#!/bin/bash
#SBATCH --partition deeplearn
#SBATCH --qos gpgpudeeplearn
#SBATCH -A punim1553
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0-24:00:00
#SBATCH --output=../slurm_log/slurm-%j.out

module load foss/2022a CUDA/11.7.0 UCX-CUDA/1.13.1-CUDA-11.7.0 cuDNN/8.4.1.50-CUDA-11.7.0
module load Anaconda3/2022.10
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda deactivate
conda activate pt37

srun ./all_unet.sh
