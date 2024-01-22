#!/bin/bash
#SBATCH --job-name=SR_Topaz_V2e1
#SBATCH --account=project_465000269
#SBATCH --output=log/training_V2_essai1.o%j # Name of stdout output file
#SBATCH --error=log/training_V2_essai1.e%j  # Name of stderr error file
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --partition=small-g
#SBATCH --gpus=1

module load LUMI/22.08 partition/G
# module load rocm

singularity exec -B"/appl:/appl" \
                 -B"$SCRATCH:$SCRATCH" \
                 /scratch/project_465000269/bernigaud/env/tensorflow_rocm5.5-tf2.11-dev.sif ./python_env.sh training_attention_res_unet3V2_essai1.py
