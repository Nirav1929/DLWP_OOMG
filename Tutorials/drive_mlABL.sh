#!/bin/bash
#SBATCH -p ai2es_a100
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH --mem=40G 
#SBATCH --exclusive
#SBATCH --time=15:00:00
#SBATCH --chdir=/home/ablowe/DLWP-CS/Tutorials/
#SBATCH --job-name="train1"
#####SBATCH --job-name="predict"
#SBATCH --mail-user=ablowe@ncsu.edu
#SBATCH --mail-type=ALL
#SBATCH --mail-type=END
#SBATCH --output=./log_%x.out
#SBATCH --error=./error_%x.out


#   source my python env
source /home/ablowe/.bashrc
bash 
#module load Python/3.6.4-foss-2018a
#module load TensorFlow/1.8.0-foss-2018a-GPU-Python-3.6.4
#module load Keras/2.2.0-foss-2018a-Python-3.6.4
conda activate dlop

#mlt
#cd /home/ablowe/DLWP-CS/Tutorials/

python -u training_DLWP_model.py
#python -u predicting_DLWP_model.py


