#!/bin/sh
#SBATCH --partition=gpu
#SBATCH --mem=180G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm.%j.out    # filename for STDOUT (%N: nodename, %j: job-ID)
#SBATCH --error=slurm.%j.err     # filename for STDERR
#SBATCH --ntasks-per-node=60

 module load cuda

cd /home/vschar2s/bitbots/rtt_tracking || exit

source /home/${USER}/.bashrc
conda activate yolo

NAME=atwork_1280_50_300

python ./yolov5/train.py \
--workers 8 \
--project out/evolve \
--name $NAME \
--img 1280 \
--batch -1 \
--epochs 50 \
--data atwork.yaml \
--weights yolov5s.pt \
--cache \
--evolve 300