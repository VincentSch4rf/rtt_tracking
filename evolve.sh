#!/bin/sh
#SBATCH --partition=gpu
#SBATCH --mem=180G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm.%j.out    # filename for STDOUT (%N: nodename, %j: job-ID)
#SBATCH --error=slurm.%j.err     # filename for STDERR

# module load cuda

cd /home/vschar2s/bitbots/rtt_tracking || exit

python ./yolov5/train.py \
--img 640 \
--batch 32 \
--epochs 50 \
--data atwork.yaml \
--weights yolov5s.pt \
--cache \
--evolve