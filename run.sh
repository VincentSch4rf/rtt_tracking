#!/bin/sh
#SBATCH --partition=gpu4test
#SBATCH --mem=460G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm.%j.out    # filename for STDOUT (%N: nodename, %j: job-ID)
#SBATCH --error=slurm.%j.err     # filename for STDERR

module load cuda

PROJECT=atwork_128_1280_low
cd /home/vschar2s/bitbots/rtt_tracking || exit

torchrun --standalone --nnodes=1 --nproc_per_node 4 ./yolov5/train.py \
--project out/train \
--name $PROJECT \
--workers 2 \
--img 1280 \
--batch 128 \
--epochs 300 \
--hyp yolov5/data/hyps/hyp.scratch-low.yaml \
--data atwork.yaml \
--weights yolov5s.pt

python ./yolov5/val.py \
--project out/val \
--name $PROJECT \
--task test \
--img 1280 \
--data atwork.yaml \
--weights out/train/$PROJECT/weights/best.pt

