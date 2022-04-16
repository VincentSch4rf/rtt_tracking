#!/bin/sh
#SBATCH --partition=gpu4test
#SBATCH --mem=460G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm.%j.out    # filename for STDOUT (%N: nodename, %j: job-ID)
#SBATCH --error=slurm.%j.err     # filename for STDERR
#SBATCH --nodelist=wr25

module load cuda

PROJECT=rtt_128_1280_high
cd /home/vschar2s/bitbots/rtt_tracking || exit

torchrun --standalone --nnodes=1 --nproc_per_node 4 ./yolov5/train.py \
--project out/train \
--name $PROJECT \
--img 1280 \
--batch 376 \
--epochs 300 \
--hyp yolov5/data/hyps/hyp.scratch-high.yaml \
--data rtt.yaml \
--weights yolov5s.pt

python ./yolov5/val.py \
--project out/val \
--name $PROJECT \
--task test \
--img 1280 \
--data rtt.yaml \
--weights out/train/$PROJECT/weights/best.pt

