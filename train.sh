#!/bin/bash
#SBATCH --partition=gpu4test
#SBATCH --mem=460G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm.%j.out    # filename for STDOUT (%N: nodename, %j: job-ID)
#SBATCH --error=slurm.%j.err     # filename for STDERR
#SBATCH --ntasks-per-node=120

module load cuda

NAME=rtt_376_1280_low_1500
cd /home/vschar2s/bitbots/rtt_tracking || exit

source /home/${USER}/.bashrc
conda activate yolo

export YOLOv5_VERBOSE=False

torchrun --standalone --nnodes=1 --nproc_per_node 4 ./yolov5/train.py \
--workers 8 \
--project out/train \
--name $NAME \
--img 1280 \
--batch 376 \
--epochs 1500 \
--hyp yolov5/data/hyps/hyp.scratch-low.yaml \
--data rtt.yaml \
--weights yolov5s.pt \
--cache

python ./yolov5/val.py \
--project out/val \
--name $NAME \
--task test \
--img 1280 \
--data rtt.yaml \
--weights out/train/$NAME/weights/best.pt

python ./yolov5/detect.py \
--project out/val \
--name $NAME \
--img 1280 \
--data rtt.yaml \
--weights out/train/$NAME/weights/best.pt \
--source /scratch/vschar2s/rtt/test/images \
--agnostic-nms

