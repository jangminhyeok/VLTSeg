#!/bin/bash
#SBATCH --job-name=train_vltseg_distributed
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:2
#SBATCH --mem=100g
#SBATCH --time=100:00:00
#SBATCH --partition=ggo

source /PATH/TO/HOME/.bashrc    # to access your conda
conda activate vltseg           # prepare this virtual environment as described in the README
cd /PATH/TO/REPO                # ending in "/VLTSeg"

CONFIG="configs/mask2former_evaclip_2xb8_5k_gta2cityscapes.py"
NUM_GPUS=2
WORK_DIR="/PATH/TO/LARGE/PARTITION/WORK_DIR"
CHECKPOINT="/PATH/TO/LARGE/PARTITION/WORK_DIR_PRETRAIN/FILENAME.pth"

sh tools/dist_train.sh $CONFIG $NUM_GPUS --work-dir $WORK_DIR --cfg-options load_from=$CHECKPOINT

# To fine-tune from a previous checkpoint, add the parameter
# --cfg-options load_from=$CHECKPOINT

# To train on a single GPU, use
# python -u tools/train.py $CONFIG --work-dir $WORK_DIR