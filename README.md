## VLTSeg: A Baseline for Domain Generalized Dense Perception by CLIP-based Transfer Learning

**[[ACCV2024 Paper]](https://arxiv.org/pdf/2312.02021v3)**

This repo implements [VLTSeg](https://vltseg.github.io/) using the open-source framework [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). Note that we are not the authors of the paper. This reimplementation merely aims to remedy the fact that the code for VLTSeg is not currently publically available. You can use the repo to train your own version of the model [TODO: Add link to relevant section] or use one of the pretrained checkpoints available via Hugging Face (?) [TODO: Add link to relevant section].

## Checkpoints

[TODO: Create a table with performance of various experiments and download links for the checkpoints]

## Setting Up Your Virtual Environment

To run this repository, you will need a machine with GCC 9.2.0, Python 3.8.x (we used 3.8.20) and a GPU compatbile with CUDA 12.1. 

We recommend setting up a new virtual environment, following the steps below. We will assume you wish to install the repo under `/PROJECTS/VLTSeg`, keep your checkpoints under `/CHECKPOINTS` and keep your datasets under `/DATA`, so change all relevant paths to fit your workspace as required.

```
cd /PROJECTS

### Create and activate the virtual environment
conda deactivate
conda clean --all
conda create --name vltseg python=3.8 -y
conda activate vltseg

### Clone VLTSeg
# [TODO: Change URL to GitHub version]
git clone https://git.tu-berlin.de/milz_at_tu-berlin/code-dev/vltseg.git VLTSeg

### Install requirements
pip install torch==2.3.0 torchvision xformers --index-url https://download.pytorch.org/whl/cu121
pip install -U openmim==0.3.9
mim install mmengine==0.10.5
mim install mmcv==2.1.0
pip install "mmdet>=3.0.0rc4"
pip install -r VLTSeg/requirements.txt

### Clone and install MMSegmentation from source
git clone -b v1.2.2 https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
mim install -e .
cd ..

### Install VLTSeg from source
cd VLTSeg
pip install -e .
ln -s /CHECKPOINTS /PROJECTS/VLTSeg/checkpoints
ln -s /DATA /PROJECTS/VLTSeg/data

### Download the pretrained EVA-CLIP weights
wget https://huggingface.co/QuanSun/EVA-CLIP/resolve/main/EVA02_CLIP_L_psz14_s4B.pt -P checkpoints

### Test the installation
# [TODO: Build nicer demo script]
python -i demo_composition.py
```

The symbolic links to `/CHECKPOINTS` and `/DATA` are required so that the config files can work with relative paths and find your checkpoints and datasets respectively, without modification. If you would rather avoid creating symbolic links, you need to modify the `pretrained` key in [`configs/_base_/models/eva-clip+mask2former.py`](configs/_base_/models/eva-clip+mask2former.py), as well as all `data_root` keys in all files under [`configs/_base_/datasets`](configs/_base_/datasets).

## Preparing the Datasets

[TODO: Explain how to download and prepare the datasets. Refer to both HRDA and maybe a separate Markdown file to set up the HRDA venv]

## Training & Testing the Model

[TODO: Show example usage of both the direct dist_train.sh utility and the slurm script]