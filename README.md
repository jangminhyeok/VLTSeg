# VLTSeg

## Introduction
This repo is an implementation of [VLTSeg](https://arxiv.org/abs/2312.02021) using the open-source framework MMSegmentation.

## Setting Up Your Virtual Environment
To run this repository, you will need a machine with CUDA installed. The following commands should get you started on our TU Berlin Math cluster.

```
salloc -p ggo --gres=gpu:nvidia_a100_2g.20gb:1 --mem=32g --cpus-per-task=6 --time=8:00:00
ssh node747
module load gcc/9.2.0

conda clean --all
conda create --name vltseg python=3.8 -y
conda activate vltseg

conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install xformers
pip install -U openmim
mim install mmengine
mim install "mmcv==2.1.0"
pip install "mmdet>=3.0.0rc4"
pip install ftfy
pip install regex

git clone https://github.com/open-mmlab/mmpretrain.git
cd mmpretrain
mim install -e .
cd ..

git clone -b main https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
mim install -e .
cd ..

git clone -b main https://git.tu-berlin.de/milz_at_tu-berlin/code-dev/vltseg.git
cd vltseg
```

TODO: Specify versions of all requirements, add requirements file.