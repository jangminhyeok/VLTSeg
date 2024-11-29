## VLTSeg: A Baseline for Domain Generalized Dense Perception by CLIP-based Transfer Learning

**[ACCV2024 Paper](https://arxiv.org/pdf/2312.02021v3)**

This repo implements [VLTSeg](https://vltseg.github.io/) using the open-source framework [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). Note that we are not the authors of the paper. This reimplementation merely aims to remedy the fact that the code for VLTSeg is not currently publically available. You can use the repo to train your own version of the model [TODO: Add link to relevant section] or use one of the pretrained checkpoints available via Hugging Face (?) [TODO: Add link to relevant section].

## Setting Up Your Virtual Environment

To run this repository, you will need a machine with GCC 9.2.0, Python 3.8.x (we used 3.8.20) and a GPU compatbile with CUDA 12.1. 

We recommend setting up a new virtual environment, following the steps below. We will assume you wish to install the repo under `/PROJECTS/VLTSeg`, keep your checkpoints under `/CHECKPOINTS` and keep your datasets under `/DATA`, so change all relevant paths to fit your workspace as required.

```
### Create and activate the virtual environment
conda deactivate
conda clean --all
conda create --name vltseg python=3.8 -y
conda activate vltseg

### Clone VLTSeg
cd /PROJECTS
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

To train the model, you will need to download the relevant datasets and, in most cases, reformat them to be compatible with MMSegmentation and the Cityscapes labels. To this end, we use the excellent guidance and converter scripts from the [HRDA repository](https://github.com/lhoyer/HRDA/tree/master?tab=readme-ov-file#setup-datasets). 

We have copied the scripts into [`tools/dataset_converters`](tools/dataset_converters) and use them in the following instructions. Please note that the converter scripts make use of various IO helper methods defined in an [older version of MMCV](https://github.com/open-mmlab/mmcv/tree/v1.3.7), that are no longer present in the latest version. As such, **you will need to set up a separate virtual environment to run the scripts**. You can find the instructions [here](tools/dataset_converters/README.md). Once you have done so, continue with the steps below.

```
### Activate the prepared virtual environment
conda deactivate
conda activate hrda
cd /DATA

### GTA
wget https://download.visinf.tu-darmstadt.de/data/from_games/data/[ID]_images.zip   # where [ID] ranges from 01 to 10
wget https://download.visinf.tu-darmstadt.de/data/from_games/data/[ID]_labels.zip   # where [ID] ranges from 01 to 10
unzip [FILENAME] -d gta                                                             # where [FILENAME] ranges over all 20 files
python /PROJECTS/VLTSeg/tools/dataset_converters/gta.py /DATA/gta --nproc 8

### SYNTHIA
wget --no-check-certificate http://synthia-dataset.cvc.uab.cat/SYNTHIA_RAND_CITYSCAPES.rar
unrar x SYNTHIA_RAND_CITYSCAPES.rar
mv RAND_CITYSCAPES synthia
python /PROJECTS/VLTSeg/tools/dataset_converters/synthia.py /DATA/synthia --nproc 8
```

It is worth pointing out that the unrar process for `SYNTHIA_RAND_CITYSCAPES.rar` may leave some of the ground truth images corrupted. In our case, this affected seven files, even after repeated attempts. If you find that the converter script fails to generate a matching `_labelTrainIds.png` for some of the files in `/DATA/synthia/GT/LABELS`, consider simply deleting the affected training images in both `/DATA/synthia/RGB` and `/DATA/synthia/GT/LABELS`. If you do not, the training script will stumble over the images with missing ground truth eventually, and crash.

```
### Cityscapes
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=[USERNAME]&password=[PASSWORD]&submit=Login' https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3
unzip [FILENAME] -d cityscapes                                                      # where [FILENAME] ranges over both files
python /PROJECTS/VLTSeg/tools/dataset_converters/cityscapes.py /DATA/cityscapes --nproc 8

### ACDC
# Manually download rgb_anon_trainvaltest.zip and gt_trainval.zip from https://acdc.vision.ee.ethz.ch/download
unzip rgb_anon_trainvaltest.zip -d /DATA/acdc
unzip gt_trainval.zip -d /DATA/acdc
# These five commands flatten the folder structure
rsync -a /DATA/acdc/rgb_anon/*/train/*/* /DATA/acdc/rgb_anon/train/
rsync -a /DATA/acdc/rgb_anon/*/val/*/* /DATA/acdc/rgb_anon/val/
rsync -a /DATA/acdc/rgb_anon/*/test/*/* /DATA/acdc/rgb_anon/test/
rsync -a /DATA/acdc/gt/*/train/*/*_labelTrainIds.png /DATA/acdc/gt/train/
rsync -a /DATA/acdc/gt/*/val/*/*_labelTrainIds.png /DATA/acdc/gt/val/
# These three commands create empty ground truth files for the test set images
rsync -a /DATA/acdc/rgb_anon/test/* /DATA/acdc/gt/test/
for f in /DATA/acdc/gt/test/*_rgb_anon.png; do mv "$f" "$(echo "$f" | sed s/_rgb_anon/_gt_labelTrainIds/)"; done
for f in /DATA/acdc/gt/test/*_gt_labelTrainIds.png; do scp /PROJECTS/VLTSeg/tools/dataset_converters/acdc_white.png "$f"; done

### BDD100k
wget https://dl.cv.ethz.ch/bdd100k/data/10k_images_test.zip
wget https://dl.cv.ethz.ch/bdd100k/data/10k_images_train.zip
wget https://dl.cv.ethz.ch/bdd100k/data/10k_images_val.zip
wget https://dl.cv.ethz.ch/bdd100k/data/bdd100k_sem_seg_labels_trainval.zip
unzip [FILENAME]                                                                    # where [FILENAME] ranges over all 4 files

### Mapillary
# Manually download v1.2 from https://www.mapillary.com/dataset/vistas
unzip mapillary-vistas-dataset_public_v1.2.zip -d /DATA/mapillary
python /PROJECTS/VLTSeg/tools/dataset_converters/mapillary.py /DATA/mapillary/training --nproc 8
python /PROJECTS/VLTSeg/tools/dataset_converters/mapillary.py /DATA/mapillary/validation --nproc 8
```

## Training & Testing the Model

[TODO: Show example usage of both the direct dist_train.sh utility and the slurm script]

## Checkpoints

[TODO: Create a table with performance of various experiments and download links for the checkpoints]