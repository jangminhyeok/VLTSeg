## Setting Up Your Virtual Environment for the Dataset Converters

To train the model, you will need to download the relevant datasets and, in most cases, reformat them to be compatible with MMSegmentation and the Cityscapes labels. To this end, we use the excellent guidance and converter scripts from the [HRDA repository](https://github.com/lhoyer/HRDA/tree/master?tab=readme-ov-file#setup-datasets). 

We have copied the scripts into this directory for ease of access. Please note that the converter scripts make use of various IO helper methods defined in an [older version of MMCV](https://github.com/open-mmlab/mmcv/tree/v1.3.7), that are no longer present in the latest version. As such, **you will need to set up a separate virtual environment to run the scripts**. The following instructions will guide you through that process.

```
### Create and activate the virtual environment
conda deactivate
conda clean --all
conda create --name hrda python=3.8 -y
conda activate hrda

### Install Requirements
pip install -r /PROJECTS/VLTSeg/tools/dataset_converters/hrda_requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.7
```

With this virtual environment established, you can now return to **[Preparing the Datasets](../../README.md#preparing-the-datasets)**