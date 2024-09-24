# Unpaved Road Segmentation of UAV Imagery via a Global Vision Transformer with Dilated Cross Window Self-Attention for Dynamic Map


## Installation

```bash
conda env create -n gvt python=3.8
conda install pytorch=1.7.1 torchvision=0.8.2 cudatoolkit=11.0
pip install mmcv-full==1.3.0
cd to/root/path
pip install -e .
```



## Deep Learning Experiments

### Source of Pre-trained models

* Transformer-based models can be found in [Google Dirve](https://drive.google.com/drive/folders/19H5O4YtIxIXaYdrS87Mh5ao1etjo3935?usp=drive_link).
* ResNet: pre-trained ResNet50 supported by Pytorch.



### Train on BJUT-URD dataset

For example, when dataset is BJUT-URD and method is GVT, we can run

```bash
python tools/train.py \
  --config configs/gvt/upernet_dcswin_small_96c_dilate2.py \
  --work-dir submits/gvt \
  --load_from path/to/pre-trained/model \
```



### Inference on BJUT-URD dataset

For example, when dataset is BJUT-URD and method is GVT, we can run

```bash
python tools/test.py \
  --config configs/gvt/upernet_dcswin_small_96c_dilate2.py \
  --checkpoint path/to/gvt/model \
  --show_dir path/to/save/segmentation/results \
```


# Code Functionality Explanation

This document provides a detailed explanation of the code located in the `mmseg/backbone` directory with the filename `swin_transformer_CrossShifted_dilation_DPE_realshift_preg.py`. This code is an improvement upon the Swin Transformer and includes three core functions: `CrossShiftAttention`, `CrossShiftTransformerBlock`, and `PixelRegionalBlock`. These functions correspond to three improvements upon the Swin Transformer, which can be utilized to achieve better performance during unpaved road segmentation.

## CrossShiftAttention

This function is an implementation of DCWin-Attention, which is designed to address deformation issues.

## CrossShiftTransformerBlock

It incorporates the shifted cross-window mechanism to tackle occlusion problems.

## PixelRegionalBlock

This function is an implementation of the pixel regional module, aiming to address roads with fuzzy edges.



## Hyperparameters Configuration

Detailed hyperparameters config can be found in configs/\_base\_/

## Acknowledgments

The code is developed based on the following repositories. We appreciate their nice implementations.

| Method           | Repository                                            |
| ---------------- | ----------------------------------------------------- |
| Swin Transformer | https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation |
| UPerNet          | https://github.com/CSAILVision/unifiedparsing |
| CSwin          | https://github.com/microsoft/CSWin-Transformer |
| PVT          | https://gitcode.com/mirrors/whai362/pvt |

## Cite this repository
If you use this software in your work, please cite it using the following metadata.

Li, W., Zhang, J., Li, J. et al. Unpaved road segmentation of UAV imagery via a global vision transformer with dilated cross window self-attention for dynamic map. Vis Comput (2024). https://doi.org/10.1007/s00371-024-03416-0
