# Unpaved Road Segmentation of UAV Imagery via a Global Vision Transformer with Dilated Cross Window Self-Attention for Path Navigation


## Installation

```bash
conda env create -n gvt python=3.8
conda install pytorch=1.7.1 torchvision=0.8.2 cudatoolkit=11.0
pip install mmcv-full==1.3.0
cd path/to/sources
pip install -e .
```



## Deep Learning Experiments

### Source of Pre-trained models

* Transformer-based models can be found in [./checkpoints](https://github.com/BJUT-AIVBD/GVT/checkpoints).
* ResNet: pre-trained ResNet50 supported by Pytorch.



### Train on BJUT-URD dataset

For example, when dataset is BJUT-URD and method is GVT, we can run

```bash
python tools/train.py \
  --config configs/gvt/upernet_dilatecswin_dep_realshift_preg_512×512_160k.py \
  --work-dir submits/gvt \
  --load_from checkpoints/swin/models \
```



### Inference on BJUT-URD dataset

For example, when dataset is BJUT-URD and method is GVT, we can run

```bash
python tools/test.py \
  --config configs/gvt/upernet_dilatecswin_dep_realshift_preg_512×512_160k.py \
  --checkpoint submits/gvt \
  --show_dir checkpoints/swin/models \
```



## Hyperparameters Configuration

Detailed hyperparameters config can be found in configs/\_base\_/

## Acknowledgments

The code is developed based on the following repositories. We appreciate their nice implementations.

| Method           | Repository                                            |
| ---------------- | ----------------------------------------------------- |
| Swin Transformer | https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation |
| UPerNet          | https://github.com/CSAILVision/unifiedparsing       

## Cite this repository
If you use this software in your work, please cite it using the following metadata.
Wensheng, Li, Jing, Zhang. (2023). GVT by BJUT-VI&VBD [Computer software]. https://github.com/BJUT-AIVBD/GVT
