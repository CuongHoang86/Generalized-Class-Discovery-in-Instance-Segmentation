## Generalized-Class-Discovery-in-Instance-Segmentation

This repository provides the official implementation of the following paper:

**Generalized Class Discovery in Instance Segmentation**

by Cuong Manh Hoang, Yeejin Lee, Byeongkeun Kang,The 39th Annual AAAI Conference on Artificial Intelligence, 2025

[Paper]() | [Source code](https://github.com/CuongHoang86/Generalized-Class-Discovery-in-Instance-Segmentation/tree/main) | [Poster]() 


**What is it?** In Generalized-Class-Discovery-in-Instance-Segmentation, the model has to segment and discover objects of unknown classes using only the labels of known classes.

<img src="./figure/Selection_330.png" width="70%"> <br/>

**How we do it?** In our framework, we first utilize an open-world instance segmentation model, pretrained with class-agnostic masks of known classes, to segment class-agnostic masks of unknown classes. Then, the objects of known and unknown classes are cropped to make a new object dataset. In this dataset, object images of known classes contain their categories and object images of unknown classes do not contain their categories. To cluster object images of unknown classes, we propose a generalized class discovery model with a novel instance-wise temperature assignment and a novel soft attention module. Finally, masks and categories of all objects are used to train the instance segmentation model with a novel reliability-based dynamic learning. 

<img src="./figure/Selection_329.png" width="70%"> <br/>

## Installation
Our implementation is based on the [Detectron2 v0.6](https://github.com/facebookresearch/detectron2) framework. For our experiments we followed [Detectron2's setup guide](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md) and picked the **CUDA=11.3** and **torch=1.10** versions (`python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html` may work). All the additional dependencies we put to `requirements.txt`. We used Python 3.8 for all experiments.

## Download datasets
Please follow [RNCDL](https://github.com/vlfom/RNCDL/tree/main) for data preparation.

## Citation
If you find our paper useful in your research or reference it in your work, please star our repository and use the folowing:
```bibtex
```
