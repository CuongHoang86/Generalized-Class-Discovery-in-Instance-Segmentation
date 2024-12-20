## Generalized Class Discovery in Instance Segmentation

This repository provides the official implementation of the following paper:

**Generalized Class Discovery in Instance Segmentation**

[Cuong Manh Hoang](https://scholar.google.com/citations?user=7sUs5o8AAAAJ&hl=en), [Yeejin Lee](https://ieeexplore.ieee.org/author/37075435600), [Byeongkeun Kang](https://scholar.google.com/citations?user=YvKVr0UAAAAJ&hl=en)

In the 39th Annual AAAI Conference on Artificial Intelligence (AAAI 2025)

[Paper]() | [Source code](https://github.com/CuongHoang86/Generalized-Class-Discovery-in-Instance-Segmentation/tree/main) | [Poster]() 


**What is it?** In generalized class discovery in instance segmentation, it assumes that the unlabeled data may contain both known and novel classes, making the problem more challenging and realistic. Given labeled and unlabeled data, the model is trained to recognize both the known classes in the labeled data and the novel categories discovered from the unlabeled data.

<img src="./figure/Selection_330.png" width="70%"> <br/>

**How we do it?** In our framework, we first utilize an open-world instance segmentation model [GGN](https://github.com/facebookresearch/Generic-Grouping), pretrained with class-agnostic masks of known classes, to segment class-agnostic masks of unknown classes. Then, the objects of known and unknown classes are cropped to make a new object dataset. In this dataset, object images of known classes contain their categories and object images of unknown classes do not contain their categories. To cluster object images of unknown classes, we propose a generalized class discovery model with a novel instance-wise temperature assignment and a novel soft attention module. Finally, masks and categories of all objects are used to train the instance segmentation model [SOLO](https://github.com/WXinlong/SOLO) with a novel reliability-based dynamic learning. 

<img src="./figure/Selection_329.png" width="70%"> <br/>

## Installation
Our implementation is based on the [Detectron2 v0.6](https://github.com/facebookresearch/detectron2) framework. For our experiments we followed [Detectron2's setup guide](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md) and picked the **CUDA=11.3** and **torch=1.10** versions (`python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html` may work). All the additional dependencies we put to `requirements.txt`. We used Python 3.8 for all experiments.

## Download datasets
Please follow [RNCDL](https://github.com/vlfom/RNCDL/tree/main) to download data.

## Citation
If you find our paper useful in your research or reference it in your work, please star our repository and use the folowing:
```bibtex
```
