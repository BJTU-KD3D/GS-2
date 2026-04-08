# GS^2: Graph-based Spatial Distribution Optimization for Compact 3D Gaussian Splatting


This repository is an official implementation of the paper [ GS^2: Graph-based Spatial Distribution Optimization for Compact 3D Gaussian Splatting](http://arxiv.org/abs/2604.01884).



## Introduction

3D Gaussian Splatting (3DGS) has demonstrated breakthrough performance in novel view synthesis and real-time rendering.   Nevertheless, its practicality is constrained by the high memory cost due to a huge number of Gaussian points.   Many pruning-based 3DGS variants have been proposed for memory saving, but often compromise spatial consistency and may lead to rendering artifacts. To address this issue, we propose graph-based spatial distribution optimization for compact 3D Gaussian Splatting (GS\textasciicircum2), which enhances reconstruction quality by optimizing the spatial distribution of Gaussian points. Specifically, we introduce an evidence lower bound (ELBO)-based adaptive densification strategy that automatically controls the densification process. In addition, an opacity-aware progressive pruning strategy is proposed to further reduce memory consumption by dynamically removing low-opacity Gaussian points. Furthermore, we propose a graph-based feature encoding module to adjust the spatial distribution via feature-guided point shifting.  Extensive experiments validate that GS\textasciicircum2 achieves a compact Gaussian representation while delivering superior rendering quality. Compared with 3DGS, it achieves higher PSNR with only about 12.5\% Gaussian points. Furthermore, it outperforms all compared baselines in both rendering quality and memory efficiency.  

<br/>
<div align="center">
  <img src="./pipeline.png" width="90%"/>

  Fig. 1: Overall architecture of the proposed GS^2 model.
</div>


## Dataset

### Mip-NeRF 360 Dataset

Please download the Mip-NeRF 360 dataset processed by colmap from [Mip-NeRF 360](https://jonbarron.info/mipnerf360/):

```
360_v2
    |---bicycle
    |   |---images
    |   |   |---<image 0>
    |   |   |---<image 1>
    |   |   |---...
    |   |---images_2
    |   |---images_4
    |   |---images_8
    |   |---sparse
    |       |---0
    |           |---cameras.bin
    |           |---images.bin
    |           |---points3D.bin
    |---bonsai
    |---...
```

### Tanks and Temples Dataset

#### Option 1

We thank [Pixel-GS](https://github.com/zhengzhang01/Pixel-GS) for constructing the processed Tanks and Temples dataset, which is available for direct download via [OneDrive](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3009782_connect_hku_hk/EehzMcKeoclAnVdgPyyBxNwB24ve5bk3ZSct38AUWPbprw?e=uWEc5a). Please agree the official license before download it.

#### Option 2 

Tanks and Temples is divided into three parts, comprising a total of 21 scenes: Intermediate ('Family', 'Francis', 'Horse', 'Lighthouse', 'M60', 'Panther', 'Playground', 'Train'), Advanced ('Auditorium', 'Ballroom', 'Courtroom', 'Museum', 'Palace', 'Temple'), and Training Data ('Barn', 'Caterpillar', 'Church', 'Courthouse', 'Ignatius', 'Meetingroom', 'Truck').

Please download the "image set" of all scenes from the Tanks and Temples dataset from [Tanks and Temples](https://www.tanksandtemples.org/download/). After unzipping, rename the image folder directories of all scenes to "input". The organized folder structure is as follows:

```
---tanks_and_temples
    |---Auditorium
    |   |---input
    |   |   |---<image 0>
    |   |   |---<image 1>
    |   |   |---...
    |---Ballroom
    |---...
```

After configuring libraries such as colmap according to the method in the original [3DGS code](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#processing-your-own-scenes), use the following command to generate camera poses for all scenes in Tanks and Temples:

```
python ./prepose.py
```

Finally, the current directory should contain the following folders:

```
---tanks_and_temples
    |---Auditorium
    |   |---images
    |   |   |---<image 0>
    |   |   |---<image 1>
    |   |   |---...
    |   |---images_2
    |   |---images_4
    |   |---images_8
    |   |---sparse
    |       |---0
    |           |---cameras.bin
    |           |---images.bin
    |           |---points3D.bin
    |---Ballroom
    |---...
```

Then, you need to rename the folder 'images_2' to 'images', since the resolution we are working with is close to 980x545.
You can also avoid renaming by simply changing `factors = [1] * len(scenes)` to `factors = [2] * len(scenes)` in `./script/tanks_and_temples.py`. This modification will instruct the code to reshape the images upon loading during training.

### Your Own Dataset

Our method requires the same data format as 3DGS. For your own data, you can use the processing method found in the ["Processing your own Scenes"](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#processing-your-own-scenes) section of the original 3DGS code.

## Getting Started 

Our code is based on the excellent official repo for [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting/tree/main). 

## Training

Modify the paths to dataset and output folder in the ```run.sh``` script. In practice, increasing the learning rate after pruning (e.g., setting `position_lr_init = 5e-6`) can partially reproduce the effect of the third stage, serving as a lightweight approximation.

```shell

cd gaussian-splatting
bash run.sh
```

## Pre-trained Models
Trained models are now available [here](https://drive.google.com/file/d/1UWPlaA02wXMosi5o1cHjHDZ-8huE34WP/view?usp=drive_link). You can download these models and provide the paths in the render and evaluation codes to get the metrics. The metrics might not exactly match those in the paper since these are re-runs with different seeds.



## Citing GS^2
If you find GS^2 useful in your research, please consider citing:
```bibtex
@article{yang2026gs2graphbasedspatialdistribution,
   title={GS^2: Graph-based Spatial Distribution Optimization for Compact 3D Gaussian Splatting}, 
      author={Xianben Yang, Tao Wang , Yuxuan Li, Yi Jin and Haibin Ling},
      year={2026},
      eprint={2604.01884}
}
```


## Acknowledgement
This project is built upon [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting) and [LightGaussian](https://github.com/VITA-Group/LightGaussian). We thank all authors for their great work!
## License

This repository is released under the Apache 2.0 license. Please see the [LICENSE](./LICENSE) file for more information.
