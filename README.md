# GS^2: Graph-based Spatial Distribution Optimization for Compact 3D Gaussian Splatting


This repository is an official implementation of the paper [ GS^2: Graph-based Spatial Distribution Optimization for Compact 3D Gaussian Splatting]().



## Introduction

3D Gaussian Splatting (3DGS) has demonstrated breakthrough performance in novel view synthesis and real-time rendering.   Nevertheless, its practicality is constrained by the high memory cost due to a huge number of Gaussian points.   Many pruning-based 3DGS variants have been proposed for memory saving, but often compromise spatial consistency and may lead to rendering artifacts. To address this issue, we propose graph-based spatial distribution optimization for compact 3D Gaussian Splatting (GS\textasciicircum2), which enhances reconstruction quality by optimizing the spatial distribution of Gaussian points. Specifically, we introduce an evidence lower bound (ELBO)-based adaptive densification strategy that automatically controls the densification process. In addition, an opacity-aware progressive pruning strategy is proposed to further reduce memory consumption by dynamically removing low-opacity Gaussian points. Furthermore, we propose a graph-based feature encoding module to adjust the spatial distribution via feature-guided point shifting.  Extensive experiments validate that GS\textasciicircum2 achieves a compact Gaussian representation while delivering superior rendering quality. Compared with 3DGS, it achieves higher PSNR with only about 12.5\% Gaussian points. Furthermore, it outperforms all compared baselines in both rendering quality and memory efficiency.  

<br/>
<div align="center">
  <img src="./framework.png" width="90%"/>

  Fig. 1: Overall architecture of the proposed GS^2 model.
</div>

## Reproducing Results
Due to the randomness of the densification process and random initialization, the metrics may be unstable in some scenes, especially PSNR.


### Checkpoints and Results
You can download our provided checkpoints from [here](). These results are reproduced with a lower error tolerance bound to keep aligned with this repo, which is different from what we use in the paper. This could lead to higher metrics but worse visualization.




## Citing GS^2
If you find GS^2 useful in your research, please consider citing:
```bibtex
@ARTICLE{},
  doi={}}
```


## License

This repository is released under the Apache 2.0 license. Please see the [LICENSE](./LICENSE) file for more information.
