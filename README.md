# Multi-Scale Denoising in the Feature Space for Low-Light Instance Segmentation [ICASSP 2025]

**Authors**: _Joanne Lin, Nantheera Anantrasirichai, David Bull_

**Institution**: Visual Information Laboratory, University of Bristol, United Kingdom

[[`Project Page`](https://joannelin168.github.io/low-light/LoIS/)][[`Paper`](https://ieeexplore.ieee.org/document/10889336)][[`arXiv`](https://arxiv.org/abs/2402.18307)]

---

> Instance segmentation for low-light imagery remains largely unexplored due to the challenges imposed by such conditions, for example shot noise due to low photon count, color distortions and reduced contrast. In this paper, we propose an end-to-end solution to address this challenging task. Our proposed method implements weighted non-local blocks (wNLB) in the feature extractor. This integration enables an inherent denoising process at the feature level. As a result, our method eliminates the need for aligned ground truth images during training, thus supporting training on real-world low-light datasets. We introduce additional learnable weights at each layer in order to enhance the network’s adaptability to real-world noise characteristics, which affect different feature scales in different ways. Experimental results on several object detectors show that the proposed method outperforms the pre-trained networks with an Average Precision (AP) improvement of at least +7.6, with the introduction of wNLB further enhancing AP by upto +1.3.

This repository provides the implementation of our proposed Weighted Non-Local Blocks (wNLB) from the paper "Multi-Scale Denoising in the Feature Space for Low-Light Instance Segmentation" [ICASSP 2025]

If you use our work in your research, please cite using the following BibTeX entry:

```bibtex
@INPROCEEDINGS{lin2025lowlightsegm,
  author={Lin, Joanne and Anantrasirichai, Nantheera and Bull, David},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Multi-Scale Denoising in the Feature Space for Low-Light Instance Segmentation}, 
  year={2025},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/ICASSP49660.2025.10889336}}
```
