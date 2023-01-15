# Paper

Available on arXiv:  
https://arxiv.org/abs/2212.12678  
or 
ACM DL (The supplements are at the end of the PDF file, which contains the description of the CIN, training details, and noise setting):  
https://dl.acm.org/doi/abs/10.1145/3503161.3547950

# Introduction

![Framework](https://github.com/rmpku/CIN/blob/main/images/1.jpg)
![Visualization](https://github.com/rmpku/CIN/blob/main/images/2.jpg)
![invertibleNet](https://github.com/rmpku/CIN/blob/main/images/6.jpg)

# Dataset Preparation

**COCO2017:**  
_Lin, Tsung-Yi, et al. "Microsoft coco: Common objects in context." European conference on computer vision. Springer, Cham, 2014._

**DIV2K:**  
_Agustsson, Eirikur, and Radu Timofte. "Ntire 2017 challenge on single image super-resolution: Dataset and study." Proceedings of the IEEE conference on computer vision and pattern recognition workshops. 2017._

# Environment

_nvidia=3080  
cuda=11.1  
python=3.8.3  
torch=1.13.0  
torchvision=0.14.0  
opencv-python=4.6.0.66  
kornia=0.6.8  
colormath=3.0.0  
pyyaml=6.0  
importlib-metadata=5.1.0_

# Pretrained model - "combined noises"

**Training with Noise pool:**  
_{'Identity', 'JpegTest', 'Crop', 'Cropout', 'Resize', 'GaussianBlur', 'Salt\*Pepper', 'GaussianNoise', 'Dropout', 'Brightness', Contrast', 'Saturation', 'Hue'}_  
**When testing:**  
you only need to modify the noise-option in _/codes/options/opt.yml/noise/option_  
**Google Cloud link:**  
https://drive.google.com/file/d/1wqnqhPv92mHwkEI4nMh-sI5aDgh-usr7/view?usp=share_link

# Citation

If you find this work useful, please cite our paper:

- @inproceedings{ma2022towards,
  title={Towards Blind Watermarking: Combining Invertible and Non-invertible Mechanisms},
  author={Ma, Rui and Guo, Mengxi and Hou, Yi and Yang, Fan and Li, Yuan and Jia, Huizhu and Xie, Xiaodong},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={1532--1542},
  year={2022}
  }
-

# ACKNOWLEDGMENTS

This work was supported by National Key R&D Program of China 2021ZD0109802 and National Science Foundation of China 61971047.

# Contact

If you have any questions, please contact rui_m@stu.pku.edu.cn or post them in the https://github.com/rmpku/CIN/issues.
