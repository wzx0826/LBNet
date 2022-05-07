# LBNet-Pytorch: Lightweight Bimodal Network for Single-Image Super-Resolution via Symmetric CNN and Recursive Transformer
### This repository is an official PyTorch implementation of the paper "Lightweight Bimodal Network for Single-Image Super-Resolution via Symmetric CNN and Recursive Transformer".


## Dependencies
```
Python>=3.7 
PyTorch>=1.1
numpy 
skimage 
imageio 
matplotlib 
tqdm
```

For more informaiton, please refer to <a href="https://github.com/thstkdgus35/EDSR-PyTorch">EDSR</a>

## Dataset

We used DIV2K dataset to train our model. Please download it from <a href="https://data.vision.ee.ethz.ch/cvl/DIV2K/">here</a>  or  <a href="https://cv.snu.ac.kr/research/EDSR/DIV2K.tar">SNU_CVLab</a>.

You can evaluate our models on several widely used [benchmark datasets](https://cv.snu.ac.kr/research/EDSR/benchmark.tar), including Set5, Set14, B100, Urban100, Manga109. Note that using an old PyTorch version (earlier than 1.1) would yield wrong results.

## Results
All our SR images can be downloaded from <a href="https://pan.baidu.com/s/1BfATKktSv9jk3LlWPRQRZg">here</a>.[百度网盘][提取码:ymii]

All pretrained model can be found in <a href="https://github.com/wzx0826/LBNet/tree/main/test_model">IJCAI2022_LBNet</a>.

The following PSNR/SSIMs are evaluated on Matlab R2017a and the code can be referred to [Evaluate_PSNR_SSIM.m]
(https://github.com/24wenjie-li/FDIWN/blob/main/FDIWN_TestCode/Evaluate_PSNR_SSIM.m).

##Training

```
  LBNet: num_heads = 8
  
# LBNet x4
python main.py --scale 4 --model LBNet --save experiments/LBNet_X4

# LBNet x3
python main.py --scale 3 --model LBNet --save experiments/LBNet_X3

# LBNet x2
python main.py --scale 2 --model LBNet --save experiments/LBNet_X2

  LBNet-T：num_heads = 6

# LBNet-T x4
python main.py --scale 4 --model LBNet-T --save experiments/LBNet-T_X4

# LBNet-T x3
python main.py --scale 3 --model LBNet-T --save experiments/LBNet-T_X3

# LBNet-T x2
python main.py --scale 2 --model LBNet-T --save experiments/LBNet-T_X2

```

##Testing

```
  LBNet: num_heads = 8
  
# LBNet x4
python main.py --scale 4 --model LBNet --pre_train test_model/LBNet/LBNet-X4.pt --test_only --save_results --n_feat 32 --data_test Set5

  LBNet-T：num_heads = 6

# LBNet-T x4
python main.py --scale 4 --model LBNet-T --pre_train test_model/LBNet-T/LBNet-T_X4.pt --test_only --save_results --n_feat 18 --data_test Set5

```


## Acknowledgements
This code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch) and [DRN](https://github.com/guoyongcs/DRN). We thank the authors for sharing their codes.

## Citation

If you use any part of this code in your research, please cite our paper:

```
@article{gao2022lightweight ,
  title={Lightweight Bimodal Network for Single-Image Super-Resolution via Symmetric CNN and Recursive Transformer},
  author={Gao, Guangwei and Wang, Zhengxue and Li, Juncheng and Li, Wenjie and Yu, Yi and Zeng, Tieyong},
  journal={arXiv preprint arXiv:2204.13286},
  year={2022}
}
```
