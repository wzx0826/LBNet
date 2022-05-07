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

All our Supplementary materials can be downloaded from <a https://pan.baidu.com/s/1JdnWHy3cwdwPpSC2G0L_ag ">Supplementary materials</a>.[百度网盘][提取码:vi3i]

The following PSNR/SSIMs are evaluated on Matlab R2017a and the code can be referred to <a href="https://github.com/wzx0826/LBNet/blob/main/Evaluate_PSNR_SSIM.m">Evaluate_PSNR_SSIM.m</a>.

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
## Performance

Our LBNet is trained on RGB, but as in previous work, we only reported PSNR/SSIM on the Y channel.

Model|Scale|Params|Multi-adds|Set5|Set14|B100|Urban100|Manga109
--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:
LBNet-T        |x2|404K|49.0G|37.95/0.9602|33.53/0.9168|32.07/0.8983|31.91/0.9253|38.59/0.9768
LBNet          |x2|731K|153.2G|38.05/0.9607|33.65/0.9177|32.16/0.8994|32.30/0.9291|38.88/0.9775
LBNet-T        |x3|407K|22.0G|34.33/0.9264|30.25/0.8402|29.05/0.8042|28.06/0.8485|33.48/0.9433
LBNet          |x3|736K|68.4G|34.47/0.9277|30.38/0.8417|29.13/0.8061|28.42/0.8559|33.82/0.9460
LBNet-T        |x4|410K|12.6G|32.08/0.8933|28.54/0.7802|27.54/0.7358|26.00/0.7819|30.37/0.9059
LBNet          |x4|742K|38.9G|32.29/0.8960|28.68/0.7832|27.62/0.7382|26.27/0.7906|30.76/0.9111

## Visual comparison

SR images reconstructed by our LBNet have richer detailed textures with better visual effects.
<p align="center">
<img src="imgs/LBNet-Patch-X3.drawio.png" width="600px" height="400px"/>
</p>
<p align="center">
<img src="imgs/LBNet-Patch.drawio.png" width="600px" height="400px"/>
</p>

## Model complexity

LBNet gains a better trade-off between model size, performance, inference speed, and multi-adds.
<p align="center">
<img src="imgs/ExecTime_x4.png" width="400px" height="300px"/>
<img src="imgs/LBNet_Tradeoff_Params.png" width="400px" height="300px"/>
<img src="imgs/LBNet_Tradeoff_MultAdds.png" width="400px" height="300px"/>
</p>


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
