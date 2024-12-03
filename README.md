# LFAVNet

###### *PyTorch implementation of TVCG paper: "View-guided Cost Volume for Light Field Arbitrary-view Disparity Estimation"*.
![LFAVNet](./LFAVNet.gif)

[Paper](https://ieeexplore.ieee.org/abstract/document/10664533)
#### Requirements

- python 3.6
- pytorch 1.8.0
- ubuntu 18.04

### Installation

First you have to make sure that you have all dependencies in place. 

You can create an anaconda environment called LFAVNet using

```
conda env create -f LFAVNet.yaml
conda activate LFAVNet
```

##### Dataset: 

Light Field Dataset: We use [HCI 4D Light Field Dataset](https://lightfield-analysis.uni-konstanz.de/) for training and test. Please first download light field dataset with its full-view depth information, and put them into corresponding folders in ***data/HCInew***.

##### Model weights: 
Please download the model weights from [Google Drive](https://drive.google.com/file/d/1lhDqVPa-QnpK_wX9oN2HPBkaVdmPzsOv/view?usp=sharing), and put them in the ***out/LFAVNet/HCInew***.

(We have cleaned up the code and retrained our model, so the metrics are slightly different from we reported.)

##### Results:
We also provide depth estimation results for the LF [center view](XX) and [full view](XX).

##### To train, run:

```
python train.py --config configs/HCInew/LFAVNet.yaml 
```

##### To generate, run:

```
python generate.py --config configs/pretrained/HCInew/LFAVNet_pretrained.yaml 
```



**If you find our code or paper useful, please consider citing:**
```
@article{chen2024view,
  title={View-guided Cost Volume for Light Field Arbitrary-view Disparity Estimation},
  author={Chen, Rongshan and Sheng, Hao and Yang, Da and Wang, Sizhe and Cui, Zhenglong and Cong, Ruixuan and Wang, Shuai},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  year={2024},
  publisher={IEEE}
}
```

