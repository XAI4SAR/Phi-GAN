$\Phi$-GAN: Physics-Inspired GAN for Generating SAR Images Under Limited Data
==== 
1.Introduction  
------- 
This project is for paper [$\Phi$-GAN: Physics-Inspired GAN for Generating SAR Images Under Limited Data](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_Ph-GAN_Physics-Inspired_GAN_for_Generating_SAR_Images_Under_Limited_Data_ICCV_2025_paper.pdf).

### 1.1 Features
![img](img/fig_method.png)

The proposed $\Phi$-GAN framework overview.





### 1.2 Contribution
* A physics-inspired GAN framework, $\Phi$-GAN, is proposed for SAR image generation, aiming to improve training stability and generalization under data-scarce conditions.
* $\Phi$-GAN consists of a physics-inspired neural module for PSC parameter inversion and two specialized physical loss functions for training regularization.
* Extensive experiments are conducted on diverse SAR image generation tasks. Built upon multiple existing conditional GAN architectures, the proposed $\Phi$-GAN consistently demonstrates strong adaptability, improved generalization, and robust generation performance.



2.Getting Started
------- 
### 2.1 Requirements
Code is based on an  object detection YOLOv5. Please refer to [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) for installation and dataset preparation.



3.Citation
------- 
If you find this repository useful for your publications, please consider citing our paper.

```
@inproceedings{zhang2025ph,
  title={Ph-GAN: Physics-inspired GAN for generating SAR images under limited data},
  author={Zhang, Xidan and Zhuang, Yihan and Guo, Qian and Yang, Haodong and Qian, Xuelin and Cheng, Gong and Han, Junwei and Huang, Zhongling},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={29075--29085},
  year={2025}
}
```
