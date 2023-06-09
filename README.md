# Image-tag-supervised learning for medical image segmentation (ISL4MIS).

## 1. Code for image-tag-supervised learning
<!-- ### 1.1 Usage -->
1. train the classification network
    ```shell
    python scripts/train_cls.py --img_dir your_img_dir
    ```

2. generate cam using trained classification network
    ```shell
    python scripts/cam.py --checkpoint best_checkpoint.pth
    ```



## 2. Literature reviews of image-tag-supervised learning approach for medical image segmentation.
### 2.1 State-of-the-art methods
| Date    | Paper source                        | Title                                                        | Code                                                         |
| ------- | ----------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 2022.09 | MICCAI                              | [Online Easy Example Mining for Weakly-supervised Gland Segmentation from Histology Images](https://arxiv.org/abs/2206.06665) | [PyTorch](https://github.com/xmed-lab/OEEM)                  |
| 2022.08 | MedIA                               | [Weakly Supervised Segmentation on Neural Compressed Histopathology with Self-Equivariant Regularization](https://www.sciencedirect.com/science/article/abs/pii/S1361841522001293) | [PyTorch](https://github.com/PhilipChicco/wsshisto)          |
| 2022.08 | MedIA                               | [Multi-Layer Pseudo-Supervision for Histopathology Tissue Semantic Segmentation using Patch-level Classification Labels](https://www.sciencedirect.com/science/article/pii/S1361841522001347) | [PyTorch](https://github.com/ChuHan89/WSSS-Tissue)           |
| 2022.06 | CVPR (oral)                         | [Adaptive Early-Learning Correction for Segmentation from Noisy Annotations](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Adaptive_Early-Learning_Correction_for_Segmentation_From_Noisy_Annotations_CVPR_2022_paper.pdf) | [PyTorch](https://github.com/Kangningthu/ADELE)              |
| 2022.06 | CVPR                                | [C-CAM: Causal CAM for Weakly Supervised Semantic Segmentation on Medical Image](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_C-CAM_Causal_CAM_for_Weakly_Supervised_Semantic_Segmentation_on_Medical_CVPR_2022_paper.pdf) |                                                              |
| 2022.03 | TMI                                 | [Deep Interpretable Classification and Weakly-Supervised Segmentation of Histology Images via Max-Min Uncertainty](https://ieeexplore.ieee.org/abstract/document/9591653) | [PyTorch](https://github.com/sbelharbi/deep-wsl-histo-min-max-uncertainty) |
| 2021.06 | JBHI                                | [Lung Lesion Localization of COVID-19 From Chest CT Image: A Novel Weakly Supervised Learning Method](https://ieeexplore.ieee.org/document/9382077) | [PyTorch](https://github.com/guaguabujianle/COVID-19-GAN)    |
| 2021.02 | MedIA                               | [An Interpretable Classifier for High-Resolution Breast Cancer Screening Images Utilizing Weakly Supervised Localization](https://www.sciencedirect.com/science/article/pii/S1361841520302723) | [PyTorch](https://github.com/nyukat/GMIC)                    |
| 2020.12 | JBHI                                | [MS-CAM: Multi-Scale Class Activation Maps for Weakly-Supervised Segmentation of Geographic Atrophy Lesions in SD-OCT Images](https://ieeexplore.ieee.org/abstract/document/9121691) | [Tensorflow](https://github.com/jizexuan/Multi-Scale-Class-Activation-Map-Tensorflow) |
| 2020.09 | MICCAI                              | [Weakly Supervised Organ Localization with Attention Maps Regularized by Local Area Reconstruction](https://link.springer.com/chapter/10.1007/978-3-030-59710-8_24) |                                                              |
| 2020.09 | Artificial Intelligence in Medicine | [Weakly-Supervised Segmentation for Disease Localization in Chest X-Ray Images](https://link.springer.com/chapter/10.1007/978-3-030-59137-3_23) | [PyTorch](https://github.com/ucuapps/WSMIS)                  |


### 2.2 Literature reviews of CAM Variants.
| Date    | Paper source                        | Title                                                        | Code                                                         |
| ------- | ----------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 2020.06 | CVPR workshop                       | [Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks](https://openaccess.thecvf.com/content_CVPRW_2020/html/w1/Wang_Score-CAM_Score-Weighted_Visual_Explanations_for_Convolutional_Neural_Networks_CVPRW_2020_paper.html) | [PyTorch](https://github.com/haofanwang/Score-CAM)                  |
| 2018.05 | WACV                       | [Grad-CAM++: Generalized Gradient-based Visual Explanations for Deep Convolutional Networks](https://ieeexplore.ieee.org/abstract/document/8354201) |                 |
| 2017.10 | ICCV                              | [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html) | [PyTorch](https://github.com/ramprs/grad-cam/)                  |
| 2016.06 | CVPR                              | [Learning Deep Features for Discriminative Localization](https://openaccess.thecvf.com/content_cvpr_2016/html/Zhou_Learning_Deep_Features_CVPR_2016_paper.html) | [PyTorch](https://github.com/zhoubolei/CAM)                  |



## Acknowledgement
- Part of the code is adapted from open-source codebase and original implementations of algorithms, we thank these authors for their fantastic and efficient codebase, such as [DRS](https://github.com/qjadud1994/DRS) and [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam).