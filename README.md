# Adversarial Attack Transferability

Notebooks to test transferability of adversarial attacks against ImageNet classifiers.

## Description

This repo contains two notebooks that test the transferability of the projected gradient descent [1] (PGD) and AutoAttack [2] adversarial attack methods on clean samples from the ImageNet-Val [3] dataset. The attacks are performed on a ResNet-50 [4] victim classifier, and the transferability to ResNet-18, ResNet-34, ResNet-152, Inception-V3 [5], and ViT [6] classifiers is calculated. To run the notebooks ensure that the environment has been loaded from the provided `environment.yml` file, and that the ImageNet dataset has been downloaded from [here](https://image-net.org/download.php) and extracted to the local file system.

Note that the AutoAttack implementation is taken from the original [repository](https://github.com/fra31/auto-attack), while the `torchattacks` [7] implementation of PGD is used. 

## References
[1] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu, Towards deep learning models resistant to adversarial attacks, 2019.</br>
[2] F. Croce and M. Hein, Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks, Auto-Attack, 2020.</br>
[3] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and F. F. Li, “ImageNet: A large-scale hierarchical image database,” in 2009 IEEE Conference on Computer Vision and Pattern Recognition, 2009.</br>
[4] K. He, X. Zhang, S. Ren, and J. Sun, Deep residual learning for image recognition, ResNet architecture, 2015.</br>
[5] C. Szegedy, W. Liu, Y. Jia, et al., Going deeper with convolutions, 2014.</br>
[6] A. Dosovitskiy, L. Beyer, A. Kolesnikov, et al., An image is worth 16x16 words: Transformers for image recognition at scale, 2021.</br>
[7] H. Kim, “Torchattacks: A pytorch repository for adversarial attacks,” arXiv preprint arXiv:2010.01950, 2020.</br>

## Installation
Install the prerequisite conda environment:
```
conda env create --name TransferTest --file environment.yml
```

## Authors

Max Collins (max.collins@research.uwa.edu.au)
