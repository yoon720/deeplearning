# Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks

This paper proposes CycleGAN which learns to automatically translate an image from one into the other and vice versa. The algorithm uses two main loss function, adversarial loss (GAN loss) and cycle consistency loss. The authors claim that by combining two loss functions, the network can learn how to change an image in one domain into another domain while preserving properties of image itself. I reproduced two experiments in the paper to demonstrate it. 
I used preprocessed datasets downloaded from user’s official implementation code[1]. Also, I referred a prior work[2] to implement image buffer but slightly changed the number of returned images to probability because CycleGAN uses batch size = 1. The implementation details are given below.

## Experiment 1. Horse to zebra and vice versa
In this experiment, I tried to reproduce one of the translation examples in the paper. I trained a CycleGAN to translate horse to zebra and vice versa. The result shows translation using CycleGAN works well. Especially, it performs better when there is one sufficiently large object in the image. Also, there are some changes in background depending on usual background in each class. Some of the examples are shown in Figure 1. 

![figure1](https://user-images.githubusercontent.com/52485688/87550491-0bbffd80-c6ea-11ea-91d0-52955158e171.png)
Figure 1. Successful examples of horse to zebra and zebra to horse translation.


## Experiment 2. Analysis of the loss function
In this experiment, I tried to reproduce section 5.1.4 and Figure 7 in the paper. The authors compare against ablations of the full loss and conclude both cycle-consistency loss and GAN loss are important to train CycleGAN. However, in my implementation, some models shows model collapse. I assume this is because of the buffers while training discriminators. In buffer, the fake images from early epoch might survive for a long time and it could harm the training. It would be much dangerous with batch size = 1. To check this, I tried to train CycleGAN without buffers but I didn’t have enough time to complete training. Therefore, I selected the best results before the model collapse from each model’s training result in Figure 2. From the result, we can see that cycle-consistency loss is good for training structural information, and GAN loss is good for extracting color information and making realistic information. Therefore, we need both loss functions to generate translate images using CycleGAN. 

![figure2](https://user-images.githubusercontent.com/52485688/87550513-124e7500-c6ea-11ea-908e-b34bcccb2d88.png)
Figure 2. Successful examples of cityscapes translations between labels and photos. As shown in the images, cycle-consistency loss is helpful to extract structural characteristic and GAN loss is helpful to catch color information and to generate realistic images.


CycleGAN is an interesting approach of GAN. However, it is not stable because the model uses too small batch size and buffer together. Also, we need to train one generator and one discriminator for one translation task. This is not efficient in terms of memory management. 




[1] https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/datasets.md

[2] A. Shrivastava, T. Pfister, O. Tuzel, J. Susskind, W. Wang, and R. Webb. Learning from simulated and unsupervised images through adversarial training. In CVPR, 2017.
