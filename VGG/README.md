# Very Deep Convolutional Networks for Large-scale image recognition

The main contribution of this paper is “a thorough evaluation of networks of increasing depth using an architecture with very small (3×3) convolution filters.” To demonstrate this, I performed two experiments. The First one is about the size of convolution layers (proposed), and the second one is about the depth of convolution layers (simplified experiment for reproducing Sec 4.1) 
For the experiments, I implemented the same Conv. layer configurations in the paper, but used CIFAR10 dataset because of limited resources and time. Therefore, I needed to adjust some settings so that it matched with the dataset. The major differences are as following:
1.	Dataset : ImageNet (256x256, 1000 classes) -> CIFAR10 (32x32, 10 classes)
2.	Batch size : 256 -> 512 
3.	Fully-connected layer size : (25088, 4096), (4096, 4096), (4096, 1000) -> (512, 512), (512, 10)
4.	Weight initialization variance : 0.01 -> 0.035

## Experiment 1. Stack of small layers vs One large layer
In section 2.3, the authors mentioned we have some benefits by using a stack of 3×3 conv. layers instead of a single larger layer. To demonstrate this, I compared the performance of each configuration with equivalent large conv. network by replacing a stack of two 3x3 conv. to a 5x5 conv., and a stack of three 3x3 conv. to a 7x7 conv. The result shows VGG networks achieved better validation and test accuracy than each corresponding Larger conv. network. The result shows VGG outperforms larger conv. net in all case.

## Experiment 2. Depth of Convolution layer
This experiment is a simplified version of experiment in section 4.1. The authors says deeper network can capture more features from image, and so it performs well. To capture the key concept, I implemented some experiments which are related to effect of depth. The result shows the deeper the net, the better the accuracy. We can check the additional non-linearity does help (C is better than B), and it is important to capture spatial context by using conv. filters with non-trivial receptive fields (D is better than C).

The result of EXP1, EXP2 is shown in Table 1. I also added a graph of each networks’ training steps in ‘result.ipynb’ file.
![table](https://user-images.githubusercontent.com/52485688/87546064-e6c88c00-c6e3-11ea-87f8-24b2140cc29e.PNG)
Table 1. Combined result of EXP1 and EXP 2. You can compare the VGG net and Larger conv. net from the row of the table (EXP1), and you can see the effect of depth to VGG nets from the left column of the table (EXP2).


