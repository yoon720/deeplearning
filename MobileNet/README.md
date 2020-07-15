# MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications


In this paper, the authors proposed a light weight deep neural network using depthwise separable convolutions and hyperparameters that shrinks the size of model. To verify the effect of them, I performed two experiments. The First one is about the model comparison (Sec 4.1), and the second one is about the model shrinking hyperparameters (Sec 4.2). For the experiments, I have implemented the same body architecture in the paper, but used different datasets because of limited resources and time. Therefore, I needed to adjust some settings so that it matched with the dataset. Details will be given below.


## Experiment 1. Model Choices (sec 4.1) (dataset : CIFAR10)
Experiment 1 is for the demonstration of the key concept of this paper. I used CIFAR10 dataset for this experiment. Therefore I slightly changed the average pooling layer because the size of the image in CIFAR10 is 32x32. (avgpool(7) -> avgpool(1)) I followed the configurations in the paper for the other layers. 
The result of the experiment is summarized in the Table 1. MobileNet only reduces test accuracy by 1.35% while it needs much less computations, parameters and time than Conv MobileNet. Between Narrow MobileNet and shallow MobileNet, the first one showed better performance even though it needs less computations and parameters. More detailed results can be found in ‘Experiment1-CIFAR10.ipynb’ 

|Model|	Val. acc|	Test acc|	Comp cost|	# Param|	Train time|
|:-----|:-----:|:-----:|:-----:|:-----:|:-----:|
|Conv MobileNet|	74.04|	73.42|	166|	30.6|	24m 23s|
|MobileNet|	73.25|	72.77|	19.0|	3.45|	9m 13s|
|0.75 MobileNet|	71.92|	71.60|	10.8|	1.95|	9m 3s|
|Shallow MobileNet|	71.44|	70.38|	13.6|	2.12|	9m 14s|

Table 1. The result of Experiment 1. MobileNet reduces accuracy slightly, but needs much less computation, parameters, time. The unit of computational cost (Comp cost) and the number of parameters (# Param) is a million, and I used the equation in the paper to compute the computational cost.


## Experiment 2. Model Shrinking Hyperparameters (sec 4.2) (dataset : STL10)
Experiment 2 is about two model shrinking hyperparameters, width multiplier (α) and resolution multiplier (ρ). To see the effect of each parameter, I used different dataset from the experiment 1. The reason is that CIFAR10 is not suitable for this experiment. I could not reduce the resolution of image anymore because a 32x32 image becomes 1x1 feature map when it goes through the MobileNet. Therefore in experiment 2, STL10 dataset is used which has 96x96 images but less training images per class.
The result of the experiment is summarized in the Table 2 and 3. The overall accuracy is much lower than experiment 1 because of the lack of the training data. However, we can see the same tendency with the paper. In Table 2, accuracy drops off smoothly between 1 and 0.75, but it drops fast at 0.5 and 0.25 which means the architecture is too small. In Table 3, accuracy drops off by 4.47% and 2.81% as ρ goes smaller. From them, we can see the accuracy, computation and size trade-offs of each multiplier. More detailed results can be found in ‘Experiment2-STL10.ipynb’

|α|	Val. acc|	Test acc|	Comp cost|	# Param|
|:-----:|:-----:|:-----:|:-----:|:-----:|
|1|	49.90|	48.98|	19.0|	3.45|
|0.75|	48.30|	48.49|	10.8|	1.95|
|0.5|	46.65|	43.35|	4.89|	0.875|
|0.25|	45.00|	40.68|	1.30|	0.225|

Table 2. The result of Experiment 2 with width multiplier. Resolution multiplier is set to 1 for all experiment. The unit of Comp cost and # Param is same with Tabel 1.

|ρ|	Val. acc|	Test acc|	Comp cost|	# Param|
|:-----:|:-----:|:-----:|:-----:|:-----:|
|1|	49.90|	48.98|	19.0|	3.45|
|2/3|	47.40|	44.51|	0.023|	3.45|
|1/3|	41.70|	41.70|	0.005|	3.45|

Table 3. The result of Experiment 2 with resolution multiplier. Width multiplier is set to 1 for all experiment. The unit of Comp cost and # Param is same with Tabel 1.

