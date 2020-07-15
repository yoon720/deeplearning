# Fully Convolutional Networks for Semantic Segmentation

In this paper, the authors build fully convolutional network that take input of arbitrary size and produce correspondingly sized output with efficient inference and learning. They show that we can adapt classification networks into fully convolutional networks and transfer their learned representations. In addition, they define a skip architecture that combines semantic information from a deep, coarse layer with appearance information from a shallow, fine layer to produce accurate and detailed segmentations.
Therefore, I reproduced two experiments (Table1, Table2) from the paper to show the concepts suggested in this paper. I followed descriptions in paper as possible, and used PASCAL VOC 2011 dataset

## Experiment 1. From classifier to dense FCN (Sec 4.1, Table 1)
In the first experiment, the authors convolutionize three proven classifiers, which are AlexNet, VGG16, GoogLeNet. I used pretrained model from pytorch to initialize each FCN and trained for 200 epochs because the paper says networks need at least 175 epochs to converge. In the paper, the FCN-VGG16 shows the best performance, and FCN-AlexNet follows. However, I got the best performance with FCN-AlexNet. The figures and table shows the result, and more detailed result and visualization can be found in ‘Exp1-result.ipynb’ 

Solarized dark             |  Solarized Ocean
:-------------------------:|:-------------------------:
![fig1](https://user-images.githubusercontent.com/52485688/87548598-b1be3880-c6e7-11ea-80a0-0ec7b08404e3.png)  |  ![fig2](https://user-images.githubusercontent.com/52485688/87548604-b387fc00-c6e7-11ea-832b-63f8485d1c1e.png)



|-|	Alex|	VGG|	GoogLe|
|:----:|:----:|:----:|:----:|
|mIU|	76.03|	72.61|46.37|
|Depth|	8|	16|	22|
|Param|	57M|	134M|	6M|
|Max stride|	32|	32|	32|


## Experiment 2. Combining what and where (sec 4.2, Table 2)
In the second experiment, the authors suggest networks that learn to combine coarse, high layer information with fine, low layer information. FCN-32s is same as FCN-VGG16 in experiment 1 and FCN-16s, FCN-8s are skip architectures of FCN-32s. They are initialized with best FCN-32s model as the paper describes. I trained FCN-32s and FCN-16s for 200 epoch, but FCN-8s for 180 epochs due to lack of time. Metrics and segmentation results are as follows. Although I initialized with best FCN-32s, FCN-16 and FCN-8 showed worse performance than FCN-32s. I think this is because of the initialization of new layers were bad. The figures and table shows the result, and more detailed result and visualization can be found in ‘Exp2-result.ipynb’ 

![fig3](https://user-images.githubusercontent.com/52485688/87548610-b551bf80-c6e7-11ea-912d-36d6f6dd94b1.png)

|-|	pixel acc.|	mean acc.|	mean IU|	f. w. IU|
|:----:|:----:|:----:|:----:|:----:|
|32s|	94.90|	85.82|	72.61|	82.92|
|16s|	94.24|	84.55|	71.10|	82.04|
|8s|	93.81|	83.65|	70.06|	81.40|


By convolutionizing the fully-connected layers of classifier, we can extend the classification nets to segmentation. Also, we can improve the architecture with multi-resolution layer combinations.
