# Wasserstein GAN

This paper proposes Wasserstein GAN which uses Wasserstein distance (or its estimate) as loss function. The authors claim that the Wasserstein distance is a meaningful loss metric which can show the progress of training, and it has improved stability. To demonstrate, I reproduced two experiments in the paper. While reproducing, I leveraged dataset and hyperparameters as the paper describes as same as possible. I referred the structure of DCGAN from a prior work [1]. I trained each model with one TITAN RTX.

## Experiment 1. Meaningful loss metric
In section 4.2, the paper illustrates how the loss correlates well with the quality of the generated sample. I confirmed the statement using DCGAN structure. In Figure 1, we can see the training curve and generated samples from some iterations. As training progresses, the Wasserstein estimate decreases and the quality of samples are getting better. The curve has some noise because I did not used a median filter as the authors did, but it still shows same tendency. Although I took only 4 samples out of 64 samples from each iteration due to the lack of the space, the whole samples from some iterations are included in submission file.

<table>
  <tr>
    <td> ![DCGAN](https://user-images.githubusercontent.com/52485688/87549268-956ecb80-c6e8-11ea-920b-4e541bc0c365.png) </td>
    <td>![figure1](https://user-images.githubusercontent.com/52485688/87549272-97388f00-c6e8-11ea-8033-1be12ca2ffe2.png)</td>
  </tr>
  <tr>
    <td colspan="2">Figure 1. Training curve and samples at different stages of training. As the training progresses, the quality of samples is improved and the curve goes down. Therefore, we can see that the curve shows correlation between lower error and better sample quality.</td>
  </tr>
</table>


|![DCGAN](https://user-images.githubusercontent.com/52485688/87549268-956ecb80-c6e8-11ea-920b-4e541bc0c365.png)|![figure1](https://user-images.githubusercontent.com/52485688/87549272-97388f00-c6e8-11ea-8033-1be12ca2ffe2.png)|
|---|---|
|Figure 1. Training curve and samples at different stages of training. As the training progresses, the quality of samples is improved and the curve goes down. Therefore, we can see that the curve shows correlation between lower error and better sample quality. ||
 
| Header ||
|--------------|
| 0 | 1 | 

## Experiment 2. Improved stability
In section 4.3, the paper shows the model’s improved stability using DCGAN generator without batch normalization (DCGAN-BN), and a 4-layer ReLU-MLP (MLP-4). DCGAN-BN produced plausible fake images and showed decreasing Wasserstein estimate curve. MLP-4, on the other hand, was not trained well but somehow generated some bedroom-like images. Therefore, I implemented 5-layer ReLU-MLP (MLP-5) additionally to test whether MLP generator can be trained with WGAN method or not. The result shows MLP generator can be trained with WGAN method.
Each model was trained for 600,000 generator iterations same as experiment 1. Training curve and samples of each models are included in 'plot.ipynb' From the training curve of DCGAN-BN, MLP-4 and MLP-5, we can see that the DCGAN with Batch Normalization shows best performance. 

![figure2](https://user-images.githubusercontent.com/52485688/87549284-999ae900-c6e8-11ea-956b-f12dbd722b86.png)
Figure 2. Samples from each model at generator iteration=600000. The result shows WGAN method is more stable than standard GAN method. You can find the generated samples with standard GAN method in the paper.

GAN is a powerful subclass of generative model. However, it is hard to interpret whether the model is trained well or not with standard GAN. In this paper, the authors propose a novel training method, which is using Wasserstein distance as a loss function of GAN training. From this method, we can check if the model is trained well. It would help users to notice if there is wrong setting or hyperparameter.
