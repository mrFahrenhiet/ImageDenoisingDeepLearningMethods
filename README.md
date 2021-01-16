# ImageDenoisingDeepLearningMethods
### [Colab Notebook Link](https://colab.research.google.com/drive/1IEtq8ScXvYR79R1guzt-dJ3hdTD4fIhP?usp=sharing)

## Image Denoising:
Image denoising thus plays an important role in current image processing systems. Image denoising is to remove noise from a noisy image, so as to restore the true image. However, since noise, edge, and texture are high frequency components, it is difficult to distinguish them in the process of denoising and the denoised images could inevitably lose some details. Image denoising is a classic problem and has been studied for a long time. However, it remains a challenging and open task. The main reason for this is that from a mathematical perspective, image denoising is an inverse problem and its solution is not unique. <br /><br />
**Mathematically, the problem of image denoising can be modeled as follows**:<br />
`y = x + n                          y = observed noisy image`                                                                           
                                    `x = image to be derived` <br />
                                    `n =additive white Gaussian noise (AWGN) to be removed from y`<br />
**This Repository contains implementation of 5 top level research papers (From A\* Conferences and Transactions) on Image Denoising**<br />
**NOTE: All the implementations were trained and tested on Chest X-Ray Dataset ([CoronaHack-Chest](https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset))** <br/>

### When Image Denoising Meets High-Level Vision Tasks: A Deep Learning Approach ([Ding Liu et al.](https://doi.org/10.24963/ijcai.2018/117))
**Summary and Novel Contribution:**
Author proposed a novel technique combining both denoising autoencoder and high-level vision tasks to train the model. The training strategy of this model is quite different it uses a combination of the losses obtained from its denoising network and its segmentation network to train both the denoising and the segmentation network. The denoising network uses deep level of down sampling and up sampling of the image. The denoising uses a series of feature contraction and expansion operations on the image for denoising. <br />
![image ding liu test](https://github.com/mrFahrenhiet/ImageDenoisingDeepLearningMethods/blob/main/media/dingliu.png) <br />
Ground Truth &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Nosiy Image &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Predicted Image

### Residual Dense Network for Image Restoration ([Yulun Zhang et al](https://doi.org/10.1109/TPAMI.2020.2968521))
**Summary and Novel Contribution:**
The author proposed a dense residual network to find hierarchical features from all the convolutional layers. A dense residual block allows you to extract abundant local features via densely connected layers. The proposed a multi-purpose model which can be used for variety of image restoration tasks including increasing image resolution, image denoising and image deburring.  <br />

![image residual dense test](https://github.com/mrFahrenhiet/ImageDenoisingDeepLearningMethods/blob/main/media/residualDense.png) <br />
Ground Truth &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Nosiy Image &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Predicted Image

### Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising ([Kai Zhang et al.](https://doi.org/10.1109/TIP.2017.2662206))
**Summary and Novel Contribution:**
Author proposed a novel technique combining both denoising autoencoder and high-level vision tasks to train the model. The training strategy of this model is quite different it uses a combination of the losses obtained from its denoising network and its segmentation network to train both the denoising and the segmentation network. The denoising network uses deep level of down sampling and up sampling of the image. The denoising uses a series of feature contraction and expansion operations on the image for denoising. <br />
![image dncnn test](https://github.com/mrFahrenhiet/ImageDenoisingDeepLearningMethods/blob/main/media/dncnn.png) <br />
Ground Truth &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Nosiy Image &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Predicted Image

### Medical image denoising using convolutional denoising autoencoder ([Gondara et al.](https://arxiv.org/abs/1608.04667))
**Summary and Novel Contribution:**
The author proposed stacked autoencoder that work well on small sample sized data. The author claims that deeper models require a larger sample size and are mostly not suitable for medical imaging. Hence the proposed technique of simpler stacked convolutional auto-encoders is used here to denoise the image. The proposed networks consists of a stacked sequential convolutional auto encoder. <br />
![image gondara 16 test](https://github.com/mrFahrenhiet/ImageDenoisingDeepLearningMethods/blob/main/media/gondara.png) <br />
Ground Truth &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Nosiy Image &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Predicted Image

### Extracting and Composing Robust Features with Denoising Autoencoder ([Vincent et al.](https://dl.acm.org/doi/10.1145/1390156.1390294))
**Summary and Novel Contribution:**
The author proposed that they can improve the robustness of the internal layers by introducing some noise in the data and then training an autoencoder network to remove that noise. This was the first paper to indicate and introduce the use of autoenders for image denoising. This network first sample downs the image using dense layers and up sample the image again using perceptron. <br />
![image vincent 8 test](https://github.com/mrFahrenhiet/ImageDenoisingDeepLearningMethods/blob/main/media/vincent.png) <br />
Ground Truth &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Nosiy Image &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Predicted Image







### Comparitive Analysis

#### SSIM Comparision
![ssim comparision](https://github.com/mrFahrenhiet/ImageDenoisingDeepLearningMethods/blob/main/media/ssim.png) <br />

#### PSNR Comparision
![psnr comparision](https://github.com/mrFahrenhiet/ImageDenoisingDeepLearningMethods/blob/main/media/psnr.png) <br />


#### MSE Comparision
![mse comparision](https://github.com/mrFahrenhiet/ImageDenoisingDeepLearningMethods/blob/main/media/mse.png) <br />
                                                   
                                                          
                                                          

