# SVD and NMF (Non-negative Matrix Factorization) on face images. 

The purpose of this study is to compare matrix factorization techniques through a facial recognition study. For this purpose, the CBCL database, originally used in (Daniel D Lee and H Sebastian Seung. Learning the parts of objects by non-negative matrix factorization. Nature, 401(6755):788–791, 1999.), have been used.

Dataset had been divided in to test and train set. I have seen that the dataset should be formed in a way which each column represents a picture. To do that each picture should be read by OpenCV as one channel and then the image matrix (19x19) should be flattened. These flattened vectors then concatenated together to create the matrix X which has the shape 361x2429 where there are 2429 different pictures in train set and 361 comes from the multiplication 19*19.

**Singular Value Decomposition (SVD)**

First, I have used the Python library NumPy for SVD to extract the U and V matrixes with the singular values. Afterwards I have plotted the singular values as follows.

![asdsadsa](https://user-images.githubusercontent.com/48417171/120924793-11e21280-c6de-11eb-8a1b-78c53cc30029.png)

From the figure above it can be seen that the singular values are ordered, which means the first singular values is the biggest singular value and the next singular is the second biggest values. In other words, the singular values are ordered from biggest to the smallest. As expected, the first singular values are noticeably big compared to the other singular values. The first singular value is for example equal to 126419 where the 50th value is equal to only 1511. That means the first singular values are more important and we can get a good estimation of the X matrix by using few numbers of singular values. I have found the energy value by summing the accumulation of the square of these singular values. Afterwards, I have normalized energy by dividing it to the max value. The plot of the normalized energy is as follows.

![image](https://user-images.githubusercontent.com/48417171/120926068-114c7a80-c6e4-11eb-8c57-423ab408e830.png)

From the figure above it can be seen that the accumulated sum’s increase rate decreases as new singular values added. This is because the first singular values are way bigger than the rest. This also includes that we can get a good approximation of the X matrix by using a small number of singular values because the curve gets near to the value of 1 quickly in the start. 

From the normalized energy I have seen that to get the normalized energy equal to 0.9 first 1, to 0.95 first 3 and to 0.99 first 29 singular values are enough. I have generated the first image by using these singular values and corresponding U, V.T columns. The results are as following.

![image](https://user-images.githubusercontent.com/48417171/120926098-35a85700-c6e4-11eb-97b7-ec38198819a0.png)
![image](https://user-images.githubusercontent.com/48417171/120926099-37721a80-c6e4-11eb-8b8f-c2085f9b89f5.png)
![image](https://user-images.githubusercontent.com/48417171/120926102-393bde00-c6e4-11eb-8886-aedb4616f6f8.png)

From these figures it can be seen that as we use more singular values and the corresponding columns and rows of the U, V matrixes, the reconstructed image gets closer to the original image. That is because most of the information is hold by low number of singular values and column rows of U, V matrixes. In other words, low number of weights and features. Therefore, I have plotted the feature, which is columns of U, where I have used the I_90 index. Since I_90=1 I have only plotted first feature.

![image](https://user-images.githubusercontent.com/48417171/120926140-58d30680-c6e4-11eb-93ba-b9de86fb04eb.png)

The importance rates of features are also ordered, which means the first feature is the most important one and the second one is the second most important feature. As the number indicating which column the feature corresponds to, the importance of the feature decreases. The first feature therefore is the most important feature which looks like a sketch of a face. It is not a particular face, but It seems this is roughly how all faces are structured. Note that these features are not localized features but distributed features because they hold information about the whole of the image. For example, this feature has dark and bright parts, but the darks values are not equal to 0. That means when we multiply this with a weight the output is not 0 which means dark areas also hold information. Furthermore, U matrix is not non-negative valued, and the features are also not non-negative valued. As mentioned before this means the dark areas that we see in the picture is not 0 but mainly negative and holds information. 

**Non-negative Matrix Factorization (NMF)**

First, I have coded the Hals algorithm with the Two-Block Coordinate Descend algorithm according to the given pseudo codes. 

![image](https://user-images.githubusercontent.com/48417171/120926168-786a2f00-c6e4-11eb-9135-ab286622157b.png)

![image](https://user-images.githubusercontent.com/48417171/120926169-7bfdb600-c6e4-11eb-8b60-3d5c378b36ad.png)

I have initialized the W and H matrixes by using SVD based initialization. Furthermore, as stopping criterion I have used two methods, firstly the user can determine the maximum number of update iteration, secondly the following criterion is implemented in the code. 

![image](https://user-images.githubusercontent.com/48417171/120926175-8455f100-c6e4-11eb-93e4-825c54e41c57.png)


Note that U^((l))is the U after l iteration. I have run the Hals algorithm and the error curve vs the iteration is as following.


Hals Update - Error vs Iteration Curve
![image](https://user-images.githubusercontent.com/48417171/120926434-89677000-c6e5-11eb-9121-496640e72365.png)

From the Paper "The Why and How of Nonnegative Matrix Factorization", figure 3
![image](https://user-images.githubusercontent.com/48417171/120926445-94ba9b80-c6e5-11eb-8825-3b9a6d9034ba.png)

From the first figure we can see that my algorithm is able to minimize the error function indicates that it can learn features from the dataset. Also, the curve of Hals in both algorithms looks so similar. The difference is that the error is lower in the second figure but we cannot compare the values because the error metrics are different. However, we can compare the shapes of the curves and they are remarkably similar. This curve indicates that with small number of iterations our algorithm is able to extract nice features. To make the error smaller after some specific number of iterations the algorithm needs a much more time/iteration, where the error gets smaller slowly. 

After seeing this curve, I have decided to compare the original and the reconstructed image with rank=9. The result is as following. 
![image](https://user-images.githubusercontent.com/48417171/120926519-e105db80-c6e5-11eb-9e17-dd644b0c59ba.png)

Lastly, I have plotted the 9 columns of the W matrix, i.e., the extracted features as following.
![image](https://user-images.githubusercontent.com/48417171/120926523-ebc07080-c6e5-11eb-8cd0-74147eeb74e0.png)

The W matrix and therefore the features are all non-negative due to the max operation we use in the training steps. That means the black areas we see in these feature pictures are equal to 0. That means these features do not hold information on some points. No matter with what weight we multiply this matrix those areas will be equal to 0, this is another way of showing that these features do not have information in some areas. Thus, these features are localized features they only hold information in some regions. For example, 8th feature is mainly responsible with having information only on the right side of a face. Whereas 9th feature is the opposite of it.


**Image Recovery from Noisy Data**

For this part I have first read the test set and added uniform noise as specified in the homework. I have than plotted the first picture with different noise levels as following. 

![image](https://user-images.githubusercontent.com/48417171/120926558-0c88c600-c6e6-11eb-9143-9cfd3be34922.png)

From this figure we can see that when the noise increases it gets less likely to the original image. The noise of “Rand*1” results with a not bad result. However, the noise “Rand*25” corrupts the image a lot. For curiosity I have taken the noisy image of “Rand*10” and tried to reconstruct it with r=45 value with both SVD and NMF. The result which is satisfying and showing that we can denoise these images up to some point, is as following. 

![image](https://user-images.githubusercontent.com/48417171/120926574-20ccc300-c6e6-11eb-97d9-9e0f18c28d7b.png)

Next, I have reconstructed whole images on the dataset. Then I have calculated the F-norm of the error |(|Img_original-Img_reconstructed |)|_F for whole images and take the mean along this 472-long vector.

![image](https://user-images.githubusercontent.com/48417171/120926591-35a95680-c6e6-11eb-97aa-f75be902bfb0.png)
![image](https://user-images.githubusercontent.com/48417171/120926593-393cdd80-c6e6-11eb-8422-161005543a81.png)
![image](https://user-images.githubusercontent.com/48417171/120926595-3b9f3780-c6e6-11eb-9b20-d175c39f1249.png)

From these figures we can see that the SVD error is mainly smaller than the NMF for different values of rank. This was expected because SVD is the most efficient way to reconstruct a matrix with small rank values.

Also, we can see that the learned features from the train dataset is able to reconstruct denoised images from noised images until a point, because the error is decreasing for all figures. Furthermore, we can see that NMF error curve is not smooth, i.e., there are points were even the rank increases the error does not decrease, my guess is the used algorithm sometimes converges/stops to a better point for lower values of rank than high value of rank value. Overall, the two error curves decrease as the used rank value increases. Furthermore, we can see by comparing these figures the error increases as we increase the noise values. This also as expected because as the noise increases the combination of learned features of the train dataset and the noisy images does not result in a noisier output image. Also, from the error curves we can see that the decrease in the error also slows down as rank increases. So, it is not always reasonable to not use a high rank value if we want to reconstruct a Matrix. A value between 20 and 40 can be used as effective rank for this dataset if we want to have a low rank value. 

**Image Recovery from Masked Data**

For this part I have first masked the images from test-set that is I have entry-wised multiplied the images with a matrix S which is defined as following.
![image](https://user-images.githubusercontent.com/48417171/120926661-815c0000-c6e6-11eb-9061-776300aac29b.png)

I have examined that the shadow masked images have a shadow in the right side of the image. Two examples are as following. 

![image](https://user-images.githubusercontent.com/48417171/120926667-8751e100-c6e6-11eb-951c-858ca0895ce3.png)

Afterwards I have followed the procedure I have done in the previous section and find the F-norm of the error (difference between original image and reconstructed from noisy image) for different rank values. The result is as following. 

![image](https://user-images.githubusercontent.com/48417171/120926675-8de05880-c6e6-11eb-84e6-8b1a0a5c87de.png)

From this figure it can be seen that the error decreases as the rank values increases and as we see from the previous results the SVD generally performs better than NMF. That is probably because SVD is the most efficient way to reconstruct a matrix. Furthermore, the error decrease rate decreases as the rank values increases so again an effective rank value can be chosen between 20-40. Lastly, the error we see here in reconstruction of masked images are higher than the reconstruct random noise. I think the reason is, the features are able to reconstruct the images if the noisy image has a distributed noise where the shape of the face gets distorted. By features we can correct the distortion in the shape/structure of the face. However, in masked images the there is a shadow in the right side and this shadow is not a distortion in the shape of the face it is just making the right side darker. Reconstructing with features therefore performs worse, the shape of the face is not distorted but it is darker, and features are not that successful to fixing that shadow as they do with fixing the shape distortion of the random noise. 

**Papers Udes**
N Gillis. The why and how of nonnegative matrix factorization, 2014. arXiv Preprint arXiv:1401.5226v2, 2014.

Nicolas Gillis et al. Nonnegative matrix factorization: Complexity, algorithms and applications. Unpublished doctoral dissertation, Université catholique de Louvain. Louvain-La-Neuve: CORE, 2011.

Daniel D Lee and H Sebastian Seung. Learning the parts of objects by non-negative matrix factorization. Nature, 401(6755):788–791, 1999.

