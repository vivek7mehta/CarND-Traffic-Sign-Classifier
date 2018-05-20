
## **Traffic Sign Recognition** 

### Writeup


### 1. Basic summary of the data set:

I used the python and numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set : 34799
* The size of the validation set is 4410
* The size of test : 12630
* The shape of a traffic sign : (32, 32, 3)
* The number of unique classes/labels : 43


### 2. Exploratory visualization of the dataset.

Here is a bar chart visulization of number of traffic signs for each classes

![bar_visual.png](attachment:bar_visual.png)


Here I have displayed sample traffic signs from various classes.

![sign_visu.png](attachment:examples/sign_visu.png)


* In first visulization we can see that training set is distributed propperly for various classes.
* Second visulization shows traffic signs from different classes.

### 3. Design and Test a Model Architecture


#### Pre-processing

* I have converted images to YIQ (Y represents brighness in image, I stands for In-phase, Q stands for quadrature) color model.
* The reason for this is that RGB represents values in range of 0 to 255 for each cahnnel, while YIQ represents represents same amount of data within smaller range. Hence using this color space we can represent our image with smaller range. 
* After this I have reshaped my image to 1 dimension.
* As a last step I have normalized my dataset, so that gradient descent would work faster.

##### Before and after pre-processing (from BGR to YIQ)

RGB            |  YIQ
:-------------------------:|:-------------------------:
![before_prepros.png](attachment:before_prepros.png) | ![after_prepros.png](attachment:after_prepros.png)

* As we can see here hogh intensity colors are luminated, and darker images stays same. Here YIQ with smaller range seems to represent data as good as RGB specially when color of signs doesn't matter.


#### Model description

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 1 dimsion image   							| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x6	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6				|
| Convolution 5x5	    | 1x1 stride, VALID padding, outputs 10x10x16 									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16				|
| Flatten		|        outputs 400  									|
| Fully connected				| input 400, output 120        									|
| Dropout			| keep probability=0.5    									|
| Fully connected				| input 120, output 84     									|
| Fully connected				| input 84, output 43     									|


* This model contains two convolution layer and three fully connected layer.
* After each convolutional layer a pooling layer (max pool) is added to reduce number of parameters. Hence it also reduces spatial dimention as well as it helps to reduce overfitting.
* RELU is used as activation function for both convolutional layer.
* I have used dropout at first fully connected layer out of three fully connected layer, which helps to reduce overfitting on training data set.


#### How I trained my model:

* I have used adam optimizer which is extension of SGD (stochastic gradient descent), with learning rate 0.002
* I have added keep probability 0.5 for dropout while training, which seems to be improving accuracy.
* I have used batch size of 128.
* I have used number of epochs 40.

##### How I reached at these parameters
* While training I plotted accuracy graph of training and validation dataset. From these graph parameters can be tuned. 
* For example if accuracy is improving too fast at the begining and then if it doesn't improve much, it means that learning rate is high and we can tune that. Too much fluctuation in accuracy since begining also suggests that we should reduce learning rate.

![accuracy_graph.png](attachment:accuracy_graph.png)

* Here blue(+) represents training accuracy and red(o) represents validation accuracy.
* From this graph I was able to figure out a point where model stops improving accuracy. Also from the curve 0.02 learning rate and 40 epochs seems good.
* When I trained my model without dropout at that time training accuracy reached very high, hence I added dropout at fully conected layer to reduce overfitting.


##### Approach

My final model results were:
* training set accuracy of 0.947392290249
* validation set accuracy of 0.947
* test set accuracy of 0.944

Training and validation accuracy is calculated in 30th cell, while test accuracy is calculated in 31st cell of notebook.

Architecture selection:
* I have started with LeNet architecture.
* Since LeNet has already proven it's accuracy for handwritten digits. It also gives good results on distorted images. Since LeNet is good network with two convolutional and three fully conected layes, it seems good enough starting point for traffic signs. 
* After adding a dropout layer in existing LeNet and pre-processing data I was able to get Validation accuracy of 94.7%, and test accuracy of 94.4% which shows that model is performing good. Also this model gives good accuracy on new images as dissused in next section.



### Test a Model on New Images

#### Traffic signs from wen:

* Here are eight signs which I have tested on my model

![new_sign.png](attachment:new_sign.png)

* Here first sign in second row is not present in original data set, rest of these signs are present.

#### Model predictions

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield sign     		| Yield sign   									| 
| turn right   			| turn right										|
| Road work				| Road work											|
| Children crossing      		| Children crossing					 				|
| Moter vehicle prohibited			| speed limit (30km/h)     							|
| speed limit (30km/h) 		| speed limit (30km/h)     							|
| turn left   			| turn left										|
| bumpy road   			| bumpy road										|

* The model was able to correctly guess 7 of the 8 traffic signs, which gives an accuracy of 87.5%. One sign which was incorrectly predicted is not part of original 43 classes hence it was incorrectly predicted as speed limit 30km/h.
* If we do not consider 1 sign (which was not present in original 43 classes), then this model predicted all signs correctly. Which seems good as testing accuracy is also 94.4%

#### 3. Prediction certainity

The code for making predictions on my final model is located in the 34, 35 and 36th cell of the Ipython notebook.

Here I have displayed all images with top prediction and top five softmax probabilities.

![softmax.png](attachment:softmax.png)

