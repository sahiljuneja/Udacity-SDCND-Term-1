## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/image_samples.png "Visualization1"
[image2]: ./output_images/data_distribution.png "Visualization2"
[image3]: ./output_images/loss_accuracy.png "Loss"
[image4]: ./output_images/test_images.png "Test"
[image5]: ./output_images/test_images_pred.png "Test pred"
[image6]: ./output_images/confusion_matrix.png "Confusion Matrix"
[image7]: ./output_images/softmax0.png
[image8]: ./output_images/softmax1.png
[image9]: ./output_images/softmax2.png
[image10]: ./output_images/softmax3.png
[image11]: ./output_images/softmax4.png
[image12]: ./output_images/softmax5.png
[image13]: ./output_images/softmax6.png
[image14]: ./output_images/softmax7.png

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it!

All the code of the project is in the `Traffic_Sign_Classifier_CNN.ipynb` jupyter notebook.

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the `Code Cells 2 and 3` of the notebook.  

I used the standard python libraries/methods to calculate the following - 

* The size of training set is `34799`
* The size of test set is `12630`
* The shape of a traffic sign image is `(32, 32, 3)`
* The number of unique classes/labels in the data set is `43`

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in `Code Cells 4 and 5` of the notebook.  

I initially looked at 10 random images obtained from the training dataset to get an idea on the quality of images we have. 

![alt text][image1]

I then plotted a bar plot for the data distribution across all the 43 classes. The bar plot was quite informative on what kind of results can be expected after training the model [which I will
discuss later].

![alt text][image2]
###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in `Code Cell 6` in the notebook.

For some of my initial models, I had grayscaled my images. However, in order to explore the potential of CNNs, I eventually used RGB images, and only normalized the images.

I normalized the training, validation, and test sets by subtracting the mean of the images and dividing by the standard deviation. 

Normalizing the data is important since it helps primarily in helping the gradient descent optimizer to converge faster by limiting the broad range of feature values by centering them
around zero, and scaling them based on standard deviation [as I have implemented it].

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

Since the training, validation, and test dataset was already provided to us by Udacity, I simply loaded the data into memory in `Code Cell 2`. The dataset was divided into

* `34799` Training Samples
* `12630` Test Samples
* `4410` Validation Samples


After normalizing the above data, using sklearn's `LabelBinarizer()` method I obtained the one-hot encoded labels for the training and validation datasets in `Code Cell 7`.

In `Code Cell 8` using sklearn's `Shuffle()` method, I randomly shuffled my features and labels for training and validation sets. Thereby reducing chances of overfitting to a subset.


In `Code Cell 9` I reshaped the the datasets to feed them into TensorFlow.

I didn't augment my data for this model.


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is defined in the `convnet_model()` function in `Code Cell 11` and all the associated helper functions are in `Code Cell 10` of the notebook.

My final model consisted of the following layers:

| Layer         		|     Description	        								 | 
|:---------------------:|:----------------------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							 			 | 
| Convolution 3x3     	| 1x1 stride, 'VALID' padding, width = 32, filter_size = 5x5 |
| RELU					| Activation Layer											 |
| Max pooling	      	| 2x2 stride					 							 |
| Drouput		      	| 0.5 probability				 							 |
| Convolution 3x3	    | 1x1 stride, 'VALID' padding, width = 64, filter_size = 5x5 |
| RELU					| Activation Layer											 |
| Max pooling	      	| 2x2 stride					 							 |
| Drouput		      	| 0.5 probability				 							 |
| Fully connected		| width = 1024		     									 |
| RELU					| Activation Layer											 |
| Drouput		      	| 0.5 probability				 							 |
| Fully connected		| Logits (output layer)										 |
| Softmax				| Activation Layer											 |

 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for the various parameters/hyperparameters is in `Code Cell 13`. 
The code for the inputs to the TF model, the cost function and optimizer, and for prediction/accuracy calculation is in `Code Cell 14`. 
The code for training the model, displaying the loss values and the accuracy percentages is in `Code Cell 15`. 

I went through several models, initially starting from a simple DNN (no CNNs) through which I obtained good results. I later moved to CNNs and experimented with numerous iterations
while discovering several issues with my implementations as well. Beyond a point it became obvious that bulk of the work could be improved by data augmentation rather than fine tuning 
the model. Data augmentation would have helped to balance out the data distribution. I decided to stick with a relatively simple model as shown above for my submission. 

The hyperparameters I used - 
* Learning Rate = 0.001
* Number of Epochs = 20
* Batch Size = 64
[Other Parameters are already covered in the model table above]

I utilized the `Adam Optimizer` since that yielded better (quicker) results than the SGD. The cost function was calculated by computing the cross entropy after applying the softmax function.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is included in `Code Cell 15`

My final model results were:
* training set accuracy of `96.5%`
* validation set accuracy of `94.2%` 
* test set accuracy of 94.8%`

I initially implemented a basic DNN to begin with. However after a bit of experimenting it was obvious that I couldn't go beyond a particular point in terms of achieving good test accuracies.
I then implemented a CNN based model. In this model I experimented with most of the parameters - the layer widths, the strides, maxpooling, dropouts, filter sizes, number of layers,
learning rate, number of epochs being my main focus. The attached file `P4 config.pdf` shows some of the results of this experimentation with an older dataset for this project. That
dataset was resulting in some problems with the validation set accuracy.

I iterated through the above to understand what changing different parameters does. But without any kind of data augmentation, I couldn't go beyond 95% test accuracy. 

For this submission I utilized a simple model with just 2 convolutional layers to get good results as described above. A learning rate of 0.001 and 20 epochs gave good results without
overfitting (which I noticed by increasing the epochs further). 

The following image shows the loss and accuracy for training and validation sets after training the model.
![alt text][image3]

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

The code for this step is in `Code Cell 17`. The images were preprocessed in `Code Cell 18`.

Here are Traffic Sign images that I found on the web:

![alt text][image4]

As per me, the image of the "Bike Crossing" would be difficult for classification because of additional text in the image. Also, the image with the speed limit of 30 is skewed, and since
I haven't performed any data augmentation it won't get classified correctly. 

After calculating the results from my training, I plotted a Confusion Matrix as shown below.

![alt text][image6]

Based on this matrix, it's expected that my model won't classify well for most of the speed limit signs. It should however classify the slippery road, stop sign, and do not enter sign correctly
with high prediction accuracy.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in `Code Cell 19`.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Pedestrians      		| Speed Limit (70 kmph)							| 
| Slippery Road     	| Slippery Road 								|
| Yield					| Yield											|
| Do Not Enter	      	| Do Not Enter					 				|
| Speed Limit (50 kmph) | Speed Limit (80 kmph)      					|
| Bike Crossing			| Stop					      					|
| Speed Limit (30 kmph) | Priority Road			      					|
| Stop					| Stop					      					|

The model's accuracy for the above images was calculated in `Code Cell 20` and it was 50%. Based on my analysis of these images, and the confusion matrix from the model training, 
this accuracy is not surprising. The test accuracy obtained on the original test data was 94.8% so it's likely the model is overfitting. Considering I didn't augment my data, the overfitting
is not surprising and the relatively poor performance on the above set is understandable. Plus, not all the images above are from the German Traffic Signs dataset, hence the discrepancy.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in `Cells 21 and 22`.

Following are the top 5 softmax probabilities for each prediction -
```
Image 0, Top 5 Probabilities: [0.97455001, 0.025210001, 0.00023999999, 0.0, 0.0]
Image 1, Top 5 Probabilities: [0.99996001, 3.9999999e-05, 0.0, 0.0, 0.0]
Image 2, Top 5 Probabilities: [1.0, 0.0, 0.0, 0.0, 0.0]
Image 3, Top 5 Probabilities: [1.0, 0.0, 0.0, 0.0, 0.0]
Image 4, Top 5 Probabilities: [0.90333998, 0.037769999, 0.033709999, 0.02101, 0.0038000001]
Image 5, Top 5 Probabilities: [0.9874, 0.00416, 0.0040500001, 0.00147, 0.00136]
Image 6, Top 5 Probabilities: [0.85771, 0.14195, 0.00019999999, 9.0000001e-05, 3.9999999e-05]
Image 7, Top 5 Probabilities: [1.0, 0.0, 0.0, 0.0, 0.0]
```

Following are the bar charts that further describe the above probabilities.

* Image 0: Ground truth is a "Pedestrians Crossings" image, whereas the model classifies it incorrectly as a "Speed Limit (70kmph)" sign with a 0.97 probability. This is not surprising
given the few data points for this class in our dataset.
![alt text][image7]

* Image 1: Ground truth is a "Slippery Road" image, and the model classifies it correctly as a "Slippery Road" sign with a full confidence (1.0 probability). 
![alt text][image8]

* Image 2: Ground truth is a "Yield" image, and the model classifies it correctly as a "Yield" sign with a full confidence (1.0 probability).
![alt text][image9]

* Image 3: Ground truth is a "Do Not Enter" image, and the model classifies it correctly as a "Do Not Enter" sign with a full confidence (1.0 probability).
![alt text][image10]

* Image 4: Ground truth is a "Speed Limit (50kmph)" image, whereas the model classifies it incorrectly as a "Speed Limit (80kmph)" sign with a 0.9 probability. As per the confusion matrix,
the model didn't learn well from the "Speed Limit" images. Perhaps better dataset (and augmentation) would help here.
![alt text][image11]

* Image 5: Ground truth is a "Bike Crossings" image, whereas the model classifies it incorrectly as a "Stop" sign with a 0.98 probability. This is not surprising
given the few data points for this class in our dataset.
![alt text][image12]

* Image 6: Ground truth is a "Speed Limit (30kmph)" image, whereas the model classifies it incorrectly as a "Priority Road" sign with a 0.85 probability and as "Road Work" with 0.15
probability. As per the confusion matrix, the model didn't learn well from the "Speed Limit" images. Also, as I mentioned earlier, the image of the sign is skewed. Without data augmentation,
the model wouldn't necessarily classify these correctly.
![alt text][image13]

* Image 7: Ground truth is a "Stop" image, and the model classifies it correctly as a "Stop" sign with a full confidence (1.0 probability).
![alt text][image14]
