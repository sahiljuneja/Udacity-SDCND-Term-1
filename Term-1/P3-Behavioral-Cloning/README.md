# Project 3: Use Deep Learning to Clone Driving Behavior

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/sample_images.png
[image2]: ./output_images/sample_images_hsv.png
[image3]: ./output_images/dataset_distribution_original.png
[image4]: ./output_images/dataset_distribution.png

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

Following is my model architecture (`model.py` lines 151-167). It contains 2 convolutional layers with 5x5 and 3x3 filter sizes, and depths as 16 and 32. "ELU" layer was used
as activation layer to introduce nonlinearity.

| Layer         		|     Description	        								 	   | 
|:---------------------:|:----------------------------------------------------------------:| 
| Input         		| 32x16x1 HSV (S channel) image   							 	   | 
| Lambda		     	| Normalization layer										 	   |
| Convolutional Layer   | 1x1 stride, 'VALID' padding, layer_depth = 16, filter_size = 5x5 |
| ELU					| Activation Layer											 	   |
| Max pooling	      	| 2x2 stride					 							 	   |
| Convolutional Layer   | 1x1 stride, 'VALID' padding, layer_depth = 32, filter_size = 3x3 |
| ELU					| Activation Layer											 	   |
| Max pooling	      	| 2x2 stride					 							 	   |
| Flatten		      	| Reshape for Dense/FC Layers					 	   			   |
| Fully connected		| width = 32			     									   |
| ELU					| Activation Layer											 	   |
| Dropout		      	| 0.5 probability				 							 	   |
| Fully connected		| Output layer, width = 1										   |
 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (`model.py` line 166). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code lines 180-185). 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

Following are the parameters I selected for tuning the model:

* Number of epochs = 20 (line 144)
* Batch Size = 64 (line 145)
* Optimizer = Adam Optimizer, learning rate = 0.0001 (lines 171-173)
* Metric = Mean Squared Error (lines 171-173)

####4. Appropriate training data

I collected around 2 laps of driving data (one way, no reverse driving). After a basic working model, I noticed that the car was driving almost on the outer edge of one small area on the track. I recorded couple
of sets of recovery data for that area (dirt patch turn after the bridge).

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

I went through several iterations of models using Udacity's data initially. My models were working to an extent, but it seemed complicated. I finally decided to collect my own data,
which was very small in size, and built a simple model to train over it.

My focus was on collecting quality data from Track 1. I only recorded two laps (no reverse driving) and for most of my driving I kept the car in the middle of the road.

Here are some sample images, with associated steering angles (Ground Truth). I resized the images to shape of (32, 16) in `preprocess_data()` function in lines 71-81.

![alt text][image1]

Another step in preprocessing my images was to only utilize the S channel in the HSV color space. This even helped distinguish the dirt patches well.

![alt text][image2]

I ended up recording `1536` images, which also included couple of sets of recovery data on a small patch of the track, right after the bridge, next to the dirt patch.

Since most of my driving was straight, this is what my steering angles distribution looked like.
![alt text][image3]

I decided to remove 75% of the steering angles which were equal to 0 (line 34). Since I wanted to include center, left, and right images for my model, I included the steering angles
for the left and right images (liens 37-46), by adding a shift of 0.25 to the steering angles for left images, and by subtracting a shift of 0.25 from the steering angles for 
right images. The code for this in lines 49-55. The shift added to the steering angles here is indicative of the offset the left and right cameras cause with respect to the center camera.

Following is the steering angle distribution based on above augmentation. 

![alt text][image4]

After adding images and steering angles for all the cameras, and removing some values from the steering angles, I had 1785 training samples. Out of these I used 15% for a validation set.
I had `1517` training samples, and `268` validation samples [line 61]. I shuffled this dataset as well in line 60.

When I initially started to train my models (with Udacity data) I noticed that validation loss, and the mean absolute error values were not indicative of how well the model would do. 
This is understandable since our data is based on one track only which has lots of similar patches, so the loss values wouldn't define how well the training was that well. 

So for training my model [lines 180 to 185], I decided to have a callback function [lines 177 to 178] that would save my model whenever I got the lowest validation loss value over all epochs.

After training, I ran `drive.py` and observed my results in the simulator's autonomous mode. The car drove quite well over the track for multiple laps before I stopped the simulator. I
recorded one lap using `video.py` and it's saved as `video_record.mp4`. Unfortunately, do to my system being quite old, the recording while the simulator was running affected the car and 
it seems to oscillate a bit on the track, although it remains inside the lane at all times. I ran the simulator with a resolution of 400x300 at the "Fastest" graphics quality settings
