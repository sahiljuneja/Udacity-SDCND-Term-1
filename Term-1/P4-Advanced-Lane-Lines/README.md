## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undist_calib_image.png "Undistorted"
[image2]: ./output_images/test1.png "Road Original"
[image3]: ./output_images/undist_test_image.png "Road Undistorted"
[image4]: ./output_images/persp_transform.png "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the `camera_calibration.ipynb` Jupyter Notebook.  

In `Code Cell 2` of this notebook, I define the number of inner corners (9x6) in the chessboards by visual inspection. After reading in the chessboard images using python's `glob`
library, each image is converted to grayscale and passed to OpenCV's `findChessboardCorners` function which outputs the corners of each square (inner) in the chessboard.

[The following explanation is provided by Udacity and explains the process quite well]
For every image, two lists, `objpoints` and `imgpoints` are updated. 

The `objpoints` contains the (x, y, z) coordinates of the chessboard corners in the world. 
Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  
Thus,`objpoints` will be appended with a copy of a replicated array of coordinates every time I successfully detect all chessboard corners in a test image.  

`imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

In `Code Cell 4` the `objpoints` and `imgpoints` are passed into OpenCV's `cv2.calibrateCamera()` function, which computes the camera calibration and distortion coefficients.

I then use these coefficients and undistort a test image using OpenCV's `cv2.undistort()` function. Following is the result of that - 

![alt text][image1]

###Pipeline (single images)
The code for the following section is in the `Advanced_Lane_Finding.ipynb` Jupyter Notebook.

####1. Provide an example of a distortion-corrected image.
The camera matrix and distortion coefficients obtained above were used to correct for distortion in test images and the video frames. Following is a sample of that process -

![alt text][image3]

####2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

In `Code Cell 5` of the notebook, I implemented the `perspective_transform()` and the `undist_img()` functions. The `undist_img` function took in the original image, and undistorted it
and also outputted the source and destination points based on the image dimensions, that were later utilized to obtain a transform using OpenCV's `getPerspectiveTransform()` function.

The resulting source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| (0, 720)      | (0, 720)      | 
| (480, 480)    | (0, 0)      	|
| (800, 480)    | (720, 0)      |
| (1280, 720)   | (1280, 720)   |

The above transform was then used to warp the undistorted image, using OpenCV's `warpPerspective()` function, into a bird's eye view as seen below. Since the destination points, are
the corners of the image, they aren't visible as such.

![alt text][image4]

####3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Initially I tried to utilize both color and sobel based thresholding to obtain a binary thresholded image. However sobel thresholding required a lot of fine-tuning to produce noise
free results. So instead, I just stuck with color thresholding. I started to experiment with some colorspaces initially, but since that was time consuming, I decided to implement an 
interactive slider using OpenCV's `createTrackbar`. The code for this is in the `last code cell` of the notebook, currently commented out [since it was for quick testing only]. 
This helped me focus on different colorspaces quickly and identify the threshold ranges as well. Based on this, and thanks to discussions with Justin Heaton [a fellow student in the ND], I ended up
focusing on the S Channel of HLS, the L channel of LUV, and the B channel of LAB. Eventually I dropped the S channel of HLS (another suggestion thanks to Justin) since it was not robust
to the shadows in certain parts of the video.


![alt text][image3]



####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

