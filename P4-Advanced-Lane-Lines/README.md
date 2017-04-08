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
[image5]: ./output_images/thresholded_image.png "Thresholded Image"
[image6]: ./output_images/histogram_plot.png "Histogram"
[image7]: ./output_images/sliding_window.png "Sliding Window"
[image8]: ./output_images/thresholded_image_lane_lines.png "Lane Lines"
[image9]: ./output_images/final_image.jpg "Final Image"
[video1]: ./project_output.mp4 "Video"

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
to the shadows in certain parts of the video. The `color_threshold` function in `Code Cell 4` of the notebook implements the above and outputs a thresholded binary image. This function
is called in the `thresholded_img()` function in `Code Cell 6`.

| Colorspace      | Channel   | Thresholds	 |
|:---------------:|:---------:| :-----------:|
| HLS (not used)  | S     	  | (180, 255)	 |
| LUV 		      | L		  | (225, 255)   |
| LAB   		  | B         | (150, 255)   |

Following is the binary image I obtained


![alt text][image5]



####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In `Code Cell 7`, the `line_coords()` function calculated the histogram of 10 equally sized sections of the thresholded image. For each section I identify the histogram peaks and the
corresponding coordinates, and filter out some of the noise. Here is the histogram of the thresholded image
shown above - 

![alt text][image6]

The indices obtained above were then utilized to implement a sliding window approach, where each window was utilized to identify which pixels are white, and the pixel coordinates were
then stored in variables corresponding to each lane. Following is an implementation of the sliding window to identify lane points - 

![alt text][image7]

In `Code Cell 9`, the `identify_lane()` function, the pixel coordinates obtained above are used to fit a 2nd order polynomial (using `numpy's polyfit` funtion) to obtain the left and right lane lines. These lane lines
are then extrapolated based on the image dimensions. Following are the lane lines drawn over the thresholded image.

![alt text][image8]

For the video, previous 20 frames were saved (using global variables) and averaged over, to replace the right lane line in any frame where there were very few lane points being detected. This helped
to smooth out any aberrations.

In `Code Cell 13`, the `draw_lane_line()` function drew the lane lines and filled the lane area using OpenCV's `polylines()` and `fillPoly()` functions on top of a blank image.
This image was then unwarped using OpenCV's `warpPerspective()` function. The output of this is shown in the 6th step below.

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature and the position of vehicle are implemented in the functions `lane_curvature()` defined in `Code Cell 11` and `distance_from_lane()` defined in `Code Cell 12`.

Since we are working with images, to obtain the radius of curvature, pixel length needs to be converted into meters. Which is done based on the following -

`ym_per_pix = 30/720 # meters per pixel in y dimension`

`xm_per_pix = 3.7/700 # meteres per pixel in x dimension`

A second order polynomial is then fit to the lane pixels converted to meters. The following equation then obtains the radius of curvature using the polynomial fit and the lowest y-coordinate
of the lane in the image (which is numerically the maximum). 

`rad_curvature = ((1 + (2*new_fit[0]*y_eval + new_fit[1])**2)**1.5)/np.absolute(2*new_fit[0])`

To calculate the distance of the car from the middle of the lane, the middle of the two lane lines is calculated (using the bottommost points of the lanes)
and the image center is subtracted from this. The result is multiplied by `xm_per_pix` defined above to obtain the offset in meters. Following is the equation for this -

`car_pos = ((left_lane[-1] + right_lane[-1])//2 - img_center[0]) * xm_per_pix`

The above values are then overlayed as text on every video frame in the `draw_lane_line()` function of `Code Cell 13` using OpenCV's `putText()` function.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Here is the output of the above pipeline on a specific frame of the video -

![alt text][image9]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_output.mp4) named `project_output.mp4` [also attached]. The entire pipeline is run through the `process_video()` function in `Code Cell 15`.

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The major issues faced in implementing this project were to find the correct parameters for thresholding. The pipeline is quite specific to the project video since currently I am
only averaging over previous frames for the right lane. Under heavy shadows this pipeling doesn't perform well.

There are several ways to make this more robust -
* A better Class structure for each Lane Line to help keep track of previous N frames.
* Better tuning for gradient based thresholding, exploring different colorspaces.
* Improved perspective transform by not hardcoding the source and destination points. One option is to use hough's transform to identify lanes in a test image and use their end points.




