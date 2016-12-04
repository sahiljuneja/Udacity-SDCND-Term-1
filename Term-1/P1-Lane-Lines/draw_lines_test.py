def draw_lines2(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    if lines is None:
        return
    
    ## Find average coordinates (centre of mass)
    avg_line_tmp = np.array([np.average(line,0) for line in lines])
	avg_line = np.average(avg_line_tmp,0)
  
    avg_left_x = min(avg_line[0],avg_line[2])
    avg_right_x = max(avg_line[0],avg_line[2])
    
    left_lane = []
    right_lane = []
    
	## Split up line coordinates into left and right lanes
		#TODO: More pythonic implementation.
    for line in lines:
        
        for x1,y1,x2,y2 in line:
            if (x1 < avg_left_x) :
                left_lane.append([x1,y1])
            if (x2 > avg_right_x):
                right_lane.append([x2,y2])
            if (x1 > avg_right_x) :
                right_lane.append([x1,y1])
            if (x2 < avg_left_x):
                left_lane.append([x2,y2])
            

    ## Individual x and y coordinates
    left_lane_x = [item[0] for item in left_lane]    
    left_lane_y = [item[1] for item in left_lane]
    right_lane_x = [item[0] for item in right_lane]
    right_lane_y = [item[1] for item in right_lane]
    
	## Fit line to coordinates
    left_fit = np.polyfit(left_lane_x, left_lane_y ,1)
    right_fit = np.polyfit(right_lane_x, right_lane_y ,1)
    
    left_fit_fn = np.poly1d(left_fit)
    right_fit_fn = np.poly1d(right_fit)
    
    
    ## Identify new line coordinates using above line function
    x_left_new = np.linspace(min(left_lane_x), max(left_lane_x), 50)
    y_left_new = left_fit_fn(x_left_new) 
    x_right_new = np.linspace(min(right_lane_x), max(right_lane_x), 50)
    y_right_new = right_fit_fn(x_right_new)
    
	## Convert to integer
    x_left_new.astype(int)
    y_left_new.astype(int)
    x_right_new.astype(int)
    x_right_new.astype(int)
    
    ## Identify slope, and find lowest x coordinate for each line, if max(y)
    left_slope = (min(y_left_new) - max(y_left_new))/(max(x_left_new) - min(x_left_new))
    left_x_lowest = ((left_slope*max(x_left_new)) + img.shape[0] - min(y_left_new))/left_slope
    
    right_slope = (min(y_right_new) - max(y_right_new))/(min(x_right_new) - max(x_right_new))
    right_x_lowest = ((right_slope*min(x_right_new)) + img.shape[0] - min(y_right_new))/right_slope
    
    ## Extrapolate and draw lines based on which line is longest 
		# This might be the possible issue for jumps in video, especially when only a small line is detected on the white broken lane.
			# Slope thresholding? 
			# Assign average slope? How? Keep track of previous frames?
			
    if (min(y_left_new) > min(y_right_new)):
        left_x_highest = ((left_slope*left_x_lowest) - img.shape[0] + min(y_right_new))/left_slope
        right_x_highest = min(x_right_new)
        left_x_highest.astype(int)
        
        cv2.line(img, (int(left_x_lowest), img.shape[0]), (int(left_x_highest), int(min(y_right_new))), color, thickness)
        cv2.line(img, (int(right_x_lowest), img.shape[0]), (int(right_x_highest), int(min(y_right_new))), color, thickness)
        
    else:
        right_x_highest = ((right_slope*right_x_lowest) - img.shape[0] + min(y_left_new))/right_slope
        left_x_highest = max(x_left_new)
        right_x_highest.astype(int)
        
        cv2.line(img, (int(left_x_lowest), img.shape[0]), (int(left_x_highest), int(min(y_left_new))), color, thickness)
        cv2.line(img, (int(right_x_lowest), img.shape[0]), (int(right_x_highest), int(min(y_left_new))), color, thickness)
    

    
        
