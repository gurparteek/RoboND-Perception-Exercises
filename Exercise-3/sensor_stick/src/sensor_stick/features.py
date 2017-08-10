import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from pcl_helper import *


def rgb_to_hsv(rgb_list):
    rgb_normalized = [1.0*rgb_list[0]/255, 1.0*rgb_list[1]/255, 1.0*rgb_list[2]/255]
    hsv_normalized = matplotlib.colors.rgb_to_hsv([[rgb_normalized]])[0][0]
    return hsv_normalized


def compute_color_histograms(cloud, using_hsv=True):

    # Compute histograms for the clusters
    point_colors_list = []

    # Step through each point in the point cloud
    for point in pc2.read_points(cloud, skip_nans=True):
    	"""Using float_to_rgb() function from pcl_helper.py to convert the 
    	RGB value packed as a float in point (X,Y,Z,RGB as float) which is point[3] 
    	to color (list): 3-element list of integers [0-255,0-255,0-255]."""
        rgb_list = float_to_rgb(point[3])
        if using_hsv:
            point_colors_list.append(rgb_to_hsv(rgb_list) * 255)
            """ * 255 as seen in the function rgb_to_hsv() above, we get normalized values,
            out of 1, so multiplying it to 255 to have rgb and hsv values the same range."""
        else:
            point_colors_list.append(rgb_list)

    # Populate lists with color values
    channel_1_vals = []
    channel_2_vals = []
    channel_3_vals = []

    for color in point_colors_list:
        channel_1_vals.append(color[0])
        channel_2_vals.append(color[1])
        channel_3_vals.append(color[2])
    
    ##### Compute histograms #####

    """The bins and the range are on the x-axis and are not a necessary arguments but
    are given to have consistency in all histograms as they will be compared accordingly.
    The chosen range is 0-256 and 32 bins so each bin has a range of 256/32 = 8.
    The first bins has a range 0-8 so if there a 'n' points in the bin, 'n' points
    in the cloud have a value [0-8].
    The range for HSV is still (0,256) despite being 360 deg cylinderical as we used
    normalized values to make the ranges for hsv and rgb same."""

    """NOTE: If the bins are too many, the model will start overfitting.
    Meaning, the data will get too precise and will not match the actual test pieces
    as the number of poses is small that we use to collect data."""
    
    ch_1_hist = np.histogram(channel_1_vals, bins=32, range=(0, 256))
    ch_2_hist = np.histogram(channel_2_vals, bins=32, range=(0, 256))
    ch_3_hist = np.histogram(channel_3_vals, bins=32, range=(0, 256))

    ##### Concatenate the histograms into a single feature vector #####

    """np.histogram() returns a tuple of two arrays.
    r_hist[0] contains the counts in each of the bins and r_hist[1] contains the bin edges.
    r_hist[0] is the feature we need, the value distribution of points in a cloud."""
    hist_features = np.concatenate((ch_1_hist[0], ch_2_hist[0], ch_3_hist[0])).astype(np.float64)

    ##### Normalize the result #####

    """Normalize to compare histograms of different total numbers of points.
    Divide by total number of points which is the sum of all the bins."""
    normed_features = hist_features / np.sum(hist_features) 
    return normed_features 


def compute_normal_histograms(normal_cloud):
    norm_x_vals = []
    norm_y_vals = []
    norm_z_vals = []

    for norm_component in pc2.read_points(normal_cloud,
                                          field_names = ('normal_x', 'normal_y', 'normal_z'),
                                          skip_nans=True):
        norm_x_vals.append(norm_component[0])
        norm_y_vals.append(norm_component[1])
        norm_z_vals.append(norm_component[2])

    ##### Compute histograms of normal values (just like with color) #####

    """Range is [-1,1] as these are the x,y,z components of normals which are unit vecotors
    so a componenet can have a max magnitude of 1."""

    norm_x_hist = np.histogram(norm_x_vals, bins=20, range=(-1, 1))
    norm_y_hist = np.histogram(norm_y_vals, bins=20, range=(-1, 1))
    norm_z_hist = np.histogram(norm_z_vals, bins=20, range=(-1, 1))

    ##### Concatenate the histograms into a single feature vector #####

    hist_features = np.concatenate((norm_x_hist[0], norm_y_hist[0], norm_z_hist[0])).astype(np.float64)

    ##### Normalize the result #####

    normed_features = hist_features / np.sum(hist_features)

    return normed_features
