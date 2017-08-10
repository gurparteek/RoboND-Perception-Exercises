#!/usr/bin/env python

# Import modules
from pcl_helper import *

# TODO: Define functions as required

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

    # Convert ROS msg (type PointCloud2) to PCL data (PointXYZRGB format) with helper function from pcl_helper.
    cloud = ros_to_pcl(pcl_msg)

    ##### Voxel Grid Downsampling #####

    """The point clouds from RGB-D cameras are too dense, hence computationally expensive. Downsampling 
    the point cloud data to reduce density but preserve important information is ideal.

    Using a Voxel Grid Filter where a grid of volumetric elements (voxels; as pixel is to picture element)
    is made and each voxel is averaged to a point cloud element; downsampled."""

    # Create a VoxelGrid filter object for our input point cloud
    vox = cloud.make_voxel_grid_filter()

    """Choose a voxel (also known as leaf) size (units in meters).
    Should start small and keep going large till loss of important information starts."""

    """A good way to choose leaf size is knowing the important information data forehand 
    such as smallest (or target) object size."""
    LEAF_SIZE = 0.01
    """A voxel (leaf) size of 0.01 results in a voxel of 1e-6 cubic meters that retains
    most of the important information, while significantly reducing the number of points in the cloud."""  

    # Set the voxel (or leaf) size. 
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)

    # Call the filter function to obtain the resultant downsampled point cloud.
    cloud_filtered = vox.filter()

    ##### PassThrough filter #####

    """More points in cloud = more coumputation; so if the target object location is known,
    the rest of the point cloud is not needed."""

    """A pass through filter is like a cropping tool. We specify an axis along which we know the limits
    within which the target objects lie, known as the region of interest. The pass through filter passes 
    through the cloud leaving only the region of interest."""

    # Create a PassThrough filter object.
    passthrough = cloud_filtered.make_passthrough_filter()

    # Assign axis and range to the passthrough filter object.
    # Applying the filter along z axis (the height with respect to the ground) to our tabletop scene.
    filter_axis = 'z'
    passthrough.set_filter_field_name (filter_axis)
    axis_min = 0.77
    axis_max = 1.1
    # The axis min and max sets the region of interest that the filter leaves out as a window as it passes.
    passthrough.set_filter_limits (axis_min, axis_max)

    # Finally use the filter function to obtain the resultant point cloud. 
    cloud_filtered = passthrough.filter()

    ##### RANSAC plane segmentation #####

    """RANSAC (Random Sample Consensus) is a two step (hypothesis and verification) iterative method
    which identifies data points belonging to a mathematical model (inliners) and those that dont (outliners)."""

    """First, the model is constructed using a min. no. of data pts. (eg. two for a line) and then the rest of
    pts. are verfied against its parameters (eg. slope and y-cutoff for a line) with certain error thresholds.
    The set of inliers obtained for that fitting model (random sample) is called a consensus set.
    The two steps are repeated until the obtained consensus set in certain iteration has enough inliers
    and that sample (mathematical model parameters) forms the solution as it had the most inliners in consensus."""

    # The points chosen are random so the solution is probalistic, increasing with the number of iterations.

    # Create the segmentation object
    seg = cloud_filtered.make_segmenter()

    # Set the model you wish to fit.
    # RANSAC plane fitting algorithm (calculate plane parameters and verfiy) already exists in the PCL library.
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)

    # Max distance for a point to be considered fitting the model.
    # This is the error threshold for the model fit and influences (increases) the consensus set.
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)

    # Call the segment function to obtain set of inliner indices and model coefficients
    inliers, coefficients = seg.segment()

    ##### Extract inliers and outliers #####

    # Extract inliers
    extracted_inliers = cloud_filtered.extract(inliers, negative=False)
    cloud_table = extracted_inliers

    # Extract outliers using the negative flag to True.
    extracted_outliers = cloud_filtered.extract(inliers, negative=True)
    cloud_objects = extracted_outliers

    ##### Euclidean Clustering #####

    """Euclidean Clustering is the DBSCAN algorithm as it uses the Euclidean Distance to identfy nearest neighbours, 
    if the distance b/w is < min. distance specified, then point is added to the cluster (inliners), else outliner.
    If the point has > (min. members of a cluster - 1) neigbours, it becomes a core member, else an edge member.
    Each point that can be in a cluster is identified and then the algorithm moves to the next random point."""

    """Using k-d trees for nearest neighbor search for PCL's Euclidian Clustering (DBSCAN)
    algorithm to decrease the computational burden.
    k-d trees segment the Euclidian Space into partitions by divinding each dimension sequentially (at each root)
    into two each time (forming a tree) using the median for each dimension, same as in the Quick Sort partion method.
    Each point is then located in a partition and the seach is focussed there instead of the whole space."""
    
    """Convert XYZRGB point cloud to XYZ with helper function from pcl_helper, because PCL's 
    Euclidean Clustering algorithm requires a point cloud with only spatial information."""
    white_cloud = XYZRGB_to_XYZ(cloud_objects)
    tree = white_cloud.make_kdtree()

    ##### Create Cluster-Mask Point Cloud to visualize each cluster separately. #####
    # Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    # Set tolerances for distance threshold (max. Euclidean Distance b/w points)
    # as well as minimum and maximum cluster size (in points).
    # Experiment and find values that work for segmenting objects.
    ec.set_ClusterTolerance(0.02)
    ec.set_MinClusterSize(10)
    ec.set_MaxClusterSize(2000)
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()

    # Assign a color corresponding to each segmented object in scene.
    cluster_color = get_color_list(len(cluster_indices))

    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                            white_cloud[indice][1],
                                            white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

    # Create new cloud containing all clusters, each with unique color.
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # Convert PCL data (PointXYZRGB format) to ROS msg (type PointCloud2) with helper function from pcl_helper.
    ros_cloud_objects = pcl_to_ros(cloud_objects)
    ros_cloud_table = pcl_to_ros(cloud_table)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)

    # Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cluster_cloud)


if __name__ == '__main__':

    # Initializing a new node called "clustering".
    rospy.init_node('clustering', anonymous=True)

    # Subscribing our node to the "sensor_stick/point_cloud" topic.
    # Anytime a message arrives, the message data (a point cloud) will be passed to the pcl_callback() function for processing.
    pcl_sub = rospy.Subscriber("/sensor_stick/point_cloud", pc2.PointCloud2, pcl_callback, queue_size=1)      

    # Creating two new publishers to publish the point cloud data for the table and the objects to topics
    # called pcl_table and pcl_objects, respectively.
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    # Creating a new publisher to publish the point cloud data for the cluster cloud to topic called pcl_cluster.
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)

    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
     rospy.spin()
