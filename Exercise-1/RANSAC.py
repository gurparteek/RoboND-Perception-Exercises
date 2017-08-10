# Import PCL module
import pcl

# Load Point Cloud file
cloud = pcl.load_XYZRGB('tabletop.pcd')

##### Voxel Grid filter #####

# The point clouds from RGB-D cameras are too dense, hence computationally expensive.
# Downsampling the point cloud data to reduce density but preserve important information is ideal.

# Using a Voxel Grid Filter where a grid of volumetric elements (voxels; as pixel is to picture element)
# is made and each voxel is averaged to a point cloud element; downsampled.

# Create a VoxelGrid filter object for our input point cloud
vox = cloud.make_voxel_grid_filter()

# Choose a voxel (also known as leaf) size (units in meters).
# Should start small and keep going large till loss of important information starts.

# A good way to choose leaf size is knowing the important information data forehand
# such as smallest (or target) object size.
LEAF_SIZE = 0.01
# A voxel (leaf) size of 0.01 results in a voxel of 1e-6 cubic meters that retains
# most of the important information, while significantly reducing the number of points in the cloud.   

# Set the voxel (or leaf) size  
vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)

# Call the filter function to obtain the resultant downsampled point cloud
cloud_filtered = vox.filter()
filename = 'voxel_downsampled.pcd'
pcl.save(cloud_filtered, filename)

##### PassThrough filter #####

# More points in cloud = more coumputation; so if the target object location is known,
# the rest of the point cloud is not needed.

# A pass through filter is like a cropping tool. We specify an axis along which we know the limits
# within which the target objects lie, known as the region of interest.
# The pass through filter passes through the cloud leaving only the region of interest.

# Create a PassThrough filter object.
passthrough = cloud_filtered.make_passthrough_filter()

# Assign axis and range to the passthrough filter object.
# Applying the filter along z axis (the height with respect to the ground) to our tabletop scene.
filter_axis = 'z'
passthrough.set_filter_field_name (filter_axis)
axis_min = 0.6
axis_max = 1.1
# The axis min and max sets the region of interest that the filter leaves out as a window as it passes.
passthrough.set_filter_limits (axis_min, axis_max)

# Finally use the filter function to obtain the resultant point cloud. 
cloud_filtered = passthrough.filter()
filename = 'pass_through_filtered.pcd'
pcl.save(cloud_filtered, filename)

##### RANSAC plane segmentation #####

# RANSAC (Random Sample Consensus) is a two step (hypothesis and verification) iterative method
# which identifies data points belonging to a mathematical model (inliners) and those that dont (outliners).

# First, the model is constructed using a min. no. of data pts. (eg. two for a line) and then the rest of
# pts. are verfied against its parameters (eg. slope and y-cutoff for a line) with certain error thresholds.
# The set of inliers obtained for that fitting model (random sample) is called a consensus set.
# The two steps are repeated until the obtained consensus set in certain iteration has enough inliers
# and that sample (mathematical model parameters) forms the solution as it had the most inliners in consensus.

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

# Extract inliers
extracted_inliers = cloud_filtered.extract(inliers, negative=False)
filename = 'extracted_inliers.pcd'
# Save pcd for table.
pcl.save(extracted_inliers, filename)

# Extract outliers using the negative flag to True.
extracted_outliers = cloud_filtered.extract(inliers, negative=True)
filename = 'extracted_outliers.pcd'
# Save pcd for tabletop objects.
pcl.save(extracted_outliers, filename)
