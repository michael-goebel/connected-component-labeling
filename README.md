# Summary
Implementation of Hoshen-Kopelman algorithm for python/numpy. Accepts an n-dimensional array with any number of classes.

# What is connected component labeling (CCL)?
In the simplest context, CCL is used on binary images to partition the set of pixels equal to 1 into subsets of connected pixels with unique labels. In post-processing segmentation maps, this can be useful to enforce certain heuristics. A user may find the largest segment, or filter out segments which do not meet a certain size threshold.

This code considers n-dimensional arrays (ex. images, 3D scans, videos), and any number of classes. Two pixels are defined here as connected if they share the same value, and are adjacent. For an image adjacency means directly up, right, down, or left. For the more general n-dimensional space, adjacency implies that two pixels have an L1 distance of 1.

# Getting started
The best place to start is by running the test.sh script in a Python 3.6+ virtual environment. The script first builds the module. Then it runs test.py, which generates a random array, and tests the CCL function on this.

