# semantic_seg_coral - Semantic segmentation of coral species

Fine tune segformer to segment images of coral species. Since dense annotation masks are typically not available, the approach here first generates approximate ground truth segmentation masks from sparse (point) annotations using Fast Multilevel Superpixel Segmentation (Fast-MSS).
The image below shows an example of the available sparse annotations and the output generated segmentation mask.
![annotations](/assets/coral_example_annotation.png)

## Create dense annotation from sparse annotation for a randomly sampled image
Run generate_ground_truth_random_image.py with two arguments: (1) path/to/image/directory/ and (2) path/to/csv_annotation.csv. This randomly samples an image from the directory, generates a segmentation mask from the sparse annotations and visualizes both.
## Create dense annotation masks for an entire directory of images
Run generate_ground_truth_directory.py with three arguments: (1)path/to/image/directory/ and (2)path/to/csv_annotation.csv and (3)path/to/store/generated/masks/ 
