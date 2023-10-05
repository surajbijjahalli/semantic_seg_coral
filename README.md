# semantic_seg_coral
Semantic segmentation of coral species

WIP - Fine tune segformer to segment images of coral species. Since dense annotation masks are typically not available, the approach here first generates approximate ground truth segmentation masks from sparse (point) annotations using Fast Multilevel Superpixel Segmentation (Fast-MSS).
To use: run generate_ground_truth_random_image.py with two arguments: (1) path/to/image/directory/ and (2) path/to/csv_annotation. This randomly samples an image from the directory, generates a segmentation mask from the sparse annotations and visualizes both.
TO DO: - generate ground truth masks for training and validation datasets 
       - Fine-tune SegFormer model on data
