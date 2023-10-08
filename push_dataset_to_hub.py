
from datasets import Dataset, DatasetDict, Image
from huggingface_hub import login
import glob


login(token = "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")
# your images can of course have a different extension
# semantic segmentation maps are typically stored in the png format
path_to_img_dir = '/media/surajb/suraj_drive/datasets-acfr/seaview/resized_images/ATL/'
path_to_annotations_dir = '/media/surajb/suraj_drive/datasets-acfr/seaview/ground_truth_masks/ATL/'

# List of paths for each file in the image directory
image_paths_train = glob.glob(path_to_img_dir+"*.png")
label_paths_train = glob.glob(path_to_annotations_dir+"*.png")
#%%
#image_paths_train = ["path/to/image_1.jpg/jpg", "path/to/image_2.jpg/jpg", ..., "path/to/image_n.jpg/jpg"]
#label_paths_train = ["path/to/annotation_1.png", "path/to/annotation_2.png", ..., "path/to/annotation_n.png"]

# same for validation
# image_paths_validation = [...]
# label_paths_validation = [...]

def create_dataset(image_paths, label_paths):
    dataset = Dataset.from_dict({"image": sorted(image_paths),
                                "label": sorted(label_paths)})
    dataset = dataset.cast_column("image", Image())
    dataset = dataset.cast_column("label", Image())

    return dataset

# step 1: create Dataset objects
train_dataset = create_dataset(image_paths_train, label_paths_train)
#validation_dataset = create_dataset(image_paths_validation, label_paths_validation)

# step 2: create DatasetDict
dataset = DatasetDict({
    "train": train_dataset,
    #"validation": validation_dataset,
  }
)

# step 3: push to hub (assumes you have ran the huggingface-cli login command in a terminal/notebook)
dataset.push_to_hub("surajbijjahalli/semantic_seg_ATL")

# optionally, you can push to a private repo on the hub
# dataset.push_to_hub("name of repo on the hub", private=True)