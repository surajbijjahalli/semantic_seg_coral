#%%
from datasets import Dataset, DatasetDict, Image,load_dataset
from huggingface_hub import login
import glob
from huggingface_hub import hf_hub_download
import json
import numpy as np
import matplotlib.pyplot as plt
from utils import color_palette,ImageSegmentationDataset
import albumentations as A
import numpy as np
from PIL import Image
from transformers import MaskFormerImageProcessor
from torch.utils.data import DataLoader
import torch
from transformers import MaskFormerForInstanceSegmentation
import evaluate
from tqdm.auto import tqdm


# Set login for hugghing face repo
login(token = "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")
repo_id = f"surajbijjahalli/semantic_seg_ATL"

# load entire dataset and split
dataset = load_dataset(repo_id)


# shuffle + split dataset
dataset = dataset.shuffle(seed=1)
dataset = dataset["train"].train_test_split(test_size=0.2)

# Pull out train and test datasets.
train_ds = dataset["train"] # Each sample is a dictionary with keys 'image' and 'label'
test_ds = dataset["test"]

# Download id2label file from the hub to map ids to labels
filename = "id2label.json"
id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
id2label = {int(k):v for k,v in id2label.items()}


#%%
# Sanity check a sample image from the training dataset to make sure images and masks are consistent
example_number = 10
example = train_ds[example_number]
image = example['image']



# load corresponding ground truth segmentation map, which includes a label per pixel
segmentation_map = np.array(example['label'])


# Grab unique labels in the segmentation map and map them to labels
labels = [id2label[label] for label in np.unique(segmentation_map)]

# set a colour palette for segmentation
palette = color_palette()
     

color_segmentation_map = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8) # height, width, 3
for label, color in enumerate(palette):
    color_segmentation_map[segmentation_map - 1 == label, :] = color

# Convert to BGR
ground_truth_color_seg = color_segmentation_map[..., ::-1]


fig,axs = plt.subplots()
axs.imshow(image)
axs.imshow(ground_truth_color_seg,alpha=0.3)


#%%

# Apply transforms
# This normalization is on 8-bit images (0-255 range). A scaled version of mean = (0.485,0.456,0.406) and std = (0.229, 0.224, 0.225)

ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255 
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

train_transform = A.Compose([
    #A.LongestMaxSize(max_size=1333),
    #A.RandomCrop(width=512, height=512),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD),
])

test_transform = A.Compose([
    #A.Resize(width=512, height=512),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD),

])

# Create pytorch datasets
train_dataset = ImageSegmentationDataset(train_ds, transform=train_transform)
test_dataset = ImageSegmentationDataset(test_ds, transform=test_transform)
     

image, segmentation_map, _, _ = train_dataset[example_number]
print(image.shape)
print(segmentation_map.shape)

#%%



unnormalized_image = (image * np.array(ADE_STD)[:, None, None]) + np.array(ADE_MEAN)[:, None, None]
unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
Image.fromarray(unnormalized_image)

labels = [id2label[label] for label in np.unique(segmentation_map)]
print(labels)

color_segmentation_map = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8) # height, width, 3
for label, color in enumerate(palette):
    color_segmentation_map[segmentation_map == label, :] = color
# Convert to BGR
ground_truth_color_seg = color_segmentation_map[..., ::-1]

img = np.moveaxis(image, 0, -1) * 0.5 + ground_truth_color_seg * 0.5
img = img.astype(np.uint8)

plt.figure(figsize=(15, 10))
plt.imshow(img)
plt.show()
     
#%%




# Create a preprocessor
# ignore_index is basically the label to be considered background. Optional parameter which may be better off not being provided.
preprocessor = MaskFormerImageProcessor(ignore_index=0,reduce_labels=False, do_resize=False, do_rescale=False, do_normalize=False) # previously ignore_index = 0 

# preprocessor also creates a set of binary masks - one mask for each class in the image - This is the format expected for the MaskFormer model



def collate_fn(batch):
    inputs = list(zip(*batch))
    images = inputs[0]
    segmentation_maps = inputs[1]
    # this function pads the inputs to the same size,
    # and creates a pixel mask
    # actually padding isn't required here since we are cropping
    batch = preprocessor(
        images,
        segmentation_maps=segmentation_maps,
        return_tensors="pt",
    )

    batch["original_images"] = inputs[2]
    batch["original_segmentation_maps"] = inputs[3]
    
    return batch


# Create pytorch train and test dataloaders. Is a batch size of 2 optimal ? 
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
     


# Each batch is a dictionary with 6 keys : ['pixel_values', 'pixel_mask', 'mask_labels', 'class_labels', 'original_images', 'original_segmentation_maps']
batch = next(iter(train_dataloader))
for k,v in batch.items():
  if isinstance(v, torch.Tensor):
    print(k,v.shape)
  else:
    print(k,v[0].shape)
#%%

# Verify that the batch and its contents are as expected
pixel_values = batch["pixel_values"][0].numpy()
pixel_values.shape

unnormalized_image = (pixel_values * np.array(ADE_STD)[:, None, None]) + np.array(ADE_MEAN)[:, None, None]
unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
Image.fromarray(unnormalized_image)

plt.figure()
plt.imshow(unnormalized_image)

# verify class labels
labels = [id2label[label] for label in batch["class_labels"][0].tolist()]
print(labels)
# verify mask labels
print('mask labels shape: ',batch["mask_labels"][0].shape)



def visualize_mask(labels, label_name):
  print("Label:", label_name)
  idx = labels.index(label_name)

  visual_mask = (batch["mask_labels"][0][idx].bool().numpy() * 255).astype(np.uint8)
  return Image.fromarray(visual_mask)
     

#visual_mask = visualize_mask(labels, "CCA")
     
#%%
#Define model



# Replace the head of the pre-trained model
model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-ade",
                                                          id2label=id2label,
                                                          ignore_mismatched_sizes=True)


#%%

# Sanity check the output of the untrained model
outputs = model(batch["pixel_values"],
                class_labels=batch["class_labels"],
                mask_labels=batch["mask_labels"])
     

outputs.loss
     
#%%



metric = evaluate.load("mean_iou")


#%%



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

running_loss = 0.0
num_samples = 0
num_epochs = 1
for epoch in range(num_epochs):
  print("Epoch:", epoch)
  model.train()
  for idx, batch in enumerate(tqdm(train_dataloader)):
      # Reset the parameter gradients
      optimizer.zero_grad()

      # Forward pass
      outputs = model(
          pixel_values=batch["pixel_values"].to(device),
          mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
          class_labels=[labels.to(device) for labels in batch["class_labels"]],
      )

      # Backward propagation
      loss = outputs.loss
      loss.backward()

      batch_size = batch["pixel_values"].size(0)
      running_loss += loss.item()
      num_samples += batch_size

      if idx % 100 == 0:
        print("Loss:", running_loss/num_samples)

      # Optimization
      optimizer.step()

  model.eval()
  for idx, batch in enumerate(tqdm(test_dataloader)):
    if idx > 5:
      break

    pixel_values = batch["pixel_values"]
    
    # Forward pass
    with torch.no_grad():
      outputs = model(pixel_values=pixel_values.to(device))

    # get original images
    original_images = batch["original_images"]
    target_sizes = [(image.shape[0], image.shape[1]) for image in original_images]
    # predict segmentation maps
    predicted_segmentation_maps = preprocessor.post_process_semantic_segmentation(outputs,
                                                                                  target_sizes=target_sizes)

    # get ground truth segmentation maps
    ground_truth_segmentation_maps = batch["original_segmentation_maps"]

    metric.add_batch(references=ground_truth_segmentation_maps, predictions=predicted_segmentation_maps)
  
  # NOTE this metric outputs a dict that also includes the mIoU per category as keys
  # so if you're interested, feel free to print them as well
  #print("Mean IoU:", metric.compute(num_labels = len(id2label))['mean_iou']) # removed ignore_index = 0 --> 
  # Store metrics on the test dataset
  eval_test_metric = metric.compute(num_labels = len(id2label), ignore_index = 0)
  print("Mean IoU:", eval_test_metric['mean_iou']) # removed ignore_index = 0


#%%

 # let's take the first test batch
batch = next(iter(test_dataloader))


for k,v in batch.items():
  if isinstance(v, torch.Tensor):
    print(k,v.shape)
  else:
    print(k,len(v))

# forward pass
with torch.no_grad():
  outputs = model(batch["pixel_values"].to(device))


original_images = batch["original_images"]
target_sizes = [(image.shape[0], image.shape[1]) for image in original_images]
# predict segmentation maps
predicted_segmentation_maps = preprocessor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)


image = batch["original_images"][0]
Image.fromarray(image)
#%%



segmentation_map = predicted_segmentation_maps[0].cpu().numpy()

color_segmentation_map = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8) # height, width, 3
for label, color in enumerate(palette):
    color_segmentation_map[segmentation_map == label, :] = color
# Convert to BGR
ground_truth_color_seg = color_segmentation_map[..., ::-1]

img = image * 0.5 + ground_truth_color_seg * 0.5
img = img.astype(np.uint8)

plt.figure(figsize=(15, 10))
plt.imshow(img)
plt.show()

#%% 
# Compare to ground truth

segmentation_map = batch["original_segmentation_maps"][0]

color_segmentation_map = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8) # height, width, 3
for label, color in enumerate(palette):
    color_segmentation_map[segmentation_map == label, :] = color
# Convert to BGR
ground_truth_color_seg = color_segmentation_map[..., ::-1]

img = image * 0.5 + ground_truth_color_seg * 0.5
img = img.astype(np.uint8)

plt.figure(figsize=(15, 10))
plt.imshow(img)
plt.show()

#%%

# Evaluate on test data
'''
for idx, batch in enumerate(tqdm(test_dataloader)):
  

    pixel_values = batch["pixel_values"]
    
    # Forward pass
    with torch.no_grad():
      outputs = model(pixel_values=pixel_values.to(device))

    # get original images
    original_images = batch["original_images"]
    target_sizes = [(image.shape[0], image.shape[1]) for image in original_images]
    # predict segmentation maps
    predicted_segmentation_maps = preprocessor.post_process_semantic_segmentation(outputs,
                                                                                  target_sizes=target_sizes)

    # get ground truth segmentation maps
    ground_truth_segmentation_maps = batch["original_segmentation_maps"]

    metric.add_batch(references=ground_truth_segmentation_maps, predictions=predicted_segmentation_maps)
    #print("Evals: ",metric.compute(num_labels = len(id2label), ignore_index = 0))
  # NOTE this metric outputs a dict that also includes the mIoU per category as keys
  # so if you're interested, feel free to print them as well
  #print("Mean IoU:", metric.compute(num_labels = len(id2label))['mean_iou']) # removed ignore_index = 0 --> 
eval_test_metric = metric.compute(num_labels = len(id2label), ignore_index = 0)
print("Eval outputs: ",eval_test_metric)
#print("Mean IoU:", metric.compute(num_labels = len(id2label), ignore_index = 0)['mean_iou']) # removed ignore_index = 0

'''