from datasets import Dataset, DatasetDict, load_from_disk
import os
from PIL import Image
import numpy as np

# Load your Arrow dataset
dataset_path = './data/imagenet-1k/default/1.0.0/07900defe1ccf3404ea7e5e876a64ca41192f6c07406044771544ef1505831e8/'
train_arrow_files = [f'{dataset_path}/imagenet-1k-train-000{i:02d}-of-00257.arrow' for i in range(257)]
val_arrow_file = [f'{dataset_path}/imagenet-1k-validation-000{i:02d}-of-00013.arrow' for i in range(13)]

# Create directories for train/val splits
output_dir = 'data/imagenet/'
train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'val')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Function to save images from Arrow dataset
def save_images_from_arrow(arrow_files, split_dir):
    for arrow_file in arrow_files:
        dataset = Dataset.from_file(arrow_file)
        for i, example in enumerate(dataset):
            image = Image.fromarray(np.array(example['image']))
            label = example['label']
            label_dir = os.path.join(split_dir, str(label))
            os.makedirs(label_dir, exist_ok=True)
            image.save(os.path.join(label_dir, f'{i}.jpeg'))

# Save train images
save_images_from_arrow(train_arrow_files, train_dir)

# Save validation images
save_images_from_arrow([val_arrow_file], val_dir)

print("Images have been saved in the correct folder structure.")
