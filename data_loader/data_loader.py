'''
    One famous saying is: If I have 8 hours to implement a model, I will spend 6 hours on handling and processing the data
    In this file, I am going to dive deep what should be done when preprocess the dataset, and how we apply pytorch dataloader
    to ease these processes.
'''

import torch
from torch import nn

device = torch.device('cpu')

'''
    Let's start by downloading a custom dataset from github
    In many scenarios, datasets will be obtained in this workflow
'''
import requests
import zipfile
from pathlib import Path

# setup the file path
data_path = Path("./data")
image_path = data_path / "pizza_steak_sushi"

# If the folder does not exist, create and download
if image_path.is_dir():
    print(f"{image_path} directory exists.")
else:
    print(f"Did not find {image_path} directory, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)

    # Download pizza, steak, sushi data
    with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
        request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
        print("Downloading pizza, steak, sushi data...")
        f.write(request.content)

    # Unzip pizza, steak, sushi data
    with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
        print("Unzipping pizza, steak, sushi data...")
        zip_ref.extractall(image_path)

import os

'''
    Inspect what are inside our folder
'''
# for dirpath, dirnames, filenames in os.walk(image_path):
#     print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


'''
    set up train and test path
'''
train_dir = image_path / "train"
test_dir = image_path / "test"

'''
    Visualize the data (image)    
'''
import random
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

random.seed(42)
# Get all image paths. The glob function inspect files under train_dir.
image_path_list = list(train_dir.glob('*/*.jpg'))
# Get random image path
random_image_path = random.choice(image_path_list)
# Open image
img = Image.open(random_image_path)
# get the class of the randomly selected image
image_class = random_image_path.parent.stem
# Plot the graph
img_as_array = np.asarray(img)

plt.figure(figsize=(10, 10))
plt.imshow(img_as_array)
plt.title(f'{image_class} | {img_as_array.shape}')
plt.axis(False)
#plt.show()

'''
    transform the data to be used in pytorch
'''

# Since it is an image data set, we can apply torchvision.transform packages to do the transformation
import torch
from torchvision import transforms

'''
    torchvision.transforms supports serial transformations, where we can add separate transformation together
    Each transformation returns a function that can be used to transform an image, with image as the input
'''
data_transform = transforms.Compose([
    # Resize the data
    transforms.Resize((64, 64)),
    # Flip the image horizontally, p denotes the possibility of the flipping on each data
    transforms.RandomHorizontalFlip(p=0.5),
    # Transform to pytorch tensor (H, W, C) -> (C, H, W)
    transforms.ToTensor(),
])

'''
    Visualize some of the transformed data
    
    Note the permute method: a good way to figure out the dimension change is
    give the original dim a number. e.x, (C, H, W), C = 0, H = 1, W = 2
    Then, based on the number, fill the new dim (H, W, C) -> (1, 2, 0)
'''


def plot_transformed_images(image_paths, transform, n=3, seed=42):
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            # Transform and plot image
            # Note: permute() will change shape of image to suit matplotlib
            # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
            transformed_image = transform(f).permute(1, 2, 0)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)


'''
    Now we have a basic understanding of how our data looks like, lets load it into pytorch
    One way to do it is: as we know it is an standard image dataset, we can use torchvision.Imagefolder to register 
    the dataset into pytorch
'''

from torchvision.datasets import ImageFolder
train_dataset = ImageFolder(root=train_dir, transform=data_transform)
test_dataset = ImageFolder(root=test_dir, transform=data_transform)

# get class names
class_names = train_dataset.classes
print(class_names)

# check the length
print(len(train_dataset), len(test_dataset))

# The dataset object can be accessed using indexing
img, label = train_dataset[0]
print(f"Image tensor:\n{img}")
print(f"Image shape: {img.shape}")
print(f"Image datatype: {img.dtype}")
print(f"Image label: {class_names[label]}")
print(f"Label datatype: {type(label)}")


'''
    Now it is time to turn pytorch dataset into dataloader
'''
from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=os.cpu_count())
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=os.cpu_count())


'''
    data augmentation:
    
    except ToTensor, other transformations are used to artificially improve the diversity of your dataset, which makes
    your model robust. And this is one most important feature pytorch dataset can provide, as it makes data augmentation 
    much easier. And this is one reason that people should spend enough time on processing the data. 
    
    A really useful method is transforms.TrivialAugmentWide, where it randomly selects a transformation and its corresponding
    magnitude of transforming from a predefined transformation set for each image, and apply the transform on it. 
    For more information, ask chatgpt
    
    Machine learning is all about harnessing the power of randomness and research shows that random transforms 
    (like transforms.RandAugment() and transforms.TrivialAugmentWide()) generally perform BETTER than hand-picked transforms.
'''

from torchvision import transforms

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31), # allows maximal transform magnitude
    transforms.ToTensor(),
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # no need to transform the test image
])

# Get all image paths
image_path_list = list(image_path.glob("*/*/*.jpg"))

# Plot random images
plot_transformed_images(
    image_paths=image_path_list,
    transform=train_transforms,
    n=3,
    seed=None,
)






