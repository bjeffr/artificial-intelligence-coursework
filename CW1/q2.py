"""

QUESTION 2

"""

import os
import torch
import matplotlib.pyplot as plt

from PIL import Image
from torchvision.transforms import transforms
from torch import nn
from sklearn.preprocessing import MinMaxScaler


# Device configuration - defaults to CPU unless GPU is available on device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Creates a folder to store the filter and feature map images
image_folder = 'sombrero'
if not os.path.exists(image_folder):
    os.makedirs(image_folder)


#########################################################################
#
#        QUESTION 2.1.2 code here
#
#########################################################################

# Read in image
image_path = "test_images/sombrero.jpg"

# Open and show image
image = Image.open(image_path)
try:
    image = image.convert('RGB')  # To deal with some grayscale images in the data
except:
    pass

plt.imshow(image)
plt.show()


# Normalisations expected by pre-trained net, to apply in the image transform
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

# Resize normalise and convert image to tensor form
image_tensor = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])(image).to(device)


# Loads the model and downloads pre-trained weights if not already downloaded
model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True).to(device)

# To see the AlexNet architecture
model.eval()


# Pass image through a single forward pass of the network
with torch.no_grad():
    output = model(image_tensor.unsqueeze(0))

# Calculate probabilities from output vector
probabilities = torch.nn.functional.softmax(output[0], dim=0)


# Read in categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())


#########################################################################
#
#        QUESTION 2.1.3
#
#########################################################################

# Layer indices of each conv layer in AlexNet
conv_layer_indices = [0, 3, 6, 8, 10]


def extract_filter(conv_layer_idx, model):
    """ Extracts a single filter from the specified convolutional layer, zero-indexed where 0 indicates the first conv layer.
            Args:
                conv_layer_idx (int): index of convolutional layer
                model (nn.Module): PyTorch model to extract from
    """

    # Accessing first sequential block of model where all the conv layers are located
    seq_block = list(model.children())[0]

    # Accessing the conv layer with the given index
    conv_layer = list(seq_block.children())[conv_layer_idx]
    if type(conv_layer) != nn.Conv2d:
        raise ValueError("Specified index is not a convolutional layer.")

    # Extract filter
    return conv_layer.weight.cpu().detach().clone().numpy()


#########################################################################
#
#        QUESTION 2.1.4
#
#########################################################################


def extract_feature_maps(input, model):
    """ Extracts all the feature maps for all convolutional layers.
            Args:
                input (Tensor): input to model
                model (nn.Module): PyTorch model to extract from
    """

    feature_maps = []
    i = 0

    for conv_layer_idx in conv_layer_indices:
        # Forward pass of all layers not yet executed up to and including the conv layer
        while i <= conv_layer_idx:
            input = model.features[i].forward(input)
            i += 1

        # Forward pass of ReLU layer after conv layer
        input = model.features[i].forward(input)

        # Saving the outputs from the ReLU layer
        feature_maps.append(input.squeeze().cpu().detach().clone().numpy())
        i += 1

    return feature_maps


conv_layer_idx = 4
filters = extract_filter(conv_layer_indices[conv_layer_idx], model)
feature_maps = extract_feature_maps(image_tensor.unsqueeze(0), model)[conv_layer_idx]

print(f"{filters.shape} -> {feature_maps.shape}")


# Normalise filters
for i in range(filters.shape[0]):
    for j in range(filters.shape[1]):
        filters[i, j] = MinMaxScaler().fit_transform(filters[i, j])


# Explore feature maps
plt.figure(figsize=(50, 50))
for i in range(64):
    plt.subplot(8, 8, i + 1)
    plt.imshow(feature_maps[i])
    plt.axis('off')
plt.show()


feature_map_idx = 49

# Explore filters
plt.figure(figsize=(50, 50))
for i in range(64):
    plt.subplot(8, 8, i + 1)
    plt.imshow(filters[feature_map_idx, i])
    plt.axis('off')
plt.show()


# Save filter image
plt.imshow(filters[feature_map_idx, 0])
plt.axis('off')
plt.savefig(image_folder + '/filter_4.svg')


# Save feature map example
plt.imshow(feature_maps[feature_map_idx])
plt.axis('off')
plt.savefig(image_folder + '/feature_map_4.svg')
