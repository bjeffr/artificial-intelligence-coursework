"""

QUESTION 1

"""

import torch
import itertools
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset
from PIL import Image
from scipy import signal
from torch import nn, optim
from torchvision.transforms import transforms
from torchinfo import summary
from mpl_toolkits.axes_grid1 import ImageGrid

#########################################################################
#
#        Config
#
#########################################################################

# Defines where the training and validation data is
ROOT_DIR = "imagenet10/train_set/"

# Defines the class labels
CLASS_LABELS = [
    "baboon",
    "banana",
    "canoe",
    "cat",
    "desk",
    "drill",
    "dumbbell",
    "football",
    "mug",
    "orange",
]

# Parameters used to normalise images
NORM_MEAN = [0.52283615, 0.47988218, 0.40605107]
NORM_STD = [0.29770654, 0.2888402, 0.31178293]


# Calculates validation loss and accuracy
def val_stats(data_loader, device, model, loss_fn):
    correct = 0
    total = 0
    running_loss = 0
    n = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)  # add in the number of labels in this minibatch
            correct += (predicted == labels).sum().item()  # add in the number of correct labels
            running_loss += loss.item()
            n += 1

    return running_loss / n, correct / total


# Calculates confusion matrix
def conf_matrix(data_loader, device, model):
    with torch.no_grad():
        targets = torch.tensor([], device=device)
        predictions = torch.tensor([], device=device)

        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            targets = torch.cat((targets, labels), dim=0)
            predictions = torch.cat((predictions, model(images)), dim=0)

    return confusion_matrix(targets.cpu(), predictions.argmax(dim=1).cpu())


# Function to plot confusion matrix with pyplot
def plot_confusion_matrix(cm, classes, normalise=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalise:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = title + ' [Normalised]'

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalise else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


#########################################################################
#
#        ImageNet10 class
#
#########################################################################

class ImageNet10(Dataset):

    def __init__(self, df, transform=None):
        """
        Args:
            df (DataFrame object): Dataframe containing the images, paths and classes
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load image from path and get label
        x = Image.open(self.df['path'][index])
        try:
            x = x.convert('RGB')  # To deal with some grayscale images in the data
        except:
            pass
        y = torch.tensor(int(self.df['class'][index]))

        if self.transform:
            x = self.transform(x)

        return x, y


#########################################################################
#
#        QUESTION 1.1
#
#########################################################################

# Use GPU if available
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Gathers the meta data for the images
paths, classes = [], []
for i, dir_ in enumerate(CLASS_LABELS):
    for entry in os.scandir(ROOT_DIR + dir_):
        if entry.is_file():
            paths.append(entry.path)
            classes.append(i)

data = {
    'path': paths,
    'class': classes
}

# Creates a dataframe with the path to each image and its associated class.
data_df = pd.DataFrame(data, columns=['path', 'class'])
data_df = data_df.sample(frac=1).reset_index(drop=True)  # Shuffles the data

# Calculates the required size for the training set given the desired splitting ratio.
train_size = int(len(data_df) * 0.8)

# Resize normalise and convert image to tensor form
data_transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])

# Splits the data into a training and validation data set.
dataset_train = ImageNet10(
    df=data_df[:train_size],
    transform=data_transform,
)

dataset_valid = ImageNet10(
    df=data_df[train_size:].reset_index(drop=True),
    transform=data_transform,
)

print("Images in training set:", len(dataset_train))
print("Images in validation set:", len(dataset_valid))

# Data loaders for use during training
train_loader = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = torch.utils.data.DataLoader(
    dataset_valid,
    batch_size=128,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

print("Batches in train loader:", len(train_loader))
print("Batches in validation loader:", len(val_loader))

# Extract first batch from training loader
images, labels = iter(train_loader).next()
images, labels = images.to(device), labels.to(device)

print(images.shape)
print(labels.shape)

# Define model
model = nn.Sequential(
    nn.Conv2d(3, 8, kernel_size=5, padding=0),  # 3*128*128 -> 8*124*124
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),  # 8*124*124 -> 8*62*62
    nn.Conv2d(8, 16, kernel_size=5, padding=0),  # 8*62*62 -> 16*58*58
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),  # 16*58*58 -> 16*29*29
    nn.Flatten(),
    nn.Linear(16 * 29 * 29, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
).to(device)

summary(model, input_size=(train_loader.batch_size, 3, 128, 128))

# Initialise empty lists to hold performance data, as well as the loss function and optimizer
training_loss, validation_loss, validation_accuracy = [], [], []
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# loop over the dataset multiple times
for epoch in range(15):
    # Zero the parameter gradients
    optimizer.zero_grad()

    # Forward, backward, and update parameters
    outputs = model(images)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()

    # Calculate and add performance metrics to associated lists
    training_loss.append(loss.item())
    val_loss, val_accuracy = val_stats(val_loader, device, model, loss_fn)
    validation_loss.append(val_loss)
    validation_accuracy.append(val_accuracy)
    print(f"{epoch}  train_loss: {loss.item():.3f}  val_loss: {val_loss:.3f}  val_acc: {val_accuracy:.1%}")

# Plot the training and validation loss over all epochs
plt.plot(training_loss, 'r', label='training loss')
plt.plot(validation_loss, 'g', label='validation loss')
plt.title('Training and validation loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper left')
plt.savefig('q1_1_2.svg')

#########################################################################
#
#        QUESTION 1.2 (Some code will be duplicated as separate files were originally used.
#
#########################################################################

# Creates a folder to store graphs and performance data for this model version
version_folder = 'q1_2_v8'
if not os.path.exists(version_folder):
    os.makedirs(version_folder)

# Use GPU if available
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Gathers the meta data for the images
paths, classes = [], []
for i, dir_ in enumerate(CLASS_LABELS):
    for entry in os.scandir(ROOT_DIR + dir_):
        if entry.is_file():
            paths.append(entry.path)
            classes.append(i)

data = {
    'path': paths,
    'class': classes
}

# Creates a dataframe with the path to each image and its associated class.
data_df = pd.DataFrame(data, columns=['path', 'class'])
data_df = data_df.sample(frac=1).reset_index(drop=True)  # Shuffles the data

# Calculates the required size for the training set given the desired splitting ratio.
train_size = int(len(data_df) * 0.8)

# Splits the data into a training and validation data set and applies transformations.
dataset_train = ImageNet10(
    df=data_df[:train_size],
    transform=transforms.Compose([
        transforms.RandomResizedCrop(size=128),
        transforms.RandomHorizontalFlip(),
        transforms.RandomPerspective(),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])
)

dataset_valid = ImageNet10(
    df=data_df[train_size:].reset_index(drop=True),
    transform=transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])
)

print("Images in training set:", len(dataset_train))
print("Images in validation set:", len(dataset_valid))

# Data loaders for use during training
train_loader = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = torch.utils.data.DataLoader(
    dataset_valid,
    batch_size=128,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# Specifies the neural network architecture
model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1)),
    nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    nn.ReLU(),
    nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1)),
    nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
    nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1)),
    nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    nn.ReLU(),
    nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1)),
    nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    nn.ReLU(),
    nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1)),
    nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
    nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1)),
    nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    nn.ReLU(),
    nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1)),
    nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
    nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1)),
    nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    nn.ReLU(),
    nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1)),
    nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
    nn.Flatten(start_dim=1, end_dim=-1),
    nn.Linear(in_features=4096, out_features=4096, bias=True),
    nn.BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    nn.ReLU(),
    nn.Linear(in_features=4096, out_features=4096, bias=True),
    nn.BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    nn.ReLU(),
    nn.Linear(in_features=4096, out_features=10, bias=True)
).to(device)

summary(model, input_size=(train_loader.batch_size, 3, 128, 128))

# Initialise empty lists to hold performance data, as well as the loss function and optimizer
training_loss, validation_loss, validation_accuracy, training_time = [], [], [], []
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# loop over the dataset until told to stop
continue_training = True

while continue_training:
    start = time.perf_counter()
    running_loss = 0
    n = 0

    # Loop through all batches of training data
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward, backward, and update parameters
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        # accumulate loss
        running_loss += loss.item()
        n += 1

    # Calculate and add performance metrics to associated lists
    train_time = time.perf_counter() - start
    training_time.append(train_time)
    train_loss = running_loss / n
    training_loss.append(train_loss)
    val_loss, val_accuracy = val_stats(val_loader, device, model, loss_fn)
    validation_loss.append(val_loss)
    validation_accuracy.append(val_accuracy)
    print(
        f"{len(training_loss)}  train_loss: {running_loss / n:.3f}  val_loss: {val_loss:.3f}  val_acc: {val_accuracy:.1%}  train_time: {train_time:.0f}")

    # Calculate smoothed version of validation loss and determine if it is time to stop training
    if len(validation_loss) >= 100:
        smoothed_val_loss = signal.savgol_filter(validation_loss, 55, 2)
        if smoothed_val_loss[-1] > smoothed_val_loss[-5]:
            continue_training = False

# Plot the training and validation loss, and validation accuracy over all epochs
fig, ax1 = plt.subplots()
plt.plot(training_loss, 'r', label='training loss')
plt.plot(validation_loss, 'y', label='validation loss')
plt.plot(signal.savgol_filter(validation_loss, 55, 2), 'g')
plt.legend(loc='upper left')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Training and validation loss, and validation accuracy')
ax2 = ax1.twinx()
ax2.plot(validation_accuracy, 'b', label='validation accuracy')
ax2.set_ylabel('accuracy')
plt.legend(loc='lower left')
plt.savefig(version_folder + '/loss_plot.svg')

# Calculate confusion matrix for the training set
cm_train = conf_matrix(train_loader, device, model)

# Save plot of confusion matrix to version folder
plt.figure(figsize=(10, 10))
plot_confusion_matrix(cm_train, CLASS_LABELS, normalise=True, title='Confusion matrix for training set')
plt.savefig(version_folder + '/cm_train.svg')

# Calculate confusion matrix for the validation set
cm_val = conf_matrix(val_loader, device, model)

# Save plot of confusion matrix to version folder
plt.figure(figsize=(10, 10))
plot_confusion_matrix(cm_val, CLASS_LABELS, normalise=True, title='Confusion matrix for validation set')
plt.savefig(version_folder + '/cm_val.svg')

# Save the model to version folder
torch.save(model, version_folder + '/model.pt')

# Save the performance metrics too the version folder
df = pd.DataFrame(
    data=[training_loss, validation_loss, validation_accuracy, training_time],
    index=['training_loss', 'validation_loss', 'validation_accuracy', 'training_time'],
    columns=range(1, len(training_loss) + 1)
).transpose()
df.to_csv(version_folder + '/perf_data.csv')

# Add version information to performance tracking sheet
df = pd.read_csv('q1_2_perf.csv')
df = df.append(pd.DataFrame([[
    train_loader.batch_size,
    train_loader.num_workers,
    dataset_train.transform.transforms,
    model,
    len(training_loss),
    optimizer.defaults.get('lr'),
    min(training_loss),
    min(validation_loss),
    max(validation_accuracy),
    sum(training_time) / 60
]], columns=df.columns), ignore_index=False)
df.to_csv('q1_2_perf.csv', index=False)

#########################################################################
#
#        QUESTION 1.3
#
#########################################################################

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Gathers the meta data for the images
paths = []
for entry in os.scandir("imagenet10/test_set/"):
    if entry.is_file():
        paths.append(entry.path)

# Load the model to use for the classification
model = torch.load('q1_2_v7/model.pt').to(device)

# Initialise lists to store the images as well the image names
images = []
names = []

# Loop over all images and load them
for path in paths:
    image = Image.open(path)

    try:
        image = image.convert('RGB')  # To deal with some grayscale images in the data
    except:
        pass

    # Apply necessary transformations
    image = transforms.Compose([
        transforms.Resize(192),
        transforms.CenterCrop(192),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])(image)

    # Save images and their names to the corresponding lists
    images.append(image)
    names.append(path[20:])


# Partitions the data into batches of size 32
def partition(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


batches = list(partition(images, 32))

# Perform forward pass of the network with all batches and store the predictions
with torch.no_grad():
    predictions = torch.tensor([], device=device)

    for images in batches:
        images = torch.tensor(np.stack(images, axis=0), device=device)
        predictions = torch.cat((predictions, model(images)), dim=0)

# Create a dataframe with the path, name and prediction of each image
df = pd.DataFrame([paths, names, predictions.argmax(dim=1).cpu().numpy()]).transpose()

# Loop over all classes and show the first 16 predicted images of each
for i in range(10):
    head = []
    for index, row in df.loc[df[2] == i].head(16).iterrows():
        head.append(Image.open(row[0]))

    fig = plt.figure(figsize=(8., 8.))
    plt.title(CLASS_LABELS[i].capitalize())
    plt.axis('off')
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(4, 4),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     share_all=True)

    grid[0].get_yaxis().set_ticks([])
    grid[0].get_xaxis().set_ticks([])

    for ax, im in zip(grid, head):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)

    plt.show()

# Save the names and predictions to a csv file
df.to_csv('test_preds.csv', header=False, index=False, columns=[1, 2])
