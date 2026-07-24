# Brain Tumor Detection from MRI with a Lightweight CNN

An independent PyTorch reimplementation of the model described by Naeem et al. in [*Lightweight CNN for Accurate Brain Tumor Detection from MRI with Limited Training Data*](https://doi.org/10.3389/fmed.2025.1636059).

## Abstract

This project independently reimplemented the lightweight convolutional neural network proposed by Naeem et al. for binary brain tumor detection in MRI images using limited training data. A dataset of 189 images was constructed from the Kaggle repository cited by the authors and divided into stratified training, validation, and test sets. The reported model architecture, data-augmentation procedures, training configuration, and evaluation methods were reproduced in PyTorch. The checkpoint with the lowest validation loss achieved a validation accuracy of 78.6% and ROC-AUC of 0.857, while its test accuracy was 85.7% with a ROC-AUC of 0.878. These results were substantially lower than the approximately 99% validation accuracy and ROC-AUC reported in the original study. Possible explanations include differences in dataset composition, stochastic variation, implementation details across deep-learning frameworks, and ambiguities or inconsistencies in the published methodology and results. This reimplementation highlights the challenges of reproducing high reported performance from small medical-imaging datasets when the precise data and experimental pipeline are not fully specified.

> This project is an educational reimplementation, not a clinical diagnostic system.

## Introduction

The reference study aimed to develop an effective computer-assisted system for the real-time identification of brain tumors in MRI images using limited data and computational resources. The CNN was trained on 189 MRI images, with a nearly equal balance of cancerous and non-cancerous cases. Data-augmentation techniques—including horizontal flipping, rotation, zoom, translation, noise injection, and brightness adjustment—were applied stochastically during each epoch using TensorFlow's `ImageDataGenerator` class. The training, validation, and test splits were stratified to maintain a balance between cancerous (tumor-positive) and non-cancerous (tumor-negative) cases. After 35 epochs of training, the network achieved a reported peak validation accuracy of approximately 98%.

The reported architecture comprised three convolutional layers, each followed by max pooling, as well as a fully connected hidden layer, dropout, and a fully connected output layer. The original implementation used the TensorFlow Layers API. For this project, I reproduced the data pipeline, architecture, training procedure, and evaluation in PyTorch.

## Results

The checkpoint with the lowest validation loss was obtained at epoch 14. It produced the following results:

| Split | BCE loss | Accuracy | Precision | Recall | F1 score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.4342 | 0.7857 | 0.7222 | 0.9286 | 0.8125 | 0.8571 |
| Test | 0.4424 | 0.8571 | 0.8125 | 0.9286 | 0.8667 | 0.8776 |

The remainder of this README documents the complete implementation that produced these results.

## Repository structure

```text
brain-tumor-cnn/
├── README.md
├── notebooks/
│   ├── data_preprocessing.ipynb
│   └── model_training_evaluation.ipynb
├── figures/
│   ├── model_architecture.png
│   ├── training_curves.png
│   ├── validation_roc_conf.png
│   └── test_roc_conf.png
├── models/
│   └── best_model.pt
├── requirements.txt
├── environment.yml
└── .gitignore
```

The MRI dataset is not included in the repository. Download and placement instructions are provided below.

## Implementation

### Data preprocessing

The MRI images used in this implementation were downloaded from the [Brain MRI Images for Brain Tumor Detection dataset on Kaggle](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection/data), which is the repository linked in the original article.

The public repository contains 155 tumor-positive images and 98 tumor-negative images. This differs from the 95 tumor-positive and 94 tumor-negative images described in the paper. To construct a dataset with the same size and class distribution as the one used in the original study, I sampled 95 tumor-positive images and 94 tumor-negative images using Python's `random` module.

The preprocessing notebook begins by importing the required libraries and locating the two class folders:

```python
from pathlib import Path
import random
import shutil

yes_source = Path("../data/raw/yes")
no_source = Path("../data/raw/no")

yes_images = [image for image in yes_source.iterdir()]
no_images = [image for image in no_source.iterdir()]

print("Tumor images:", len(yes_images))
print("No-tumor images:", len(no_images))
```

```text
Tumor images: 155
No-tumor images: 98
```

A seed of 100 was set before sampling so that the same subset could be selected again:

```python
random.seed(100)

selected_yes = random.sample(yes_images, 95)
selected_no = random.sample(no_images, 94)

print("Selected tumor images:", len(selected_yes))
print("Selected no-tumor images:", len(selected_no))
```

```text
Selected tumor images: 95
Selected no-tumor images: 94
```

The selected images were copied into intermediate class folders:

```python
yes_output = Path("../data/Yes")
no_output = Path("../data/No")

for image in selected_yes:
    shutil.copy(image, yes_output / image.name)

for image in selected_no:
    shutil.copy(image, no_output / image.name)

print("Finished copying images.")
```

One sampled image was inspected to confirm its dimensions and image mode:

```python
from PIL import Image
import matplotlib.pyplot as plt

image_path = selected_no[49]
image = Image.open(image_path)

print("Image size:", image.size)
print("Image mode:", image.mode)

plt.imshow(image, cmap="gray")
plt.axis("off")
plt.show()
```

The sampled images were then shuffled independently within each class and divided into stratified training, validation, and test sets:

```python
random.seed(100)
shuffle_yes = random.sample(selected_yes, k=len(selected_yes))
shuffle_no = random.sample(selected_no, k=len(selected_no))

train_yes = shuffle_yes[0:67]
val_yes = shuffle_yes[67:81]
test_yes = shuffle_yes[81:]

train_no = shuffle_no[0:66]
val_no = shuffle_no[66:80]
test_no = shuffle_no[80:]

print(len(train_yes), len(val_yes), len(test_yes))
print(len(train_no), len(val_no), len(test_no))
```

| Split | Tumor | No tumor | Total |
| --- | ---: | ---: | ---: |
| Training | 67 | 66 | 133 |
| Validation | 14 | 14 | 28 |
| Test | 14 | 14 | 28 |
| **Total** | **95** | **94** | **189** |

Finally, each split was copied into the directory structure expected by PyTorch's `ImageFolder`:

```python
for image in train_yes:
    shutil.copy(image, Path("../data/splits/train/yes") / image.name)

for image in val_yes:
    shutil.copy(image, Path("../data/splits/validation/yes") / image.name)

for image in test_yes:
    shutil.copy(image, Path("../data/splits/test/yes") / image.name)

for image in train_no:
    shutil.copy(image, Path("../data/splits/train/no") / image.name)

for image in val_no:
    shutil.copy(image, Path("../data/splits/validation/no") / image.name)

for image in test_no:
    shutil.copy(image, Path("../data/splits/test/no") / image.name)
```

This produces the following local data structure:

```text
data/
├── raw/
│   ├── yes/
│   └── no/
├── Yes/
├── No/
└── splits/
    ├── train/
    │   ├── no/
    │   └── yes/
    ├── validation/
    │   ├── no/
    │   └── yes/
    └── test/
        ├── no/
        └── yes/
```

### Image augmentation and data loading

Using PyTorch's `transforms` module, I reproduced all of the listed data-augmentation techniques except Gaussian noise, which required a custom transformation. The model notebook begins by importing the required libraries, locating the split folders, and setting the Python and PyTorch seeds:

```python
from pathlib import Path
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import random
from torch import nn
from torch import optim
import torchmetrics
import matplotlib.pyplot as plt

train_folder = Path("../data/splits/train")
validation_folder = Path("../data/splits/validation")
test_folder = Path("../data/splits/test")

random.seed(42)
torch.manual_seed(42)
generator = torch.Generator().manual_seed(42)
```

Gaussian noise was implemented as a custom callable transform:

```python
class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)
```

Training images were randomly flipped, rotated, translated, zoomed, and adjusted for brightness before noise was added. All images were converted to one-channel grayscale images and resized to 128 × 128 pixels:

```python
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05)
    ),
    transforms.ColorJitter(brightness=0.1),
    transforms.ToTensor(),
    AddGaussianNoise(mean=0.0, std=0.01),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128))
])

evaluation_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
```

The transformations were assigned to the corresponding `ImageFolder` datasets:

```python
train_dataset = ImageFolder(
    root=train_folder,
    transform=train_transform
)

validation_dataset = ImageFolder(
    root=validation_folder,
    transform=evaluation_transform
)

test_dataset = ImageFolder(
    root=test_folder,
    transform=evaluation_transform
)
```

The data loaders used a batch size of 18, matching the value reported in the paper. Training batches were shuffled, whereas validation and test batches were not:

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=18,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=18,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=18,
    shuffle=False
)
```

### Model creation

I reimplemented the CNN architecture using PyTorch's `nn.Module` class. The network contained three convolutional layers, each followed by a ReLU activation and max-pooling layer. The resulting feature maps were flattened and passed through a fully connected layer with a ReLU activation. A dropout layer randomly set half of the activations to zero before the output was passed through a final fully connected layer with a sigmoid activation function.

![Diagram of the BrainTumorCNN architecture](figures/model_architecture.png)

*Figure 1. Architecture of the reimplemented CNN. Three convolutional blocks extract image features before a fully connected layer, dropout layer, and sigmoid output produce the tumor-class probability.*

The complete model definition was:

```python
class BrainTumorCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(
            in_features=128 * 16 * 16,
            out_features=128
        )

        self.fc2 = nn.Linear(
            in_features=128,
            out_features=1
        )

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))

        return x.squeeze(dim=1)


model = BrainTumorCNN()
```

The shape of the data after each stage is summarized below:

| Stage | Operation | Output shape |
| --- | --- | --- |
| Input | 128 × 128 grayscale MRI | 1 × 128 × 128 |
| Block 1 | 3 × 3 convolution, 32 channels; ReLU; 2 × 2 max pooling | 32 × 64 × 64 |
| Block 2 | 3 × 3 convolution, 64 channels; ReLU; 2 × 2 max pooling | 64 × 32 × 32 |
| Block 3 | 3 × 3 convolution, 128 channels; ReLU; 2 × 2 max pooling | 128 × 16 × 16 |
| Flatten | Flatten feature maps | 32,768 |
| Hidden layer | 128-unit fully connected layer; ReLU | 128 |
| Regularization | Dropout, `p=0.5` | 128 |
| Output | One-unit fully connected layer; sigmoid | 1 |

### Model training

The network was trained using binary cross-entropy loss (`BCELoss`) and the Adam optimizer with the default learning rate and other optimizer parameters reported in the paper:

```python
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

The required classification metrics and Python's `copy` module were then imported:

```python
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryAUROC,
    BinaryROC,
)
import copy
```

Training and validation loss and accuracy were recorded at each of the 35 epochs. Validation loss, accuracy, precision, recall, F1 score, and ROC-AUC were also calculated at every epoch using TorchMetrics. The weights from the epoch with the lowest validation loss were copied and retained along with the corresponding epoch number.

```python
num_epochs = 35
best_validation_loss = float("inf")

train_loss_list = []
validation_loss_list = []
train_accuracy_list = []
validation_accuracy_list = []

best_model_weights = None
best_epoch = None

for epoch in range(num_epochs):
    # Training
    model.train()
    running_train_loss = 0.0
    train_accuracy_metric = BinaryAccuracy()

    for features, target in train_loader:
        target = target.float()

        optimizer.zero_grad()

        probabilities = model(features)
        loss = criterion(probabilities, target)

        loss.backward()
        optimizer.step()

        train_accuracy_metric.update(probabilities, target)
        running_train_loss += loss.item()

    train_loss = running_train_loss / len(train_loader)
    train_loss_list.append(train_loss)
    train_accuracy_list.append(
        train_accuracy_metric.compute().item()
    )

    # Validation
    model.eval()
    running_validation_loss = 0.0

    accuracy_metric = BinaryAccuracy()
    precision_metric = BinaryPrecision()
    recall_metric = BinaryRecall()
    f1_metric = BinaryF1Score()
    roc_metric = BinaryROC()
    auc_metric = BinaryAUROC()

    with torch.no_grad():
        for features, target in validation_loader:
            target_float = target.float()

            probabilities = model(features)
            loss = criterion(probabilities, target_float)

            running_validation_loss += loss.item()

            # TorchMetrics accepts probabilities and applies
            # the 0.5 threshold internally where appropriate.
            accuracy_metric.update(probabilities, target)
            precision_metric.update(probabilities, target)
            recall_metric.update(probabilities, target)
            f1_metric.update(probabilities, target)
            roc_metric.update(probabilities, target)

            # AUC must receive probabilities, not 0/1 predictions.
            auc_metric.update(probabilities, target)

    validation_loss = (
        running_validation_loss / len(validation_loader)
    )
    validation_loss_list.append(validation_loss)

    validation_accuracy = accuracy_metric.compute().item()
    validation_accuracy_list.append(validation_accuracy)
    validation_precision = precision_metric.compute().item()
    validation_recall = recall_metric.compute().item()
    validation_f1 = f1_metric.compute().item()
    validation_auc = auc_metric.compute().item()

    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        best_model_weights = copy.deepcopy(model.state_dict())
        best_epoch = epoch + 1
        print(f"Best model updated at epoch {epoch + 1}")

    print(
        f"Epoch {epoch + 1:02d}/{num_epochs} | "
        f"Train loss: {train_loss:.4f} | "
        f"Validation loss: {validation_loss:.4f} | "
        f"Accuracy: {validation_accuracy:.4f} | "
        f"Precision: {validation_precision:.4f} | "
        f"Recall: {validation_recall:.4f} | "
        f"F1: {validation_f1:.4f} | "
        f"ROC-AUC: {validation_auc:.4f}"
    )
```

The training and validation curves were generated with the following code:

```python
epochs = range(1, len(train_loss_list) + 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss
axes[0].plot(
    epochs,
    train_loss_list,
    label="Training loss"
)
axes[0].plot(
    epochs,
    validation_loss_list,
    label="Validation loss"
)
axes[0].set_title("Loss by Epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Binary cross-entropy loss")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Accuracy
axes[1].plot(
    epochs,
    train_accuracy_list,
    label="Training accuracy"
)
axes[1].plot(
    epochs,
    validation_accuracy_list,
    label="Validation accuracy"
)
axes[1].set_title("Accuracy by Epoch")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_ylim(0, 1)
axes[1].legend()
axes[1].grid(alpha=0.3)

axes[0].axvline(
    x=best_epoch,
    color="red",
    linestyle="--",
    linewidth=0.9,
    label=f"Best epoch ({best_epoch})"
)

axes[1].axvline(
    x=best_epoch,
    color="red",
    linestyle="--",
    linewidth=0.9,
    label=f"Best epoch ({best_epoch})"
)

plt.tight_layout()
plt.show()
```

![Training and validation loss and accuracy across 35 epochs](figures/training_curves.png)

*Figure 2. Training and validation performance across 35 epochs. The left panel shows binary cross-entropy loss, while the right panel shows classification accuracy. The dashed red line marks epoch 14, at which the lowest validation loss was recorded and the model weights were saved.*

### Model evaluation

The selected checkpoint was restored before final evaluation. Classification metrics were calculated on the validation and test sets, and ROC curves and confusion matrices were generated for both.

The validation metrics were calculated as follows:

```python
from torchmetrics.classification import BinaryConfusionMatrix

model.load_state_dict(best_model_weights)
model.eval()

best_roc_metric = BinaryROC()
best_auc_metric = BinaryAUROC()
best_accuracy_metric = BinaryAccuracy()
best_precision_metric = BinaryPrecision()
best_recall_metric = BinaryRecall()
best_f1_metric = BinaryF1Score()
best_confusion_matrix_metric = BinaryConfusionMatrix(
    threshold=0.5
)
running_best_loss = 0

with torch.no_grad():
    for features, target in validation_loader:
        probabilities = model(features)
        running_best_loss += criterion(
            probabilities,
            target.float()
        )

        best_roc_metric.update(probabilities, target)
        best_auc_metric.update(probabilities, target)
        best_confusion_matrix_metric.update(
            probabilities,
            target
        )
        best_accuracy_metric.update(probabilities, target)
        best_precision_metric.update(probabilities, target)
        best_recall_metric.update(probabilities, target)
        best_f1_metric.update(probabilities, target)

best_loss = running_best_loss / len(validation_loader)
best_fpr, best_tpr, best_thresholds = (
    best_roc_metric.compute()
)
best_accuracy = best_accuracy_metric.compute().item()
best_precision = best_precision_metric.compute().item()
best_recall = best_recall_metric.compute().item()
best_f1 = best_f1_metric.compute().item()
best_auc = best_auc_metric.compute().item()
best_confusion_matrix = (
    best_confusion_matrix_metric.compute()
)

print(
    f"Best Epoch ({best_epoch}): "
    f"BCELoss: {best_loss:.4f} | "
    f"Accuracy: {best_accuracy:.4f} | "
    f"Precision: {best_precision:.4f} | "
    f"Recall: {best_recall:.4f} | "
    f"F1: {best_f1:.4f} | "
    f"ROC-AUC: {best_auc:.4f}"
)
```

```text
Best Epoch (14): BCELoss: 0.4342 | Accuracy: 0.7857 |
Precision: 0.7222 | Recall: 0.9286 | F1: 0.8125 |
ROC-AUC: 0.8571
```

The validation ROC curve and confusion matrix were plotted together:

```python
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ROC curve
axes[0].plot(
    best_fpr.cpu(),
    best_tpr.cpu(),
    linewidth=2,
    color="orange",
    label=f"Best model (AUC = {best_auc:.3f})"
)

axes[0].plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="blue",
    label="Random classifier"
)

axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title(
    f"Validation ROC Curve — Best Epoch ({best_epoch})"
)
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 1)
axes[0].grid(alpha=0.3)
axes[0].legend()

# Confusion matrix
sns.heatmap(
    best_confusion_matrix.cpu().numpy(),
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No tumor", "Tumor"],
    yticklabels=["No tumor", "Tumor"],
    ax=axes[1]
)

axes[1].set_xlabel("Predicted label")
axes[1].set_ylabel("True label")
axes[1].set_title(
    f"Validation Confusion Matrix — Best Epoch ({best_epoch})"
)

plt.tight_layout()
plt.show()
```

![Validation ROC curve and confusion matrix](figures/validation_roc_conf.png)

*Figure 3. ROC curve and confusion matrix for the epoch-14 checkpoint on the validation set. The dashed diagonal represents random classification.*

The validation confusion matrix contained 9 true negatives, 5 false positives, 1 false negative, and 13 true positives.

The same metrics were then calculated on the held-out test set:

```python
test_roc_metric = BinaryROC()
test_auc_metric = BinaryAUROC()
test_accuracy_metric = BinaryAccuracy()
test_precision_metric = BinaryPrecision()
test_recall_metric = BinaryRecall()
test_f1_metric = BinaryF1Score()
test_confusion_matrix_metric = BinaryConfusionMatrix(
    threshold=0.5
)
running_test_loss = 0

with torch.no_grad():
    for features, target in test_loader:
        probabilities = model(features)
        running_test_loss += criterion(
            probabilities,
            target.float()
        )

        test_roc_metric.update(probabilities, target)
        test_auc_metric.update(probabilities, target)
        test_confusion_matrix_metric.update(
            probabilities,
            target
        )
        test_accuracy_metric.update(probabilities, target)
        test_precision_metric.update(probabilities, target)
        test_recall_metric.update(probabilities, target)
        test_f1_metric.update(probabilities, target)

test_loss = running_test_loss / len(test_loader)
test_fpr, test_tpr, test_thresholds = (
    test_roc_metric.compute()
)
test_accuracy = test_accuracy_metric.compute().item()
test_precision = test_precision_metric.compute().item()
test_recall = test_recall_metric.compute().item()
test_f1 = test_f1_metric.compute().item()
test_auc = test_auc_metric.compute().item()
test_confusion_matrix = (
    test_confusion_matrix_metric.compute()
)

print(
    f"Test: BCELoss: {test_loss:.4f} | "
    f"Accuracy: {test_accuracy:.4f} | "
    f"Precision: {test_precision:.4f} | "
    f"Recall: {test_recall:.4f} | "
    f"F1: {test_f1:.4f} | "
    f"ROC-AUC: {test_auc:.4f}"
)
```

```text
Test: BCELoss: 0.4424 | Accuracy: 0.8571 |
Precision: 0.8125 | Recall: 0.9286 | F1: 0.8667 |
ROC-AUC: 0.8776
```

The test ROC curve and confusion matrix were plotted with:

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ROC curve
axes[0].plot(
    test_fpr.cpu(),
    test_tpr.cpu(),
    linewidth=2,
    color="orange",
    label=f"Test (AUC = {test_auc:.3f})"
)

axes[0].plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="blue",
    label="Random classifier"
)

axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("Test ROC Curve")
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 1)
axes[0].grid(alpha=0.3)
axes[0].legend()

# Confusion matrix
sns.heatmap(
    test_confusion_matrix.cpu().numpy(),
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No tumor", "Tumor"],
    yticklabels=["No tumor", "Tumor"],
    ax=axes[1]
)

axes[1].set_xlabel("Predicted label")
axes[1].set_ylabel("True label")
axes[1].set_title("Test Confusion Matrix")

plt.tight_layout()
plt.show()
```

![Test ROC curve and confusion matrix](figures/test_roc_conf.png)

*Figure 4. ROC curve and confusion matrix for the epoch-14 checkpoint on the held-out test set. The dashed diagonal represents random classification.*

The test confusion matrix contained 11 true negatives, 3 false positives, 1 false negative, and 13 true positives. The reimplementation therefore retained high sensitivity to tumor-positive images, with a recall of 0.929 on both splits, but produced more false positives, particularly on the validation set.

Finally, the selected weights were saved as a PyTorch state dictionary:

```python
model_path = Path("../models/best_model.pt")
model_path.parent.mkdir(parents=True, exist_ok=True)

torch.save(best_model_weights, model_path)
```

## Discussion

The reimplemented network performed substantially worse than the model reported by Naeem et al. The authors reported a validation loss near zero at epoch 10, along with classification metrics of at least 98.75%.

Several factors could explain the discrepancy between my results and those reported in the paper: stochastic variation during data preprocessing and training, differences between the PyTorch and TensorFlow implementations, differences in dataset quality, errors in my implementation, or inaccurate reporting by the authors.

Stochastic variation alone is unlikely to explain such a substantial decrease in performance, particularly because I repeated the training process using multiple random seeds.

Random sampling and the creation of the training, validation, and test splits may also have affected performance, although I did not repeat the dataset-splitting procedure. If performance is highly dependent on the particular images assigned to each split, this would represent an important obstacle to the authors' goal of increasing accessibility by reducing computational and data requirements.

Differences between PyTorch and TensorFlow are similarly unlikely to account for the full discrepancy, assuming that the network was implemented correctly in both frameworks. Both are widely used for the tasks performed in this project, making framework choice alone an unlikely explanation for the full discrepancy.

I consider the remaining three explanations more plausible. The possibility of implementation errors could be investigated by continuing to improve my machine-learning skills and seeking feedback from more experienced practitioners.

Differences in dataset quality could be investigated using other datasets, such as those referenced by the authors in their other publications. However, evaluating dataset quality presents an additional challenge because identifying mislabeled MRI images requires specialized subject-matter expertise. Some images labeled as tumor-negative contain regions that appear abnormal and could plausibly represent tumors. Determining whether these images are mislabeled would require greater domain knowledge or collaboration with an expert.

Finally, several inconsistencies in the published article suggest that unclear or inaccurate reporting may have contributed to the difficulty of reproducing its results. In the Results section, the authors state that the loss decreased from 0.412 to nearly zero, whereas the loss curve in Figure 7 appears to begin above 0.6. They also compare their network with a more complex model trained on a larger dataset, for which they report a loss of 0.704 despite accuracy and ROC-AUC values of 98%. Given these apparent inconsistencies, incomplete or inaccurate reporting of the data-processing, model-development, training, or evaluation procedures may have contributed to the lower performance of my reimplementation.

## Reference

Naeem, A. B., Osman, O., Alsubai, S., Cevik, T., Zaidi, A., & Rasheed, J. (2025). Lightweight CNN for accurate brain tumor detection from MRI with limited training data. *Frontiers in Medicine, 12*, 1636059. https://doi.org/10.3389/fmed.2025.1636059