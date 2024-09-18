import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 5

# Define the transformations (resize, normalize, convert to tensor)
transform = transforms.Compose([
    transforms.Resize((64, 64)),        # Resize images to 64x64 pixels
    transforms.ToTensor(),              # Convert the image to PyTorch tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image between -1 and 1
])

# Load the datasets
train_dataset = datasets.ImageFolder(root='/home/zubair/Downloads/CNN Data/Training Dataset', transform=transform)
test_dataset = datasets.ImageFolder(root='/home/zubair/Downloads/CNN Data/Test Dataset', transform=transform)

# DataLoader (to handle batch processing)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define the Deep Neural Network
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)  # 32 channels, 16x16 image size after pooling
        self.fc2 = nn.Linear(128, len(train_dataset.classes))  # Number of classes in output layer

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # Conv1 + ReLU + MaxPool
        x = self.pool(torch.relu(self.conv2(x)))  # Conv2 + ReLU + MaxPool
        x = x.view(-1, 32 * 16 * 16)              # Flatten the tensor
        x = torch.relu(self.fc1(x))               # Fully connected layer 1
        x = self.fc2(x)                           # Fully connected layer 2 (output layer)
        return x

# Initialize the model, loss function, and optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()  # Cross entropy loss for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for images, labels in train_loader:
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute the loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update the weights

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Evaluate the model on the test set
def test_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # No need to track gradients for testing
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')

# Train the model
train_model(model, train_loader, criterion, optimizer, num_epochs)

# Test the model
test_model(model, test_loader)



#################

import matplotlib.pyplot as plt
import numpy as np


# Function to display images
def imshow(img):
    img = img / 2 + 0.5  # Unnormalize the image (undo normalization)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Modified test function to collect correct and incorrect predictions
def test_model_with_plots(model, test_loader, num_images_to_display=5):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    correct_images = []
    incorrect_images = []

    correct_labels = []
    incorrect_labels = []

    predicted_labels_correct = []
    predicted_labels_incorrect = []

    with torch.no_grad():  # No need to track gradients for testing
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Collect correct and incorrect images
            for i in range(len(labels)):
                if predicted[i] == labels[i]:
                    if len(correct_images) < num_images_to_display:
                        correct_images.append(images[i])
                        correct_labels.append(labels[i].item())
                        predicted_labels_correct.append(predicted[i].item())
                else:
                    if len(incorrect_images) < num_images_to_display:
                        incorrect_images.append(images[i])
                        incorrect_labels.append(labels[i].item())
                        predicted_labels_incorrect.append(predicted[i].item())

    # Display results
    print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')

    # Plot correct predictions
    print("\nCorrect Predictions:")
    plt.figure(figsize=(10, 5))
    for i in range(len(correct_images)):
        plt.subplot(1, num_images_to_display, i + 1)
        imshow(correct_images[i])
        plt.title(f"Pred: {predicted_labels_correct[i]}, Actual: {correct_labels[i]}")
        plt.axis('off')

    # Plot incorrect predictions
    print("\nIncorrect Predictions:")
    plt.figure(figsize=(10, 5))
    for i in range(len(incorrect_images)):
        plt.subplot(1, num_images_to_display, i + 1)
        imshow(incorrect_images[i])
        plt.title(f"Pred: {predicted_labels_incorrect[i]}, Actual: {incorrect_labels[i]}")
        plt.axis('off')

    plt.show()


# Test the model and plot some correct and incorrect predictions
test_model_with_plots(model, test_loader, num_images_to_display=5)
