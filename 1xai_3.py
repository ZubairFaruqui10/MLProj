import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load the MNIST dataset
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# DataLoaders for batching
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


# Define the model
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.softmax = nn.LogSoftmax(dim=1)  # LogSoftmax for classification

    def forward(self, x):
        x = x.reshape(-1, 28 * 28)  # Flatten the image
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)  # Apply log softmax to output probabilities
        return x


# Initialize the model, loss function, and optimizer
model = NeuralNet()
criterion = nn.NLLLoss()  # Use NLLLoss since LogSoftmax is applied in the model
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 5
for epoch in range(epochs):
    running_loss = 0
    for images, labels in train_loader:
        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

# %% md
#
# %% md
# 2. Select Test Images and Use LIME to Explain Predictions
#
# %%
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
import torch.nn.functional as F
import numpy as np


# Function to predict using the PyTorch model
def predict(images):
    # Convert images to PyTorch tensor, and reshape to (batch_size, channels, height, width)
    images = torch.Tensor(images).permute(0, 3, 1,
                                          2)  # Convert (batch_size, height, width, 3) to (batch_size, 3, height, width)
    images = images / 255.0  # Normalize the images to [0, 1]

    # Pass the images through the model and get predictions (log-probabilities)
    with torch.no_grad():
        outputs = model(images)  # Outputs shape: (batch_size, num_classes)

    # Apply exponentiation to get probabilities from log-probabilities
    probs = torch.exp(outputs)

    print(f"Number of images: {images.shape[0]}")  # Should be 10 for LIME's input
    print(f"Number of predictions (probabilities): {probs.shape[0]}")  # Should be 10

    return probs.detach().cpu().numpy()  # Return the probabilities for each class


# Function to convert grayscale image (1 channel) to RGB (3 channels)
def grayscale_to_rgb(image):
    """
    Convert a grayscale image of shape (28, 28, 1) to RGB of shape (28, 28, 3).
    """
    if image.shape[-1] == 1:
        image = np.squeeze(image, axis=-1)  # Remove the last dimension if it’s (28, 28, 1)
    return np.stack((image,) * 3, axis=-1)  # Stack grayscale into 3 channels (RGB)


# Select test images
dataiter = iter(test_loader)
images, labels = next(dataiter)  # Use next() to get the batch of images and labels

# Take a few test images
test_images = images[:5]
test_labels = labels[:5]

# Create a LIME explainer
explainer = lime_image.LimeImageExplainer()

# Explain a single image
i = 0  # Index of the image to explain
image = test_images[i].numpy().transpose(1, 2, 0)  # Convert PyTorch tensor to numpy array and reshape to (28, 28, 1)

# Fix the shape by squeezing if necessary, then convert to RGB
image_rgb = grayscale_to_rgb(image)  # Convert the grayscale image to RGB (shape should be (28, 28, 3))

# Check the shape to ensure it's correct before passing to LIME
print("Image RGB shape:", image_rgb.shape)  # Should output (28, 28, 3)

# Generate the explanation using LIME
explanation = explainer.explain_instance(image_rgb, predict, top_labels=1, hide_color=0, num_samples=1000)

# Visualize the LIME explanation
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5,
                                            hide_rest=False)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.title(f"LIME Explanation for Test Image {i}")
plt.show()
