import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import os
from model import swin_mnist as create_model


# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load dataset
mnist_test = datasets.MNIST(root='./data/', train=False, download=True)

# Set up DataLoader
batch_size = 64
num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

# Create model
model = create_model(num_classes=10).to(device)
model_weight_path = ''
# Load model weights
model.load_state_dict(torch.load(model_weight_path, map_location=device))
model.eval()

# Define loss function
criterion = torch.nn.CrossEntropyLoss()

# Initialize metrics
total_loss = 0.0
correct_predictions = 0
total_samples = 0

# Evaluate model
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        # Compute loss
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        # Compute predictions
        _, predicted_classes = torch.max(outputs, dim=1)

        # Compute accuracy
        correct_predictions += (predicted_classes == labels).sum().item()
        total_samples += labels.size(0)

# Calculate average loss and accuracy
average_loss = total_loss / len(test_loader)
accuracy = (correct_predictions / total_samples) * 100  # percentage

print(f'Average Loss: {average_loss:.4f} Accuracy: {accuracy:.6f}%')

