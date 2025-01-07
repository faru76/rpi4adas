import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
import tqdm  # Import tqdm for progress bar

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Custom dataset class
class ImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Load and preprocess data
def load_data(data_path):
    images = []
    labels = []
    for filename in os.listdir(data_path):
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(data_path, filename))
            img = cv2.resize(img, (64, 64))  # Resize image to 64x64 pixels
            images.append(img)
            label = filename.split('_')[-2]  # Extract label from filename
            labels.append(label)
    return np.array(images), np.array(labels)

# Manually define the label to index mapping
label_to_index = {
    "forward": 0,
    "left": 1,
    "right": 2,
}

# Convert labels to numeric values using the predefined mapping
def encode_labels(labels, label_to_index):
    return np.array([label_to_index[label] for label in labels])

data_path = "C:\\Users\\Fairuz\\Videos\\training_data"
images, labels = load_data(data_path)
labels = encode_labels(labels, label_to_index)

# Normalize images and split data
images = images.astype('float32') / 255.0
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Define data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Create datasets and dataloaders
train_dataset = ImageDataset(X_train, y_train, transform=transform)
test_dataset = ImageDataset(X_test, y_test, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the CNN model
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 12 * 12, 128),  # Adjust input size based on the flattened dimensions
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        #print(f"Feature map shape after conv layers: {x.shape}")  # Debugging
        x = self.fc_layers(x)
        return x

# Instantiate the model
num_classes = len(label_to_index)
model = CNNModel(num_classes).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, save_path="checkpoint.pt"):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.save_path = save_path

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(val_loss, model)
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f"Validation loss improved. Saving model...")
        torch.save(model.state_dict(), self.save_path)

# Training loop with progress bar
def train_model(model, train_loader, criterion, optimizer, num_epochs=50):
    model.train()
    train_loss_history, train_acc_history = [], []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        # Wrap the training loop with tqdm for progress bar
        with tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Track loss and accuracy
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                # Update progress bar with loss and accuracy
                pbar.set_postfix(loss=running_loss / (pbar.n + 1), accuracy=correct / total)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        # Early stopping check
        early_stopping(epoch_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered. Training halted.")
            break

    return train_loss_history, train_acc_history

# Train the model with early stopping and progress bar
early_stopping = EarlyStopping(patience=5, verbose=True, save_path="best_model.pth")
train_loss, train_acc = train_model(model, train_loader, criterion, optimizer)

# Save the model
torch.save(model.state_dict(), "final_model.pth")

# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        # Wrap the evaluation loop with tqdm for progress bar
        with tqdm.tqdm(test_loader, desc="Evaluating") as pbar:
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = outputs.max(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Update progress bar with evaluation metrics
                pbar.set_postfix()

    return np.array(all_preds), np.array(all_labels)

# Evaluate the model
y_pred, y_true = evaluate_model(model, test_loader)

# Confusion matrix and classification report
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_to_index.keys(), yticklabels=label_to_index.keys())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')

print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=label_to_index.keys()))

# Plot accuracy and loss
plt.figure()
plt.plot(train_acc, label='Train Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()
plt.savefig('accuracy_history.png')

plt.figure()
plt.plot(train_loss, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.savefig('loss_history.png') 
