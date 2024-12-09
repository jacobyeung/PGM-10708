import torch.optim as optim
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import random
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from models.simple_classifiers import NNClassifier

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loaded_data = np.load('power_bands_data.npz')

# Access each array by its name
left_amygdala_power_bands = loaded_data['left_amygdala_power_bands']  # Shape: (320, 5)
right_amygdala_power_bands = loaded_data['right_amygdala_power_bands']
left_hippocampus_power_bands = loaded_data['left_hippocampus_power_bands']
right_hippocampus_power_bands = loaded_data['right_hippocampus_power_bands']
left_hippocampus_pos_power_bands = loaded_data['left_hippocampus_pos_power_bands']
labels = loaded_data['labels'].ravel()  # Ensure labels is a 1D array

print('Labels shape:', labels.shape)

# Define frequency bands
freq_bands = ['band1', 'band2', 'band3', 'band4', 'band5']

# Initialize an empty DataFrame
data = pd.DataFrame()

# Function to add features to the DataFrame
def add_features(region_data, region_name):
    for i, band in enumerate(freq_bands):
        data[f'{region_name}_{band}'] = region_data[:, i]

# Add features from each region
add_features(left_amygdala_power_bands, 'LA')
add_features(right_amygdala_power_bands, 'RA')
add_features(left_hippocampus_power_bands, 'LH')
add_features(right_hippocampus_power_bands, 'RH')
add_features(left_hippocampus_pos_power_bands, 'LHP')

# Add labels
data['label'] = labels

# Separate features and target
X = data.drop('label', axis=1)
y = data['label']
print('X',X.shape)
# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

# Split the data
X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Convert to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train).float()
Y_train_tensor = torch.from_numpy(Y_train.to_numpy()).float().unsqueeze(1)
X_val_tensor = torch.from_numpy(X_val).float()
Y_val_tensor = torch.from_numpy(Y_val.to_numpy()).float().unsqueeze(1)


# Initialize classifier
input_dim = X.shape[1]  # Number of features (should be 25)
classifier = NNClassifier(input_dim).to(device)

# Loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(classifier.parameters(), lr=0.001, weight_decay=1e-4)

# Training parameters
num_epochs = 200
batch_size = 128

for epoch in range(num_epochs):
    classifier.train()
    epoch_loss = 0

    # Shuffle the training data
    indices = torch.randperm(X_train_tensor.size(0))
    X_train_tensor = X_train_tensor[indices]
    Y_train_tensor = Y_train_tensor[indices]

    # Batch training
    for i in range(0, len(X_train_tensor), batch_size):
        # Extract batches
        X_batch = X_train_tensor[i:i+batch_size].to(device)
        Y_batch = Y_train_tensor[i:i+batch_size].to(device)

        # Forward pass
        outputs = classifier(X_batch)
        loss = criterion(outputs, Y_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss / len(X_train_tensor):.4f}")

    # Validation step
    classifier.eval()  # Set the classifier to evaluation mode
    with torch.no_grad():
        outputs = classifier(X_val_tensor.to(device))
        val_loss = criterion(outputs, Y_val_tensor.to(device)).item()
        predictions = (outputs > 0.5).float()  # Convert probabilities to binary predictions
        accuracy = (predictions == Y_val_tensor.to(device)).float().mean().item()

    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy * 100:.2f}%")
# Save model parameters and state
save_path = "./pretrained_models/AHN_nn_classifier.pt"

torch.save({
    'input_dim': input_dim,  # Save input dimension
    'model_state_dict': classifier.state_dict(),  # Save model weights
    'optimizer_state_dict': optimizer.state_dict(),  # Save optimizer state (optional)
    'training_params': {  # Save any additional parameters
        'batch_size': batch_size,
        'learning_rate': 0.001,
        'num_epochs': num_epochs,
    }
}, save_path)

print(f"Model saved to {save_path}")
