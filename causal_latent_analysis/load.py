import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler

# Load data
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

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the classifier
gnb = GaussianNB()

# Define the cross-validator
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation and compute accuracy
cv_scores = cross_val_score(gnb, X_scaled, y, cv=cv, scoring='accuracy')

print(f'Cross-validated Accuracy: {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}')

# Get cross-validated predictions
y_pred = cross_val_predict(gnb, X_scaled, y, cv=cv)

# Classification report
print(classification_report(y, y_pred))
