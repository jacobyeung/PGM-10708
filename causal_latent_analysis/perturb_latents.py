import numpy as np
import scipy.io as sio
import os
import torch
import matplotlib.pyplot as plt
import util
from models.simple_classifiers import NNClassifier
import plotting
from GCE import GenerativeCausalExplainer
import numpy as np
import pandas as pd
from util import binary_accuracy
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler

K = 2 #causal
L = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loaded_data = np.load('train_val_split.npz')
X = loaded_data['X']
vaX = loaded_data['vaX']
Y = loaded_data['Y']
vaY = loaded_data['vaY']

#pass X trough autoencoder to get X_hat
train_steps = 4000
gce_path = './GCE_models/neural'
gce = torch.load(os.path.join(gce_path,'neural_model'+'_steps_'+str(train_steps)+'.pt'), map_location=device)

#load pretrained classifier
classifier_path = './pretrained_models/neural_nn_classifier.pt'  # Adjust as necessary
checkpoint = torch.load(classifier_path, map_location=device)
input_dim = checkpoint['input_dim']
model_state_dict = checkpoint['model_state_dict']
classifier = NNClassifier(input_dim).to(device)
classifier.load_state_dict(model_state_dict)
classifier.eval()


# --- Reconstruct X_val to obtain X_hat ---
X_val_tensor = torch.tensor(vaX, dtype=torch.float32).to(device)
X_train_tensor = torch.tensor(X, dtype=torch.float32).to(device)

# Get latent representation and reconstruct
encoder = gce.encoder
decoder = gce.decoder

latent_val, _, _ = encoder(X_val_tensor)  
latent_train, _, _ = encoder(X_train_tensor)
deltas = [-1,-0.5,0.5,1]

results =[]
results_C = []
results_NC =[]
perturbs = []

for i in range(20):
    #perturb causal latents
    latent_val_pert_C = latent_val.clone()
    perturb = []
    for j in range(K):
        delta = np.random.choice(deltas)
        perturb.append(delta)
        latent_val_pert_C[:,j] += delta
    
    latent_val_pert_NC = latent_val.clone()
    for k in range(L):
        delta = np.random.choice(deltas)
        perturb.append(delta)
        latent_val_pert_NC[:,k+K] += delta
    
    perturbs.append(perturb)
    X_hat_val = decoder(latent_val)  # Extract only the reconstructed data
    X_hat_val_pert_C = decoder(latent_val_pert_C)
    X_hat_val_pert_NC = decoder(latent_val_pert_NC)
    # apply classifier

    x_val_pred = classifier(X_hat_val.to(device))
    x_val_pred_pert_C = classifier(X_hat_val_pert_C.to(device))
    x_val_pred_pert_NC = classifier(X_hat_val_pert_NC.to(device))

    results.append(binary_accuracy(x_val_pred,vaY))
    results_C.append(binary_accuracy(x_val_pred_pert_C,vaY))
    results_NC.append(binary_accuracy(x_val_pred_pert_NC,vaY))


original = np.array(results)
causal = np.array(results_C)
non_causal = np.array(results_NC)


