import numpy as np
import scipy.io as sio
import os
import torch
import util
import plotting
from GCE import GenerativeCausalExplainer
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
X = scaler.fit_transform(X)

#--- parameters ---
# dataset
data_classes = [0,1]
# vae
K = 2
L = 3
train_steps = 2000
Nalpha = 25
Nbeta = 100
lam = 0.05
batch_size = 64
lr = 5e-4
# other
randseed = 0
retrain_gce = True # train explanatory VAE from scratch
save_gce = True # save/overwrite pretrained explanatory VAE at gce_path


# --- initialize ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if randseed is not None:
    np.random.seed(randseed)
    torch.manual_seed(randseed)
ylabels = range(0,len(data_classes))


# --- load data ---
'''
from load_mnist import load_mnist_classSelect
X, Y, tridx = load_mnist_classSelect('train', data_classes, ylabels)
vaX, vaY, vaidx = load_mnist_classSelect('val', data_classes, ylabels)
ntrain, nrow, ncol, c_dim = X.shape
x_dim = nrow*ncol
'''

from sklearn.model_selection import train_test_split

# Assuming X and Y are already NumPy arrays
# Split the data: 80% training, 20% validation
X, vaX, Y, vaY = train_test_split(X, y, test_size=0.2, random_state=42)
x_dim = X.shape[1]
np.savez('train_val_split.npz', X=X, vaX=vaX, Y=Y, vaY=vaY)
print("Variables saved to train_val_split.npz")

# Check the shapes
#print(f"X_train shape: {X.shape}")
#print(f"Y_train shape: {Y.shape}")
#print(f"X_val shape: {vaX.shape}")
#print(f"Y_val shape: {vaY.shape}")


# --- load classifier ---
'''
from models.CNN_classifier import CNN
classifier = CNN(len(data_classes)).to(device)
checkpoint = torch.load('%s/model.pt' % classifier_path, map_location=device)
classifier.load_state_dict(checkpoint['model_state_dict_classifier'])
'''
from models.simple_classifiers import NNClassifier
classifier = NNClassifier(25).to(device)
classifier_path = './pretrained_models/neural_nn_classifier.pt'
checkpoint = torch.load(classifier_path, map_location=device)
input_dim = checkpoint['input_dim']
model_state_dict = checkpoint['model_state_dict']
classifier = NNClassifier(input_dim).to(device)
classifier.load_state_dict(model_state_dict)
classifier.eval()
# --- train/load GCE ---
#from models.CVAE import Decoder, Encoder
from models.VAE import Decoder, Encoder
if retrain_gce:
    for i in range(1,5):
        train_steps = i*2000
        print('train_steps: ',train_steps)
        encoder = Encoder(x_dim, K+L).to(device)
        decoder = Decoder(x_dim, K+L).to(device)
        encoder.apply(util.weights_init_normal)
        decoder.apply(util.weights_init_normal)
        gce = GenerativeCausalExplainer(classifier, decoder, encoder, device,debug_print = False)
        traininfo = gce.train(X, K, L,
                            steps=train_steps,
                            Nalpha=Nalpha,
                            Nbeta=Nbeta,
                            lam=lam,
                            batch_size=batch_size,
                            lr=lr)
        if save_gce:
            gce_path = './GCE_models/neural'
            if not os.path.exists(gce_path):
                os.makedirs(gce_path)
            torch.save(gce, os.path.join(gce_path,'neural_model'+'_steps_'+str(train_steps)+'.pt'))
            sio.savemat(os.path.join(gce_path, 'training_info'+'_steps_'+str(train_steps)+'.mat'), {
                'data_classes' : data_classes, #'classifier_path' : classifier_path,
                'K' : K, 'L' : L, 'train_step' : train_steps, 'Nalpha' : Nalpha,
                'Nbeta' : Nbeta, 'lam' : lam, 'batch_size' : batch_size, 'lr' : lr,
                'randseed' : randseed, 'traininfo' : traininfo})
        # --- compute final information flow ---
        I = gce.informationFlow()
        Is = gce.informationFlow_singledim(range(0,K+L))
        print('num_steps -> ',train_steps )
        print('Information flow of K=%d causal factors on classifier output:' % K)
        print(Is[:K])
        print('Information flow of L=%d noncausal factors on classifier output:' % L)
        print(Is[K:])
else: # load pretrained model
    gce = torch.load(os.path.join(gce_path,'neural_model.pt'), map_location=device)


# --- compute final information flow ---
I = gce.informationFlow()
Is = gce.informationFlow_singledim(range(0,K+L))
print('num_steps -> ',train_steps )
print('Information flow of K=%d causal factors on classifier output:' % K)
print(Is[:K])
print('Information flow of L=%d noncausal factors on classifier output:' % L)
print(Is[K:])
