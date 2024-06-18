'''
The following script contains the necessary methods to evaluate a model
'''

import os
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve

# Behaviour to evaluate
BEHAVIOR_NAMES = [
    'Grooming',
    'Rearing'
]

def load_dataset(path, backbone, sets=['validation']):
    '''
    Load the dataset
    '''
    if backbone not in ['resnet', 'inception_resnet']:
        raise Exception('Invalid backbone')
    dataset = {}
    for set in sets:
        feature_files = [f for f in sorted(os.listdir(os.path.join(path, backbone, set, 'features')))]
        label_files = [f for f in sorted(os.listdir(os.path.join(path, backbone, set, 'labels')))]
        
        for f, l in zip(feature_files, label_files):
            name = f.split('.')[0]
            features = np.load(os.path.join(path, backbone, set, 'features', f))
            l = pd.read_csv(os.path.join(path, backbone, set, 'labels', l))
            # If dataframe has 2 columns for rearing (mid and wall), we can convert them to just one
            if 'Mid Rearing' and 'Wall Rearing' in l.columns:
                l["Rearing"] = np.max(l[['Mid Rearing', 'Wall Rearing']], axis=1)
                l = l.drop(['Mid Rearing', 'Wall Rearing'], axis = 1)
            labels = l.values
            
        
            dataset[name] = {
                'features': features,
                'labels': labels,
            }

    return dataset

def generate_sequences(data, seq_length, normalize=False):
    '''
    Generate the sequences by splitting the data
    '''
    # Normalize values
    if normalize:
        data['features'] = data['features'] / data['features'].max()


    # Check if number of frames is divisble by the sequence length
    if len(data['features']) // seq_length == 0:
        x = np.array([data['features'][i:i+seq_length] for i in range(0, data['features'].shape[0],seq_length)])
        y = np.array([data['labels'][i:i+seq_length] for i in range(0, data['labels'].shape[0], seq_length)])
    
    else:
        # If number of frames is no divisible by the sequence_length  we will leave the last few
        x = np.array([data['features'][i:i+seq_length] for i in range(0,  len(data['features']) - (len(data['features'])% seq_length), seq_length)])
        y = np.array([data['labels'][i:i+seq_length] for i in range(0,  len(data['labels']) - (len(data['labels'])% seq_length), seq_length)])
        # We will add the last few with the necessary previous ones
        x = np.concatenate((x, np.array([data['features'][-seq_length:]])))
        y = np.concatenate((y, np.array([data['labels'][-seq_length:]])))   
   
    return x, y


def plot_behaviours(label, pred, behaviour):
    '''
    This method crates a graphic representation of the behaviour evolution along the video, it serves as a comparisson between the labels and the predictions 
    '''
    sns.set_theme(rc={'figure.figsize':(15,5)})
    plt.plot(label, color='sandybrown')
    plt.fill_between(range(len(label)),label, color='sandybrown', alpha=0.3)
    plt.plot(pred)
    plt.legend(['Label', '', 'Prediction'])
    plt.xlabel('Frames')
    plt.ylabel('Probability')
    plt.title(f'Prediction vs Label for {behaviour}')
    plt.show()

def plot_confussionmatrix(label, pred, behaviour, threshold=0.3):
    '''
    This method plots a Confusion Matrix given a prediction and its labels
    '''
    cm = confusion_matrix(label, pred > threshold)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f'Confusion matrix for {behaviour}')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

def plot_PRC_Curve(label, pred, behaviour):
    '''
    This method plots the PRC curve given a prediction and its labels
    '''
    precision, recall, thresholds  = precision_recall_curve(label, pred)
    
    sns.set_theme(rc={'figure.figsize':(5,5)})
    plt.plot(thresholds, precision[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recall[:-1], 'r--', label='Recall')
    plt.xlabel('Threshold')
    plt.legend(loc='lower left')
    plt.ylim([0,1])
    plt.title(f'Optimal thershold for {behaviour}')
    plt.show()


