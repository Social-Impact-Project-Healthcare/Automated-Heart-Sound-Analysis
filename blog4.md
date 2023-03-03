# Deep Learning based Heart Sound Classification Example

## Overview of Transfer Learning
Transfer learning is a machine learning technique that involves leveraging the knowledge gained from training a machine learning model on one task and applying it to another task. This technique can be particularly useful in situations where there is limited data available for the task at hand or where the training data is expensive or time-consuming to acquire.

In transfer learning, a pre-trained model is used as a starting point, and its learned features are transferred to a new model that is trained on the target task. The pre-trained model can be trained on a similar or related task, or even on a completely different domain, as long as it has learned useful features that can be transferred to the target task.

There are several different approaches to transfer learning, including fine-tuning, feature extraction, and domain adaptation. Fine-tuning involves retraining the pre-trained model on the new task, adjusting its weights to better fit the new data. Feature extraction involves using the pre-trained model as a fixed feature extractor and training a new model on top of these features. Domain adaptation involves adapting the pre-trained model to the new task by adjusting its weights to better match the distribution of the new data.

This code explains the use of Yamnet as a feature extractor in heart sound classification.

## Dataset Preperation
We have used a publicly available dataset which can be found on GitHub. The link : https://github.com/yaseen21khan/Classification-of-Heart-Sound-Signal-Using-Multiple-Features-

The dataset consists of 1000 data points belong to five classes. They are as follows,

* AS - Aortic Stenosis
* MR - Mitral Regergitation
* MS - Mitral Stenosis
* MVP - Mitral Valve Prolapse
* N - Normal

Each class includes 200 datapoints and It's well balanced.

## Import Libraries

```python
import os
import librosa.util
import librosa
from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio

import random
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
```
## Load Yamnet
YAMNet is a pre-trained deep neural network that can predict audio events from 521 classes, such as laughter, barking, or a siren. This will be using as the feature extractor later. A bit about Yamnet,

* It employs the MobileNetV1 depthwise-separable convolution architecture.
* This version uses frames that are 0.96 second long and extracts one frame every 0.48 seconds.
* It accepts single-channel (mono) 16 kHz samples in the range (-1.0, +1.0).
* It returns 3 outputs, including the class scores, embeddings and the log mel spectrogram.

```python
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)
```
## Preprocess and load the Datapoints
In this step, all the datapoints are resampled to 16kHz and normalized. Label map is applied considering all the five classes.

```python
# Define the dataset directory
dataset_dir = ''

# Define the audio parameters
sr = 16000  # sampling rate

# Load the audio files and labels
audio_files = []
labels = []
for label in os.listdir(dataset_dir):
    label_dir = os.path.join(dataset_dir, label)
    if not os.path.isdir(label_dir):
        continue
    for audio_file in os.listdir(label_dir):
        if not audio_file.endswith('.wav'):
            continue
        audio_path = os.path.join(label_dir, audio_file)
        audio, _ = librosa.load(audio_path, sr=sr, mono=True)
        # Normalize the audio to have maximum absolute value of 1
        audio = librosa.util.normalize(audio)
        audio_files.append(audio)
        labels.append(label)

label_map = {
    "N_New" : 0,
    "AS_New" : 1,
    "MR_New" : 2,
    "MS_New" : 3,
    "MVP_New" : 4,
}

# Convert the original class labels to your desired labels using the dictionary
labels  = [label_map[label] for label in labels]
```

## Extract Embeddings and Split the Dataset
Now we have the audio files list and labels list. As the next step we need to extract the embeddings of each signal and label them. The prepared nested list will be shuffeled and splitted into feature list and target list. At last, they will modified as numpy arays which facilitates trainable dataset. We have used randomly splitted prtion of datapoints as train set. Rest if for testing purposes.

```python
# applies the embedding extraction model to a wav data
ds = []
train_length = len(audio_files)
for index in range(train_length):
    scores, embeddings, spectrogram = yamnet_model(audio_files[index])
    num_embeddings = tf.shape(embeddings)[0]
    
    for i in range (num_embeddings):
        ds.append([embeddings[i],labels[index]])

# Shuffle the nested list
random.shuffle(ds)

X = np.array([x[0] for x in ds])
y = [x[1] for x in ds]

y = to_categorical(y, 5)
y= np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size = .8)

```

## Model Architecture and Training
At the point we have the required feature vectors and target vectors with us. Here we have occupied a simple classification architecture with three dense layers and one dropout layer. We introduce non-linearity with Relu activation function at each dense layer. The purpose of using a 0.6 dropout is to make things harder for the model to predict classes. Model outputs an array of probability value of each class sums upto 1. We use Adam optimizer with categorical crossentropy loss function. In the training process we occupy Early Stopping to avoid overfitting the model with the training set.

```python
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

classification_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1024), dtype=tf.float32,name='input_embedding'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(.6),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
], name='classification_model')

classification_model.summary()

# Compile the model
classification_model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.0001),
              metrics=['accuracy'])

# Train the model
classification_model.fit(X_train,y_train,
          batch_size=32,
          epochs=100,
          validation_split =0.2,
          shuffle=True,
          callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode='min')])
```
## Evaluate the Performance
The trained model is capable of classifying the heart sounds with more than 90% accuracy.

```python
loss, accuracy = classification_model.evaluate(X_test,y_test)

print("Loss: ", loss)
print("Accuracy: ", accuracy)
```

* Loss:  0.2620827257633209 
* Accuracy:  0.9137001037597656

Thank you and congratulations! You made it. Please do comment any improvements and try things on your spare time. Kaggle Notebook : https://www.kaggle.com/code/asirisirithunga/heart-sound-signal-classification

Reference : https://www.tensorflow.org/tutorials/audio/transfer_learning_audio