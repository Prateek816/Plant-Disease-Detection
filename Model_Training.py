#importin libraries
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

#data Preprocessing

##imagePreprocessing
training_set = tf.keras.utils.image_dataset_from_directory(
    'train',
    labels="inferred",
    label_mode="categorical",#for multi-class classification
    class_names=None,
    color_mode="rgb",
    batch_size=32, #to speed up the training process , increase the value of batch size
    image_size=(128, 128), # original size of images is 256x256
    shuffle=True, #shuffling means it will randomly select images for training(not in linear way)
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
    data_format=None, #YT removed this #3 14:03
    format="tf",
    verbose=True,
)

#Validiation Image Preprocessing

validation_set = tf.keras.utils.image_dataset_from_directory(
    'valid',
    labels="inferred",
    label_mode="categorical",#for multi-class classification
    class_names=None,
    color_mode="rgb",
    batch_size=32, #to speed up the training process , increase the value of batch size
    image_size=(128, 128), # original size of images is 256x256
    shuffle=True, #shuffling means it will randomly select images for training(not in linear way)
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
    data_format=None,
    format="tf",
    verbose=True,
)

print(training_set) #remove this line later

for x,y in training_set:
    print(x,x.shape)
    print(y,y.shape)
    break

#ending of #3
class_names = validation_set.class_names
print(class_names)
##model building
#to aboid overshooting -
#choose small learning rate default 0.001 we are choosing 0.0001
#There may be chance of underfitting , so we increase the nummber of neurons
#add more convolutional layers to extract more features from images , there may be possiblitiy
#that model unable to captuer the relavent feature from images
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D,Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow import keras

model = Sequential([
    # Input Layer
    tf.keras.Input(shape=(128,128,3)),

    # Block 1
    Conv2D(32, (3,3), padding='same', activation='relu'),
    Conv2D(32, (3,3), activation='relu'),
    MaxPool2D(pool_size=2, strides=2),

    # Block 2
    Conv2D(64, (3,3), padding='same', activation='relu'),
    Conv2D(64, (3,3), activation='relu'),
    MaxPool2D(pool_size=2, strides=2),

    # Block 3
    Conv2D(128, (3,3), padding='same', activation='relu'),
    Conv2D(128, (3,3),  activation='relu'),
    MaxPool2D(pool_size=2, strides=2),

    # Block 4
    Conv2D(256, (3,3), padding='same', activation='relu'),
    Conv2D(256, (3,3), activation='relu'),
    MaxPool2D(pool_size=2, strides=2),

    Conv2D(512, (3,3), padding='same', activation='relu'),
    Conv2D(512, (3,3), activation='relu'),
    MaxPool2D(pool_size=2, strides=2),

    Dropout(0.25),#to avoid overfitting
    # Dense Layers
    Flatten(),
    Dense(units=1500, activation='relu'),
    Dropout(0.25),

    # Output Layer
    Dense(38, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(
    learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


#training the model

training_history =model.fit(x=training_set,validation_data = validation_set,epochs=10)

#model evaluation
trail_loss_train_acc = model.evaluate(training_set)

#Model evaluation on validation set
val_loss,val_acc = model.evaluate(validation_set)

#Saving Model
model.save('trained_model.h5') #save as .keras as well (it will take less size)

hist = training_history.history

 #Recoring History in json

import json
with open('history.json','w') as f:
        json.dump(hist,f)

#Accuracy Visualization
epochs = {i for i in range(1,11)}
plt.plot(epochs,hist['accuracy'],color='red',label='Training Accuracy')
plt.plot(epochs,hist['val_accuracy'],color='blue',label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#some other metrics for model evaluation

class_name = validation_set.class_names

test_set =  tf.keras.utils.image_dataset_from_directory(
    'valid',
    labels="inferred",
    label_mode="categorical",#for multi-class classification
    class_names=None,
    color_mode="rgb",
    batch_size=32, #to speed up the training process , increase the value of batch size
    image_size=(128, 128), # original size of images is 256x256
    shuffle=False, #shuffling means it will randomly select images for training(not in linear way)
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
    data_format=None,
    format="tf",
    verbose=True,
)

y_pred = model.predict(test_set)

predicted_classes = np.argmax(y_pred, axis=1)

true_categories = np.concat([y for x, y in test_set], axis=0)
  
y_true = np.argmax(true_categories, axis=1)

from sklearn.metrics import classification_report, confusion_matrix

classification_report = classification_report(y_true, predicted_classes, target_names=class_name)

cm = confusion_matrix(y_true, predicted_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_name, yticklabels=class_name)

