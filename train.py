# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 15:09:12 2021

@author: hiren
"""

# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

DIRECTORY = r"C:\Users\hiren\Desktop\Face Mask Detector\dataset"
CATEGORIES = ["with_mask", "without_mask"]

print("[INFO] loading images...")

data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category) #joining the directory & category
    for img in os.listdir(path): #lists down all the images in category
    	img_path = os.path.join(path, img)
    	image = load_img(img_path, target_size=(224, 224)) #using tensorflow.keras
    	image = img_to_array(image)
    	image = preprocess_input(image)

    	data.append(image) # appending array in data list
    	labels.append(category) # appending category in labels list

# convert text to Binary
lb = LabelBinarizer() # converting categories into categorical variables
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32") # converted binary to numpy arrays
labels = np.array(labels) # deep learning model works on numpy arrys

(trainX, testX, trainY, testY) = train_test_split(data, labels, 
            test_size=0.20, random_state=42) # splitting training and testing data
# test size means 20% of dataset is used for testing

# initialize the initial learning rate
INIT_LR = 1e-4
EPOCHS = 20 # how many times our network “sees” each training example and learns patterns from it
BS = 32 # batch size of images

# construct the training image generator for data augmentation
# Generate batches of tensor image data with real-time data augmentation
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2, # These parameters will help in transforming the image vectors for maximum feature extraction
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# load the MobileNetV2 model
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3))) # 3 is used for coloured and 1 is used for grey 

# Create head and the base model
headModel = baseModel.output # creating a new model using base model output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel) # Downsamples the input along its spatial dimensions
headModel = Flatten(name="flatten")(headModel) # Flattens the input
headModel = Dense(128, activation="relu")(headModel) # dense the input using 128 neurons and activation relu is used when dealing with images
headModel = Dropout(0.5)(headModel) # used to avoid overfitting of models
headModel = Dense(2, activation="softmax")(headModel) # output is 2 layers as it has 2 categories

# Call head and the base model
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them
for layer in baseModel.layers:
	layer.trainable = False

# compile our model V2
print("Compilation of the MODEL is going on...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS) # used the ADAM algorithm
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"]) # Used as a loss function for binary classification model. 
#The binary_crossentropy function computes the cross-entropy loss between true labels and predicted labels.

# train the head of the network
print("Training Head Started...")
H = model.fit( # After creaation of ImageDataGenerator, must fit it on your data.
	aug.flow(trainX, trainY, batch_size=BS),  # get batches of images by calling the flow() function
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# make predictions on the testing set
print("Network evaluation...")
predIdxs = model.predict(testX, batch_size=BS)
# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# serialize the model to disk
print("saving mask model...")
model.save("mask_detector.model", save_format="h5")

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")