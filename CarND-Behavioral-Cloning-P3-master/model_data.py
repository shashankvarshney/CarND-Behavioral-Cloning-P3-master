# Importing all the required libraries
import csv
import numpy as np
from scipy import ndimage
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import keras
import cv2

#Input image path and driving log paths.
image_path = "/opt/second_track/IMG/"
driving_log = "/opt/second_track/driving_log.csv"

#List placeholder to store all the line data. 
lines = []


#Reading all the lines from the log file and appending
#lines to placeholder.
with open(driving_log) as file:
    reader = csv.reader(file)
    for line in reader:
        lines.append(line)


#Placeholder to store images and measurement data.		
images = []
measurements = []


# Reading all files from the image directory
for line in lines:
	#Reading center, left and right image from the collected data.
    center_image_name = line[0].split("\\")[-1]
    left_image_name = line[1].split("\\")[-1]
    right_image_name = line[2].split("\\")[-1]
	
	#Correction in measurement for left and right images.
    correction = 0.2
	
	#Steering data for center, left and right images.
    center_steering = float(line[3])
    left_steering = center_steering + correction
    right_steering = center_steering - correction
    #print(filename)
	
	#Reading all the image files and cropping it to get only the road view.
    center_image = ndimage.imread(image_path + center_image_name)[71:136, :, :]
    left_image = ndimage.imread(image_path + left_image_name)[71:136, :, :]
    right_image = ndimage.imread(image_path + right_image_name)[71:136, :, :]
    #center_image = cv2.resize(center_image, None, fx = 0.25, fy = 0.25, interpolation = cv2.INTER_AREA)
    #left_image = cv2.resize(left_image, None, fx = 0.25, fy = 0.25, interpolation = cv2.INTER_AREA)
    #right_image = cv2.resize(right_image, None, fx = 0.25, fy = 0.25, interpolation = cv2.INTER_AREA)
    #print(center_image.shape, left_image.shape, right_image.shape)
	
	#Appending all 3 images to image placeholder
    images.append(center_image)
    images.append(left_image)
    images.append(right_image)
	
	#Appending measurment data of all 3 images.
    measurements.append(center_steering)
    measurements.append(left_steering)
    measurements.append(right_steering)
    
 

#Placeholders for augmented data and augmented measurements.
augmented_images = []
augmented_measurements = []


#Flipping all the images and storing them along with original images.
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(np.fliplr(image))
    augmented_measurements.append(measurement*-1.0)


#Creating training data
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)


#Creating the model by using architecture of NVIDIA
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = (65, 320, 3)))
model.add(Convolution2D(24, 5, 5, subsample = (2, 2), activation = "relu"))
model.add(Convolution2D(36, 5, 5, subsample = (2, 2), activation = "relu"))
model.add(Convolution2D(48, 5, 5, subsample = (2, 2), activation = "relu"))
model.add(Convolution2D(64, 3, 3, activation = "relu"))
model.add(Convolution2D(64, 3, 3, activation = "relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


#Compiling and fitting the model.
model.compile(loss = "mse", optimizer = "adam")
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, epochs = 2)


#Saving the model.
model.save("model_data.h5")