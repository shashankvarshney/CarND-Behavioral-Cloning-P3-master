#Importing all required libraries

import csv
import numpy as np
from scipy import ndimage
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import keras
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import sklearn


# Saving image and driving log paths to 2 variables.
image_path = "/opt/second_track/IMG/"
driving_log = "/opt/second_track/driving_log.csv"


#Creating a placeholder for storing the lines from the driving logs file.
samples = []


# Reading the lines from the driving log file and storing it in the placeholder.
with open(driving_log) as file:
    reader = csv.reader(file)
    for line in reader:
        samples.append(line)

        

# Dividing the data into training and validation set.
train_samples, validation_samples = train_test_split(samples, test_size=0.2, random_state = 20)



# Creating a generator function so that model can data in batches and need not to store
# whole data in the memory.
def generator(samples, batch_size):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples, random_state = 1)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
				#Creating names for all 3 images.
                center_name = image_path+batch_sample[0].split('\\')[-1]
                left_name = image_path+batch_sample[1].split('\\')[-1]
                right_name = image_path+batch_sample[2].split('\\')[-1]
				
				#Reading al three images and cropping them.
                center_image = ndimage.imread(center_name)[71:136, :, :]
                left_image = ndimage.imread(left_name)[71:136, :, :]
                right_image = ndimage.imread(right_name)[71:136, :, :]
				
				# Reading steering measurement and correcting it for left
				# and right images as well.
                correction = 0.2
                center_angle = float(batch_sample[3])
                left_angle = center_angle + correction
                right_angle = center_angle - correction
				
				# Storing all the images and steering measurements.
                images.append(center_image)
                images.append(left_image)
                images.append(right_image)
				
				# Creating the flipped images for data augmentation.
                images.append(np.fliplr(center_image))
                images.append(np.fliplr(left_image))
                images.append(np.fliplr(right_image))
                angles.append(center_angle)
                angles.append(left_angle)
                angles.append(right_angle)
				
				# Creating the augmented measurements.
                angles.append(center_angle*-1.0)
                angles.append(left_angle*-1.0)
                angles.append(right_angle*-1.0)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train, random_state = 10)

# declaring batch size.            
batch_size = 128  

# creating generator for training and validation set.          
train_generator = generator(train_samples, batch_size)
validation_generator = generator(validation_samples, batch_size)


#Creating model. NVIDIA model has been used.
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


#Compiling the model.
model.compile(loss='mse', optimizer='adam')

#Fitting the model along with the fit_generator.
model.fit_generator(train_generator, steps_per_epoch= int(len(train_samples)/batch_size), 
                    validation_data=validation_generator, 
                    validation_steps=int(len(validation_samples)/batch_size), epochs=6, verbose = 1, 
                    use_multiprocessing=True)

					
# Saving the model
model.save("model_data_1_1.h5")
print("model has been saved")