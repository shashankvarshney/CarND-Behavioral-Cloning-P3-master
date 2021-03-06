# Project Report : Behavioral Cloning


[image1]: ./example_image.jpg "Example Image"
[image2]: ./cropped_image.jpg "Cropped Image"
[image3]: ./flipped_image.jpg "Flipped Image"
[image4]: ./model_architecture.png "Model Architecture"


### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model_data.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model_data.h5 containing a trained convolution neural network 
* project_report.md summarizing the results


#### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing **python drive.py model_data.h5**


#### 3. Submission code is usable and readable

The **model_data.py** file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

As learnt in the classroom, I have employed NVIDIA model presented by Autonomous driving team. Following is the reference architecure of the model taken from NVIDIA website.

![Reference_acrhitecture][image4]

In this acrhitecture Normalizing layer is being realized with **Lambda** layer of Keras. Normalizing technique and mean centering the data has been used for pre-processing the data.

Also cropped image size is **65x320x3** which is different than the reference architecture. 

Architecture contains total 9 hidden layers and input and output layers.

Total 5 convolution layer has been used. Out of 5 convolution layers, first 3 layers are using stride size of **(2, 2)**.

After convolution, 1 Flatten layer and 3 Dense layers are used to match the reference architecture.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 65x320x3 RGB image   							| 
| Convolution 5x5 (ReLU)| 2x2 stride, valid padding, outputs 31x158x24 	|
| Convolution 5x5 (ReLU)| 2x2 stride, valid padding, outputs 14x77x36 	|
| Convolution 5x5 (ReLU)| 2x2 stride, valid padding, outputs 5x37x48 	|
| Convolution 3x3 (ReLU)| 1x1 stride, valid padding, outputs 3x35x64 	|
| Convolution 3x3 (ReLU)| 1x1 stride, valid padding, outputs 1x33x64 	|
| Flatten	            | Output 2112 neurons      						|
| Dense		            | Output 100 neurons        					|
| Dense				    | Output 50 neurons        						|
| Dense					| Output 10 neurons								|
| Dense (Output Layer)	| Steering measurement Output					|


#### 2. Attempts to reduce overfitting in the model

With the training data set, model was fitting well and not very high overfitting was observed. The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually 


#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a 2 laps of center lane driving with occasional recovering from drifting towards edge of the road for both the tracks. I also included 2 laps of similar driving by driving in counter clockwise for both the tracks. 


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I followed the learnings from the classroom and used the similar technique. I followed following techniques to reach to the final model :

1. I started with very simple model with one convolution layer and one output to ensure that everything was working fine.

2. Data was collected and all 3 camera images (center, left and right) has been used to build the model.

3. More data was generated by augmenting the data. Augmentation was done by flipping the images and taking the negative of the steering measurement.

4. Dataset contains total 42888 images from both the tracks and after including flipped images as well I doubled the size of the dataset.

5. Dataset was broken into training and validation set and 20% of the overall dataset was used for validation to ensure no overfitting should be there. Also I shuffled the dataset before breaking it into training and validation set.

6. First I tried with dataset from the 1 lap driving but that dataset was overfitting so I collected more data so that model can generalize well as second track was more tricky with lots of curves and turns.

7. After ensuring that model is fitting and generalizing well, I used the saved model to run with simulator in autonomous mode. I found that vehicle was driving well in both the tracks. 

8. To reach to the final solution, I required to collect the data 4-5 times as track 1 was working fine but there was issue with track 2. In the end I collected almost equal amount of data from both the tracks. I collected 2 laps in clockwise and 2 laps in counter-clockwise from both the tracks.

9. I have used generator so that data can be fed to model in batches. This is will help to reduce the memory requirement as whole of the data is not required to put into the memory while training the model.

10. I used reading of images and steering data and also flipping the images as a part of data augmentation in the generator instead of reading and augmenting the data first and then use the generator.

11. As per classroom, I should have used **steps_per_epoch= len(train_samples)** and  **validation_steps=len(validation_samples)** which increases the training time drastically. I searched in keras documentation and found that this should be **steps_per_epoch= int(len(train_samples)/batch_size)** and **validation_steps=int(len(validation_samples)/batch_size)** respectively which gave very good improvement in training time.




#### 2. Final Model Architecture

Final model architecture has been discussed in the section **1. An appropriate model architecture has been employed**. 


#### 3. Creation of the Training Set & Training Process

I recorded 2 laps in clockwise and 2 laps in counter-clockwise from both the tracks. During data collection, I enured that i should collect some occasional drifting to sides of the road and recovering back to middle of the road.

I cropped the image while reading them from the files instead of cropping it by using Keras **Cropping2D** layer. This gave me advantage to train the model without using generator as it reduced overall size of the image before putting them int training process.

Cropping was required so that we can only have road part of the image instead of trees, hills, etc which was of no use while fitting the model and also distracting the model and ultimately drifting the car from the road.

I also augmented the data by using flipped images as well. This helped me in combating overfitting and also helped to generalize the model well for both the tracks.

Following are the example of original image, cropped image and flipped image.

![original_image][image1]
![cropped_image][image2]
![flipped_image][image3]


Finally while fitting the model, 20% of data was used for validation purposes.

As I had already cropped the images before loading them into the model so I had to make one change in **drive.py** file so that autonomous driving can work well.

I had to change **image_array = np.asarray(image)** to **image_array = np.asarray(image)[71:136, :, :]** in **telemetry()** function so that image should be cropped right after reading.