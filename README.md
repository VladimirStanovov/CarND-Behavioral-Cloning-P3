# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model_track1.py containing the script I used to create model for track 1
* model_track1.h5 containing a trained convolution neural network for track 1
* model_track2.py containing the script I used to create model for track 2
* model_track2.h5 containing a trained convolution neural network for track 2
* drive.py for driving the car in autonomous mode
* README.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model_track1.h5
```
 or
 ```sh
python drive.py model_track2.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model used for track 1 consists of a convolution neural network with 3x3 filter sizes and 4 layers of depths between 32 and 96 (model.py lines 223-230) with maxpooling between them.

The model includes RELU layers to introduce nonlinearity (code line 224), and the data is normalized in the model using a Keras lambda layer (code line 221). After flattening, there were 3 fully connected layers (1536, 128 and 16 neurons).

The model for track 2 is more compicated, as it contains 5 convolutional layers, with 3x3 filters, and 4 fully connected layers (1164, 100, 50, 10 neurons, this part is like in NVIDIA DAVE-2 architecture).

####2. Attempts to reduce overfitting in the model

The model does not contain any dropout layers, because introducing them did not deliver any significant improvement (according to my tests). 

The model was trained and validated on different data sets to ensure that the model was not overfitting using a generator (code line 240). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model_track1.py line 239).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. For track 1, the most challenging part was to pass through the bridge.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was first to use simple architecture, and then improve it by adding more layers, trying different filter sizes, different number of fully-connected layers, making deeper convolutions and so on.

My first step was to use a convolution neural network model which has several convolution layers, followed by several fully connected layers. I thought this model might be appropriate because it worked really well for image classification.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model, having only 2 convolutions and 2 fully-connected layers can barely drive the car straight. Moreover, it overfits quite soon, i.e. the difference between training and validation accuracy is quite large.

This brought me to an idea that I should use more complicated models, so I added more layers. I had to perform a lot of experiments, trying different layers depth.

I finally randomly shuffled the data set and put 10-20% of the data into a validation set. 

I used the training data for training the model. The validation set helped determine if the model was over or under fitting.  I used an adam optimizer so that manually training the learning rate wasn't necessary.

####2. Final Model Architecture

I ended up with 32-48-64-96 depth and a maxpooling after each layer. As for fully connected layers, the main concept that I found useful is to set several fully connected layers, each following having smaller size (1536-128-16-1 in my model_track1.py).

####3. Creation of the Training Set & Training Process

At first, I used the dataset provided in the project resources. The final step was to run the simulator to see how well the car was driving around track one. However, it appeared that although I was able to achieve high accuracy, the model falls off the track at places where the yellow lines dissapear. Then I had to collect more data, and add it to what was provided in the project resources.

![alt text][image2]

This helped to avoid falling off the track, however, I encoutered another problem - after adding new data, the model stopped behaving well on the bridge, i.e. it bounced from one side to another many times, being unable to hit the road at the end of the bridge. The recovery data I have collected looked as following:

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I had to collect even more data only on the bridge, and this finally solved all problems. The model I got is now capable of driving both directions on the track, as I used random flipping of images and angles. Flipping appeared to be very important, as without it the model was able to perform well only left turns, because track 1 mainly has left turns. The flipped images looked like this:

![alt text][image6]
![alt text][image7]

Exept the image flipping, I used data from left and right cameras, with the angle changed by +-0.25. From my experiments, 0.25 is a little bit better value than recommended 0.2, as it adds more penalty for hitting the edge of the road. I have also added random shadows on the left or right sides of the image, i.e. the part of the image was made darker, so that the network would learn to rely only on half of the image to determine the steering angle (lines 61-69 in model_track1.py)

You may check my succeful compliting track 1 at [![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/a8uZswck93k/0.jpg)](https://youtu.be/a8uZswck93k)



