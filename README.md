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

From my point of view, the overfitting is not a big issue for this problem, because basically underfitting is what I have expirienced more (especially for track 2). However, overfitting may occur, if the model is trained too long, 400+ epochs.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 239).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. For track 1, the most challenging part was to pass through the bridge.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was first to use AlexNet-like architecture, and then improve it by adding more layers, trying different filter sizes, different number of fully-connected layers, making deeper convolutions and so on.

My first step was to use a convolution neural network model similar to the AlexNet. I thought this model might be appropriate because it worked really well for image classification.

Next, I've tried to use VGG-like architecture, which appeared to be more powerfull.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
