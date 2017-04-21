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

###Here is the [video](https://www.youtube.com/edit?o=U&video_id=fW1b2EHvz6c) showing the my successful lap 

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consisted of the following layers:
<br/>
A Lambda layer first normalizes the images
<br/>
The images are then cropped to eliminate excessive, unimportant pixels
<br/>
An activation layer with a 5X5 kernel
<br/>
A pooling layer
<br/>
another activation layer also with a 5X5 kernel
<br/>
another pooling layer
<br/>
A layer to flatten the accumulated layers
<br/>
3 FC layers with 120, 84 and 1 neurons, respectively
<br/>
Finally the NN is compiled with a Mean Square Error function and adam optimizer


####2. Attempts to reduce overfitting in the model


The model was trained and validated on different data sets to ensure that the model was not overfitting 
<br/>
The first data set contained data for just a single lap in manual mode at approximately 12MPH
<br/>
The second data set contained data for 10 laps in manual mode at 30MPH.
<br/>
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.
<br/>
Interestingly, the first data set proved to be better at keeping the car on the road

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

I experimented with 3 different networks and 2 data sets.
<br/>
The first network was the default network presented in the training materials
<br/>
The second network was the [NVIDIA network](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) referenced in the training materials.
<br/> The third network was one of my own creation and we will not speak of it further.

The step that seems to have helped my model the most was NOT adding images with a measurement of 0 to the array of images.  After this step was implemented the model succeeded.

####2. Final Model Architecture

The final model architecture consisted of a convolution neural network that closely resembled the network presented in the training material



####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one lap on track one using center lane driving. 

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to self correct

To augment the data sat, I also flipped images and angles thinking that this would offset the left turn bias of the track. I added or subtracted 0.05 from the measurement
for left and right camera views.


After the collection process, I had 2,134 number of data points. 

I finally randomly shuffled the data set and put 20 of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3.
