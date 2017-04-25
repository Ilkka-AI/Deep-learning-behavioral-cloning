#**Behavioral Cloning** 


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report



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

I used simply the Lenet architecture provided in the instructions. I got it working quite well. I also tried the Nvidia model, but at first glance it didn't work so well. My model structure is the following:

I used the lambda layer for normalization, cropping to remove unnecesary parts of the image. I used RELU layers to induce nonlinearity. The training images needed to be transformed from BGR to RGB.

####2. Attempts to reduce overfitting in the model

The model contains two maxpooling layers in order to reduce overfitting. I also noticed the only a few epochs were needed to avoid overfitting. Two epochs turned out to be the optimum, after that the validation error started increasing again. 

The model was trained and validated on different data sets to ensure that the model was not overfitting using the Keras validation_split. The model was tested by running it through the simulator and it runs great. 

I augmented the dataset by flipping the images to avoid directional bias. 


####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I drove two laps, and one in the opposite direction. I gathered extensively recovery examples, especially in the curves. This is probably the reason why such a simple model worked so well. Keyboard controls was sufficient, no need for joystick. 


###Model Architecture and Training Strategy

####1. Solution Design Approach

I started by experimenting. I compared the provided Lenet to the Nvidia model, also trying to use dropout in the latter. I also tried different numbers of epochs in training. It quickly started looking like Lenet was performing much better when seeing how the car was performing in autonomous model. I also noticed more than two epochs made the model perform worse. 

I also noticed that the training data was a more important factor than the model architecture. That's why I chose Lenet and focused on the training data gathering. In the end it worked.  


In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. In any case I noticed the training loss was decreasing with number of epochs, while the validation loss stopped decreasing after two epochs. Choosing two epochs always gave the best performance on the track. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, namely the sharp curves, especially the one were the ledge changed color after the bridge. I noticed extensive recovery training, focusing on the difficult curves, was the key to improve performance. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

My final model architecture was:

Cropping2D(cropping=((70,25), (0,0)),input_shape=(160,320,3))
Lambda(lambda x: (x / 255.0) - 0.5))    
Convolution2D(6, 5, 5,activation='relu')
MaxPooling2D((2, 2))
Convolution2D(6, 5, 5,activation='relu')
MaxPooling2D((2, 2))
Flatten()
Dense(120)
Dense(84)
Dense(1)



####3. Creation of the Training Set & Training Process

Explained above. 

After the collection process, I had 11000 data points. I then preprocessed this data by lambda normalizationa and image augmentation. 


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 as evidenced by validation error not decreasing or even increasing. I used an adam optimizer so that manually training the learning rate wasn't necessary.
