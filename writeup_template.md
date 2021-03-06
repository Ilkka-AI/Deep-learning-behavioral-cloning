#**Behavioral Cloning** 


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report




#### Model Architecture and Training Strategy

I used simply the Lenet architecture provided in the instructions. I got it working quite well. I also tried the Nvidia model, but at first glance it didn't work so well. 

I used the lambda layer for normalization, cropping to remove unnecesary parts of the image. I used RELU layers to induce nonlinearity. The training images needed to be transformed from BGR to RGB.

Attempts to reduce overfitting in the model

A major choise for avoiding overfitting is to try to choose as simple model as possible. That is why Lenet is a great choice compared to the Nvidia model, especially as it seems to be sufficient. Also, the model contains two maxpooling layers in order to reduce overfitting. I also noticed that only a few epochs were needed in training the model to avoid overfitting. Two epochs turned out to be the optimum, after that the validation error started increasing again or stayed constant. 

The model was trained and validated on different data sets to ensure that the model was not overfitting using the Keras validation_split. The model was tested by running it through the simulator and it runs great. 




#### Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####  Appropriate training data

Training data was generated to keep the vehicle driving on the road. I drove two laps, plus one in the opposite direction. I then put extra emphasis on driving the curves, in both directions. Furthermore, I gathered extensively recovery examples, especially in the curves. Three example images are included, one from the center of the road and two recovery images. This is probably the reason why such a simple model worked so well. Keyboard controls were sufficient, no need for joystick. 

I augmented the dataset by flipping the images to avoid directional bias. 


### Model Architecture and Training Strategy

####  Solution Design Approach

I started by experimenting. I compared the provided Lenet to the Nvidia model, also trying to use dropout in the latter. I also tried different numbers of epochs in training. It quickly started looking like Lenet was performing much better when checking how the car was performing in autonomous mode. I also noticed more than two epochs made the model perform worse in the the autonomous mode. 

I also noticed that the training data was a more important factor than the model architecture. That's why I chose Lenet and focused on the training data gathering. In the end it worked.  


In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. In all experiments I noticed the training loss was decreasing with number of epochs, while the validation loss stopped decreasing after two epochs. Choosing two epochs also always gave the best performance in the the autonomous mode driving. 

During experimenting, I ran the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, namely the sharp curves, especially the one were the ledge changed color after the bridge. I noticed extensive recovery training, focusing on the difficult curves, was the key to improve performance. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####  Final Model Architecture

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



####  Creation of the Training Set & Training Process

Explained above. 

After the collection process, I had 11000 data points. This was double by augmenting the dataset by image flipping. I then preprocessed the data by lambda layer normalization. The data just fit into the memory so no generators were needed. Two epochs ran fast enough so I did not need AWS GPU, although I tested that as well.   

In the experimentation I randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 as evidenced by validation error not decreasing or even increasing. I used an adam optimizer so that manually training the learning rate wasn't necessary.


#### Track 2

I gathered another training data set of 5500 samples from track 2 and taught a similar model on it, now using 3 epochs as it was the optimum. The Lenet model works really well on track 2 as well. I have included a video and .h5 file from track 2. 



