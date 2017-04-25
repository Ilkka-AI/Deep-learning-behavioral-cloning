import csv
import cv2
import numpy as np



# Load data
def bringfile(mesfile,imagefolder):
    
    lines=[]
    with open(mesfile) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line) 
    images=[]
    measurements=[]
    for line in lines[1:-1]:
        source_path=line[0] 
        #print(source_path)
        filename=source_path.split('\\')[-1]
        #print(filename)
        #current_path=imagefolder+filename
        current_path=source_path
        #print(current_path)
        image=cv2.imread(current_path)
        # change to RGB for consistency
        im3=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(im3)
        measurement=float(line[3])
        
        measurements.append(measurement)
    
    mes=measurements
    ims=images
    return mes,ims

# Training data was recorded in several batches. Load all of them and combine.

mesfile='mydata/mydrive1/driving_log.csv'
imagefolder='mydata/mydrive1/IMG/'
[mess,imss]=bringfile(mesfile,imagefolder)

mesfile2='mydata/mydrive2/driving_log.csv'
imagefolder2='mydata/mydrive2/IMG/'
[mess2,imss2]=bringfile(mesfile2,imagefolder2)

mesfile3='mydata/mydrive3/driving_log.csv'
imagefolder3='mydata/mydrive3/IMG/'
[mess3,imss3]=bringfile(mesfile3,imagefolder3)

mesfile4='mydata/mydrive4/driving_log.csv'
imagefolder4='mydata/mydrive4/IMG/'
[mess4,imss4]=bringfile(mesfile4,imagefolder4)

mesfile5='mydata/mydrive5/driving_log.csv'
imagefolder5='mydata/mydrive5/IMG/'
[mess5,imss5]=bringfile(mesfile5,imagefolder5)

mesfile6='mydata/mydrive6/driving_log.csv'
imagefolder6='mydata/mydrive6/IMG/'
[mess6,imss6]=bringfile(mesfile6,imagefolder6)


mesfile7='mydata/mydrive7/driving_log.csv'
imagefolder7='mydata/mydrive7/IMG/'
[mess7,imss7]=bringfile(mesfile7,imagefolder7)

mesfile8='mydata/mydrive8/driving_log.csv'
imagefolder8='mydata/mydrive8/IMG/'
[mess8,imss8]=bringfile(mesfile8,imagefolder8)

# track2
mesfile9='mydata/mydrive9/driving_log.csv'
imagefolder9='mydata/mydrive9/IMG/'
#[mess9,imss9]=bringfile(mesfile9,imagefolder9)

mesfile10='mydata/mydrive10/driving_log.csv'
imagefolder10='mydata/mydrive10/IMG/'
[mess10,imss10]=bringfile(mesfile10,imagefolder10)

# combined
images=imss+imss2+imss3+imss4+imss5+imss6+imss7+imss8+imss10
print(len(imss))
print(len(mess))
print(len(imss2))
print(len(mess2))

print(len(imss3))
print(len(mess3))

print(len(imss4))
print(len(mess4))

print(len(imss5))
print(len(mess5))

print(len(imss6))
print(len(mess6))

print(len(imss7))
print(len(mess7))

print(len(imss8))
print(len(mess8))

print(len(imss10))

print(len(mess10))
# delete to save memory
del imss
del imss2
del imss3
del imss4
del imss5
del imss6
del imss7
del imss8
del imss10
# combine driving angle measurements
measurements=mess+mess2+mess3+mess4+mess5+mess6+mess7+mess8+mess10
print(len(images))
print(len(measurements))

# change to numpy
X_train=np.array(images)
y_train=np.array(measurements)
print(X_train[1].shape)
print(np.fliplr(X_train[1]).shape)
print(X_train.shape)
flipped_images2=np.zeros(shape=X_train.shape,dtype=float)

for ii in range(0,X_train.shape[0]):
    flipped_images2[ii,:,:,:]=cv2.flip(X_train[ii,:,:,:], flipCode=1)

import matplotlib.image as mpimg

# opposite sign for flipped steering angles
measurements_flipped=np.multiply(measurements,(-1))

# Concatenate data with flipped data
X_train=np.concatenate((np.array(images),flipped_images2),axis=0)
y_train=np.concatenate((measurements,measurements_flipped),axis=0)
print(X_train.shape)
print(y_train.shape)

# Run the Keras model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, MaxPooling2D, Convolution2D, Cropping2D,Dropout

model=Sequential()
def lenet():
    model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    model.add(Convolution2D(6, 5, 5,activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Convolution2D(6, 5, 5,activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))

def nvidia():
    model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    model.add(Convolution2D(24, 5, 5,subsample=(2,2),activation='relu'))
    
    model.add(Convolution2D(36, 5, 5,subsample=(2,2),activation='relu'))
    
    model.add(Convolution2D(48, 5, 5,subsample=(2,2),activation='relu'))
    
    model.add(Convolution2D(64, 3, 3,activation='relu'))
    
    model.add(Convolution2D(64, 3, 3,activation='relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.25))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

# implement the model, lenet worked better   
lenet()

model.compile(loss='mse',optimizer='adam')
history_object = model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=2,verbose=1)
model.save('model.h5')
