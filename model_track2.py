import os
import csv
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt
from random import random
from random import randint
from keras.callbacks import ModelCheckpoint

samples = []
with open('Data_track2_1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.01)

import cv2
import numpy as np
import sklearn

def generator_train(samples, batch_size=16):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
		
            for batch_sample in batch_samples:
                imagetotake = randint(0,2)
                image_name = 'Data_track2_1/IMG/'+batch_sample[imagetotake].split('/')[-1]
                image = cv2.imread(image_name)
                angle = float(batch_sample[3])
                if(imagetotake == 1):
                    angle += 0.20
                if(imagetotake == 2):
                    angle -= 0.20
                    
                if(randint(0,1) == 1):
                    gammavalue = 0.5 + random() * 1.0
                    invertedGamma = 1.0 / gammavalue
                    table = np.array([((i / 255.0) ** invertedGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                    image = cv2.LUT(image, table)
                    
                if(randint(0,1) == 1):
                    brightness_random = 0.5 + random()*1.0
                    image = np.array(image)
                    image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
                    image[:,:,2] = image[:,:,2]*brightness_random
                    image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)
                    
                if(randint(0,1) == 1):
                    saturation_random = 0.5 + 1.0*random()
                    image = np.array(image)
                    image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
                    image[:,:,1] = image[:,:,1]*saturation_random
                    image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)
                
                if(randint(0,1) == 1):
                    lightness_random = 0.5 + 1.0*random()
                    image = np.array(image)
                    image = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
                    image[:,:,1] = image[:,:,1]*lightness_random
                    image = cv2.cvtColor(image,cv2.COLOR_HLS2RGB)
                    
                if(randint(0,1) == 1):    
                    image = np.array(image)
                    image_rows,image_cols,image_rgb = image.shape
                    steer_angle = -0.5
                    random_for_x = random()
                    translate_y_random = -5 + random()*10
                    translate_x_random = -30 + random_for_x*60
                    temp_M = np.float32([[1,0,translate_x_random],[0,1,translate_y_random]])
                    image = cv2.warpAffine(image,temp_M,(image_cols,image_rows))
                    angle = (steer_angle+(random_for_x-0.5)*0.2)
                 
                if(randint(0,1) == 1): 
                    angle = -angle
                    image = np.array(image)
                    image = cv2.flip(image,1)
                    
                if(randint(0,0) == 0):      
                    image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
                    max_x = 160
                    max_y = 320
                    if(randint(0,1) == 0):
                        index1 = (0,0)
                        index2 = (0,max_y)
                        index3 = (random()*max_x,max_y)
                        index4 = (random()*max_x,0)
                    else:
                        index1 = (random()*max_x,0)
                        index2 = (random()*max_x,max_y)
                        index3 = (max_x,max_y)
                        index4 = (max_x,0)

                    random_vertices = np.array([[index1,index2,index3,index4]], dtype = np.int32)         
                    random_brightness = 0.5 + random()*1.0
                    image_mask = np.zeros_like(image)
                    ignore_mask = [0,0,255]
                    cv2.fillPoly(image_mask, random_vertices, ignore_mask)
                    indices255 = image_mask[:,:,2] == 255
                    image[:,:,2][indices255] = image[:,:,2][indices255]*random_brightness
                    image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)

                images.append(image)
                angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
            
def generator_valid(samples, batch_size=16):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                imagetotake = randint(0,2)
                image_name = 'Data_track2_1/IMG/'+batch_sample[imagetotake].split('/')[-1]
                image = cv2.imread(image_name)
                angle = float(batch_sample[3])
                if(imagetotake == 1):
                    angle += 0.20
                if(imagetotake == 2):
                    angle -= 0.20

                images.append(image)
                angles.append(angle)

            X_valid = np.array(images)
            y_valid = np.array(angles)
            yield sklearn.utils.shuffle(X_valid, y_valid)


train_generator = generator_train(train_samples, batch_size=256)
validation_generator = generator_valid(validation_samples, batch_size=256)

print(len(train_samples))

ch, row, col = 3, 160, 320  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(row, col, ch)))
model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(160,320,3)))

model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Convolution2D(48, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Convolution2D(96, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Convolution2D(128, (3, 3), activation='relu'))

model.add(Flatten())

model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

weight_save_callback = ModelCheckpoint('model.h5', monitor='val_loss', verbose=0, save_best_only=False, mode='auto')
#model.fit(X_train,y_train,batch_size=batch_size,nb_epoch=nb_epoch,callbacks=[weight_save_callback])

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, steps_per_epoch = 100, validation_data=validation_generator, validation_steps = 15, epochs=500, verbose = 1,callbacks=[weight_save_callback])

print(history_object.history.keys())

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model.h5')
