import os
import csv
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt

samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    counter = 0
    for line in reader:
        if counter == 0:
            counter = counter + 1
            continue
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.1)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=16):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_name = 'data/IMG/'+batch_sample[0].split('/')[-1]
                left_name = 'data/IMG/'+batch_sample[1].split('/')[-1]
                right_name = 'data/IMG/'+batch_sample[2].split('/')[-1]
                center_image = cv2.imread(center_name)
                left_image = cv2.imread(left_name)
                right_image = cv2.imread(right_name)
                center_angle = float(batch_sample[3])
                left_angle = float(batch_sample[3])+0.25
                right_angle = float(batch_sample[3])-0.25

                hsv_center = cv2.cvtColor(center_image, cv2.COLOR_BGR2HSV)
                hsv_center[:,:,2] *= int(np.random.uniform(0.1,1.5))
                hsv_center = np.clip(hsv_center,0,100)
                center_image2 = cv2.cvtColor(hsv_center, cv2.COLOR_HSV2BGR)

                hsv_center = cv2.cvtColor(center_image, cv2.COLOR_BGR2HSV)
                hsv_center[:,:,1] *= int(np.random.uniform(0.1,1.5))
                hsv_center = np.clip(hsv_center,0,100)
                center_image3 = cv2.cvtColor(hsv_center, cv2.COLOR_HSV2BGR)

                hsv_center = cv2.cvtColor(center_image, cv2.COLOR_BGR2HSV)
                hsv_center[:,:,0] *= int(np.random.uniform(0.1,1.5))
                hsv_center = np.clip(hsv_center,0,100)
                center_image4 = cv2.cvtColor(hsv_center, cv2.COLOR_HSV2BGR)
                
                hsv_center = cv2.cvtColor(center_image, cv2.COLOR_BGR2HSV)
                hsv_center[:,0:170,2] -= 30
                hsv_center = np.clip(hsv_center,0,100)
                center_image5 = cv2.cvtColor(hsv_center, cv2.COLOR_HSV2BGR)

                hsv_center = cv2.cvtColor(center_image, cv2.COLOR_BGR2HSV)
                hsv_center[:,150:320,2] -= 30
                hsv_center = np.clip(hsv_center,0,100)
                center_image6 = cv2.cvtColor(hsv_center, cv2.COLOR_HSV2BGR)

                hsv_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2HSV)
                hsv_left[:,:,2] *= int(np.random.uniform(0.1,1.5))
                hsv_left = np.clip(hsv_left,0,100)
                left_image2 = cv2.cvtColor(hsv_left, cv2.COLOR_HSV2BGR)

                hsv_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2HSV)
                hsv_left[:,:,1] *= int(np.random.uniform(0.1,1.5))
                hsv_left = np.clip(hsv_left,0,100)
                left_image3 = cv2.cvtColor(hsv_left, cv2.COLOR_HSV2BGR)

                hsv_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2HSV)
                hsv_left[:,:,0] *= int(np.random.uniform(0.1,1.5))
                hsv_left = np.clip(hsv_left,0,100)
                left_image4 = cv2.cvtColor(hsv_left, cv2.COLOR_HSV2BGR)

                hsv_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2HSV)
                hsv_left[:,0:170,2] -= 30
                hsv_left = np.clip(hsv_left,0,100)
                left_image5 = cv2.cvtColor(hsv_left, cv2.COLOR_HSV2BGR)

                hsv_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2HSV)
                hsv_left[:,150:320,2] -= 30
                hsv_left = np.clip(hsv_left,0,100)
                left_image6 = cv2.cvtColor(hsv_left, cv2.COLOR_HSV2BGR)

                hsv_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2HSV)
                hsv_right[:,:,2] *= int(np.random.uniform(0.1,1.5))
                hsv_right = np.clip(hsv_right,0,100)
                right_image2 = cv2.cvtColor(hsv_right, cv2.COLOR_HSV2BGR)

                hsv_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2HSV)
                hsv_right[:,:,1] *= int(np.random.uniform(0.1,1.5))
                hsv_right = np.clip(hsv_right,0,100)
                right_image3 = cv2.cvtColor(hsv_right, cv2.COLOR_HSV2BGR)

                hsv_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2HSV)
                hsv_right[:,:,0] *= int(np.random.uniform(0.1,1.5))
                hsv_right = np.clip(hsv_right,0,100)
                right_image4 = cv2.cvtColor(hsv_right, cv2.COLOR_HSV2BGR)

                hsv_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2HSV)
                hsv_right[:,0:150,2] -= 30
                hsv_right = np.clip(hsv_right,0,100)
                right_image5 = cv2.cvtColor(hsv_right, cv2.COLOR_HSV2BGR)

                hsv_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2HSV)
                hsv_right[:,170:320,2] -= 30
                hsv_right = np.clip(hsv_right,0,100)
                right_image6 = cv2.cvtColor(hsv_right, cv2.COLOR_HSV2BGR)

                images.append(center_image)
                angles.append(center_angle)
                images.append(left_image)
                angles.append(left_angle)
                images.append(right_image)
                angles.append(right_angle)

                #images.append(center_image2)
                #angles.append(center_angle)
                #images.append(left_image2)
                #angles.append(left_angle)
                #images.append(right_image2)
                #angles.append(right_angle)

                #images.append(center_image3)
                #angles.append(center_angle)
                #images.append(left_image3)
                #angles.append(left_angle)
                #images.append(right_image3)
                #angles.append(right_angle)

                #images.append(center_image4)
                #angles.append(center_angle)
                #images.append(left_image4)
                #angles.append(left_angle)
                #images.append(right_image4)
                #angles.append(right_angle)

                images.append(center_image5)
                angles.append(center_angle)
                images.append(left_image5)
                angles.append(left_angle)
                images.append(right_image5)
                angles.append(right_angle)

                images.append(center_image6)
                angles.append(center_angle)
                images.append(left_image6)
                angles.append(left_angle)
                images.append(right_image6)
                angles.append(right_angle)

                images.append(np.fliplr(center_image))
                angles.append(-center_angle)
                images.append(np.fliplr(left_image))
                angles.append(-left_angle)
                images.append(np.fliplr(right_image))
                angles.append(-right_angle)

                #images.append(np.fliplr(center_image2))
                #angles.append(-center_angle)
                #images.append(np.fliplr(left_image2))
                #angles.append(-left_angle)
                #images.append(np.fliplr(right_image2))
                #angles.append(-right_angle)

                #images.append(np.fliplr(center_image3))
                #angles.append(-center_angle)
                #images.append(np.fliplr(left_image3))
                #angles.append(-left_angle)
                #images.append(np.fliplr(right_image3))
                #angles.append(-right_angle)

                #images.append(np.fliplr(center_image4))
                #angles.append(-center_angle)
                #images.append(np.fliplr(left_image4))
                #angles.append(-left_angle)
                #images.append(np.fliplr(right_image4))
                #angles.append(-right_angle)

                images.append(np.fliplr(center_image5))
                angles.append(-center_angle)
                images.append(np.fliplr(left_image5))
                angles.append(-left_angle)
                images.append(np.fliplr(right_image5))
                angles.append(-right_angle)

                images.append(np.fliplr(center_image6))
                angles.append(-center_angle)
                images.append(np.fliplr(left_image6))
                angles.append(-left_angle)
                images.append(np.fliplr(right_image6))
                angles.append(-right_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=16)
validation_generator = generator(validation_samples, batch_size=16)

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
model.add(MaxPooling2D((2,2), strides=(3,3)))
model.add(Convolution2D(96, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(3,3)))

model.add(Flatten())
model.add(Dense(1536))
model.add(Dense(128))
model.add(Dense(16))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, steps_per_epoch = 50, validation_data=validation_generator, validation_steps = 59, epochs=20, verbose = 1)

print(history_object.history.keys())

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model.h5')
