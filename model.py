import csv
import cv2
import numpy as np
from pathlib import Path

lines = []
with open('driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []
for line in lines:
	if float(line[3]) != 0: #Don't import data without steering info
		for i in range(3): #Append all three camera angles
			source_path = line[i]
			filename = source_path.split('\\')[-1]
			current_path = './data/IMG/' + filename
			image = cv2.imread(current_path)
			images.append(image)

			if i==0: #CENTER CAMERA
				measurement = float(line[3])
			elif i==1: #LEFT CAMERA
				measurement = float(line[3]) + 0.05
			elif i==2: #RIGHT CAMERA
				measurement = float(line[3]) - 0.05
			
			measurements.append(measurement)


augmented_images, augmented_measurements = [],[]
for image, measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	augmented_images.append(cv2.flip(image, 1))
	augmented_measurements.append(measurement*-1.0)


X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D


model = Sequential()
#Add a layer for normalization
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
#crop the images per David's instructions
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
#using default 'adam' optimizer
model.compile(loss='mse', optimizer='adam')
# To reduce overfitting - we only need 3 Epochs
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)
model.save('model.h5')


