import matplotlib.pyplot as plt
import matplotlib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten,Conv2D
import numpy as np
from skimage.draw import circle_perimeter_aa
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

size = 100 
bunch = 25000 
max_rad = 50

def noisy_circle(size, radius, noise):
    img = np.zeros((size, size), dtype=np.float)

    # Circle
    row = np.random.randint(size)
    col = np.random.randint(size)
    rad = np.random.randint(10, max(10, radius))
    draw_circle(img, row, col, rad)

    # Noise
    img += noise * np.random.rand(*img.shape)
    return (row, col, rad), img
def draw_circle(img, row, col, rad):
    rr, cc, val = circle_perimeter_aa(row, col, rad)
    valid = (
        (rr >= 0) &
        (rr < img.shape[0]) &
        (cc >= 0) &
        (cc < img.shape[1])
    )
    img[rr[valid], cc[valid]] = val[valid]


images = np.zeros((bunch, size, size), dtype=np.float)
circles = np.zeros((bunch, 3)) 
for i in range(bunch):
    circles[i], images[i] = noisy_circle(size, max_rad, 2)

y = circles.reshape(bunch, -1)# / size
print(f"Reshaped circles' parameters: {y.shape}, {np.mean(y)}, {np.std(y)}\n",type(y),len(y))
train = images.reshape(-1, size, size, 1)


def find_circles(input_shape,batch_size,datax,datay):  
  model = Sequential()
  model.add(Conv2D(32, (3, 3), input_shape=input_shape))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(32, (3, 3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(64, (3, 3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Flatten()) 
  model.add(Dense(64))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(3))
  model.add(Activation('softmax'))

  model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics=['accuracy'])
  history =model.fit(train, y,batch_size= batch_size, epochs=15, verbose=1, validation_split=0.2)
  return history , model

batch_size = 200
input_shape = train.shape[1:]

his , mod = find_circles(input_shape,batch_size,train,y)  
