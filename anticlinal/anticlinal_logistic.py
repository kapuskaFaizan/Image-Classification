import os
import cv2
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score



datadir ='/home/codersarts/Desktop/Anticlinal/Anticlinal/images/Train'
datadirtest ='/home/codersarts/Desktop/Anticlinal/Anticlinal/images/Test'
catagories = ['Class1','Class2']

img_size = 150
im_l =240
im_b =320
training_data = []

def create_data(datadir,catagories):
    for category in catagories:
        path = os.path.join(datadir,category)
        classnum = catagories.index(category)
        for img in os.listdir(path):
            img_arr = cv2.imread(os.path.join(path,img),0)
            print(img)
            if img_arr is not None:
            	new_arr = cv2.resize(img_arr,(img_size,img_size))
            	training_data.append([new_arr,classnum])
            
create_data(datadir,catagories)
print(len(training_data))
random.shuffle(training_data)

x=[]
y=[]
for feature , label in training_data:
    x.append(feature)
    y.append(label)
print(len(x),len(y))
X = np.array(x)

print(X.shape)

train_imgs_scaled = X.astype('float32')
train_imgs_scaled /= 255

x_train_flatten = train_imgs_scaled.reshape(len(train_imgs_scaled),train_imgs_scaled.shape[1]*train_imgs_scaled.shape[2])

clf = LogisticRegression()
clf.fit(x_train_flatten, y)

create_data(datadirtest,catagories)
print(len(training_data))
random.shuffle(training_data)

test_x=[]
test_y=[]

for feature , label in training_data:
    test_x.append(feature)
    test_y.append(label)
print(len(test_x),len(test_y))
tx = np.array(test_x)

print(X.shape)

test_imgs_scaled = tx.astype('float32')
test_imgs_scaled /= 255

x_test_flatten = test_imgs_scaled.reshape(len(test_imgs_scaled),test_imgs_scaled.shape[1]*test_imgs_scaled.shape[2])

y_hat = clf.predict(x_test_flatten)
print("Accuracy Score -> ",accuracy_score(test_y, y_hat)*100)
print(classification_report(test_y,y_hat))
print(y_hat)
