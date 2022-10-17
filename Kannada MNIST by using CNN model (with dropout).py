# Importing Basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix,accuracy_score
%matplotlib inline

# Extracting the data
train = pd.read_csv('../manzitlo/Kannada-MNIST/train.csv')
test = pd.read_csv('../manzitlo/Kannada-MNIST/test.csv')
dig_mnist = pd.read_csv('../manzitlo/Kannada-MNIST/Dig-MNIST.csv')
train.head()  # 5 rows * 785 columns
train.info()
print(train.shape)  # (60000, 785)
print(test.shape)   # (5000, 785)

# Checking whether data is balanced or not.
values = train['label'].value_counts()
sns.barplot(values.index,values)

# Checking the rows by specifying the row number.
features = ["pixel{}".format(pixel_num) for pixel_num in range (0,784)]# Stores the pixel array from pixel1 to pixel785
rows_to_examine = 5  # try 5
image_data = np.reshape(train[features][rows_to_examine:rows_to_examine + 1].to_numpy(),(28,28))
plt.imshow(image_data,cmap='gray')
plt.show()

# Preparing Data

x = train.drop('label',axis = 1)
y = train['label']
print(x.shape)  # (60000, 784)
print(y.shape)  # (60000,)

# Dividing Dataset into Test and Train

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=46)
print(x_train.shape)  #(45000, 784)
print(y_train.shape)  #(45000,)
print(x_test.shape)   #(15000,784)
print(y_test.shape)   #(15000,)

# Converting DataFrame to array
x_train = x_train.to_numpy()
x_test = x_test.to_numpy()

#Normalization

x_train = x_train.reshape(x_train.shape[0],28,28,1).astype('float64')/255
x_test = x_test.reshape(x_test.shape[0],28,28,1).astype('float64')/255
print(x_train.shape)  #(45000, 28, 28, 1)
print(x_test.shape)   #(15000, 28, 28, 1)

# Visualization
class_names = [0,1,2,3,4,5,6,7,8,9]
plt.figure(figsize = (15,15))
for k in range(10):
    xx = x_train[y_train == k]
    yy = y_train[y_train == k].reset_index()['label']
    for i in range(10):
        plt.subplot(10,10,k * 10 + i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(xx[i][:,:,0],cmap = 'gray')
        label_index = int(yy[i])
        plt.title('{}. {}'.format(k,class_names[label_index]))
plt.show()

# to_categorical is used to perform one-hot encoding for label values
from keras.utils import np_utils
y_train1 = np_utils.to_categorical(y_train,10)
y_test1 = np_utils.to_categorical(y_test,10)

# Data Augumentation

# create more sample images using ImageDataGenerator.This ensures we have more data to train on. 
from keras.preprocessing.image import ImageDataGenerator
imagegen = ImageDataGenerator(
        featurewise_center=False, 
        samplewise_center=False, 
        featurewise_std_normalization=False, 
        samplewise_std_normalization=False,
        zca_whitening=False, 
        rotation_range=9, 
        zoom_range = 0.25,
        width_shift_range=0.25,
        height_shift_range=0.25, 
        horizontal_flip=False, 
        vertical_flip=False)

imagegen.fit(x_train)

# importing DL packages 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten,Conv2D,MaxPooling2D,Dense,Dropout

## Building CNN Model (With dropout)!!
tf.random.set_seed(100)
tf.keras.backend.clear_session()
model = None
model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),input_shape=(28,28,1),activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))# First Hidden Layer with fully connected
model.add(Dense(64,activation='relu'))# First Hidden Layer with fully connected
model.add(Dense(10,activation='softmax')) # Output layer for 10 classification

#Training the model
model.compile(loss='categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
model.summary() # total params: 232,650; Trainable params: 232,650

history = model.fit_generator(imagegen.flow(x_train,y_train1,batch_size=128),epochs=30,validation_data=(x_test,y_test1),verbose = 1)

# Evaluation and Prediction

test_score = model.evaluate(x_test,y_test1)
train_score = model.evaluate(x_train,y_train1)
print('\n Train Accuracy : %4f' % (train_score[1]))  # 99.2267%
print('\n Test Accuracy : %4f' % (test_score[1]))    # 99.2667%

#Accuray and Error Graph

train_accuracy = history.history['accuracy']
train_loss = history.history['loss']

val_accuracy = history.history['val_accuracy']
val_loss = history.history['val_loss']

epochs = range(len(train_accuracy))
plt.plot(epochs,train_accuracy,'o-g',label='Train')
plt.plot(epochs,val_accuracy,'o-b',label='Validation/Test')
plt.title('Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
#plt.figure()
plt.show()

plt.plot(epochs,train_loss,'o-g',label='Train')
plt.plot(epochs,val_loss,'o-r',label = 'Validation/Test')
plt.title('Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

