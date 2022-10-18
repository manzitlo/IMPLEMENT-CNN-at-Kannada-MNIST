# IMPLEMENT CNN at Kannada MNIST

Convolutional neural network -- CNN is good at picture processing. It's inspired by the human visual nervous system.

CNN has two major characteristics:
1. Can effectively reduce the dimensionality of large data into small data;
2. Can effectively retain the characteristics of the picture, in line with the principles of image processing. At present, CNN has been widely used.

# Why Kannda MNIST?
Because convolutional neural networks have been applied to MNIST handwritten digit datasets for various experiments, I use the Kannada hand-written dataset . Kannada is the official administrative language of Karnataka, India, and is spoken by nearly 60 million people worldwide. Different symbols are used to represent the numbers 0-9 in the language, which are different from the modern Arabic numerals popular in many parts of the world today.

<img src="https://pic1.zhimg.com/v2-2ea3b454dacb13f077d497daa54fb550_720w.jpg?source=172ae18b" width=100% />

Like the above picture shows, "1" in Kannada hand writting more seems like "0", "3" and "7" in Kannada is more like "2". Unlike the MNIST handwritten digit dataset, Kannada's replacement of MNIST adds a new challenge to the CNN experiment.

Many machine learning problems (especially CNNS applied to image processing) consist of hundreds or thousands of features. 
The main problems with having so many characteristics are:

* It slows down the training process
* It is difficult to find a cost-effective and efficient solution


Regarding the application of CNN in Kannada handwritten data processing, I have achieved **TWO GOALS:**


## **1. Use dimensionality reduction tools to reduce and visualize hyperdimensional datasets. Getting down to two or three features helps us visualize the data, which is an important part of data analysis; The differences between PCA and t-SNE were also compared (in folder 1)**

<img src="https://github.com/manzitlo/IMPLEMENT-CNN-at-Kannada-MNIST/blob/main/images/Visualization%20by%20using%20PCA.png" width="400px"/>   <img src="https://github.com/manzitlo/IMPLEMENT-CNN-at-Kannada-MNIST/blob/main/images/using%20t-SNE%20to%20visualize.png" width="350px"/>

Through comparative analysis, I learned that the visualization produced by PCA does not distinguish all numbers well. This is because PCA is a linear projection, which means that it cannot capture nonlinear dependencies. 

**t-SNE or T-distributed random neighborhood embedding reduces high-dimensional datasets to low-dimensional graphs that retain a large amount of original information. It does this by giving each data point a location on a two-dimensional or three-dimensional map. This technique looks for clusters in the data to ensure that the embedding preserves meaning in the data. t-SNE reduces the dimensionality while trying to keep similar instances close and different instances separate.**

We can also implement 3D scatter plot by using plotly like the following screenshot shows:



## **2. Establish the CNN model (including dropout), analyze the impact of epochs on accuracy, and ensure that the accuracy is more than 98%**

*Using '/train.csv', '/test.csv', and '/Dig-MNIST.csv' to read the data. Checking the shape of train and test, train:(60000, 785); test: (5000, 785)*

    print(train.shape)
    print(test.shape)
    
To build CNN model, we have to import DL package:

    # importing DL packages
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Flatten,Conv2D,MaxPooling2D,Dense,Dropout

The following screenshot shows the layer strcture and total params (**232,650**)

<img src="https://github.com/manzitlo/IMPLEMENT-CNN-at-Kannada-MNIST/blob/main/images/Layer%20structure.png" width="350px" />

- [x] Setting epoch=30 (running 30 times). The result of accuracy is **Train Accuracy : 0.992267; Test Accuracy : 0.992667**

Like the following picture shows:

<div align=center>
<img src="https://github.com/manzitlo/IMPLEMENT-CNN-at-Kannada-MNIST/blob/main/images/The%20result%20of%20accuracy%20and%20loss.png" width="400px" />
</div>

**Yes, CNN is better for image processing**
Through the use of CNN model for Kannada MNIST data sets at this project,

I fully realized the advantages of CNN:

- The input image is in good agreement with the topology structure of the network;

- Excellent performance despite using fewer parameters;

- It avoids explicit feature extraction and learns implicitly from the training data;

- Feature extraction and pattern classification are carried out at the same time and generated in training at the same time, so the network can learn in parallel;

- Weight sharing reduces the training parameters of the network, reduces the complexity of the network structure, and has stronger applicability;

- There is no need to manually select features, and the weight is trained well to obtain features, and the classification effect is good;

- It can be directly input into the network, avoiding the complexity of data reconstruction in the process of feature extraction and classification.

<div align=center>
<img src="https://github.com/manzitlo/IMPLEMENT-CNN-at-Kannada-MNIST/blob/main/images/CNN.png" width="800px"/>
</div>
