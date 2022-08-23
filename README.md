### `Read and catch up on content:`
- [Gradient Descent article](https://github.com/MoinDalvs/Gradient_Descent_For_beginners/blob/main/README.md) :books:

## 0.1 Table of Contents<a class="anchor" id="0.1"></a>
1. [Let's talk about Neural Networks.](#1)
2. [Some Basic Concepts Related to Neural Networks](#2)
    - 2.1 [Different layers of a Neural Network](#2.1)
    - 2.2 [The Cost function](#2.2)
    - - 2.2 A) [The Difference between Loss and Cost Function](#2.2A)
    - 2.3 [Linear Regression using Gradient Descent](#2.3)
    - 2.4 [Initialization](#2.4)
    - 2.5 [Direction and learning Rate](#2.5)
    - 2.6 [Challenges with Gradient Descent](#2.6)
    - 2.7 [Types of Gradient Descent](#2.7)
    - 2.8 [Variants of Gradient Descent Algorithm](#2.8)
    - 2.9 [Overview](#2.9)

## 1. Let's talk about Neural Networks<a class="anchor" id="1"></a>

<img width="720" height='580' alt="" src="https://user-images.githubusercontent.com/99672298/186175978-c9737e54-f6a0-4f4a-924a-1b33061780fa.png">

Supervised learning can be used on both structured and unstructered data. For example of a structured data let's take house prices prediction data, let's also assume the given data tells us the size and the number of bedrooms. This is what is called a well structured data, where each features, such as the size of the house, the number of bedrooms, etc has a very well defined meaning.
	
In contrast, unstructured data refers to things like audio, raw audio, or images where you might want to recognize what's in the image or text (like object detection and OCR Optical character recognition). Here, the features might be the pixel values in an image or the individual words in a piece of text. It's not really clear what each pixel of the image represents and therefore this falls under the unstructured data.
	
Simple machine learning algorithms work well with structured data. But when it comes to unstructured data, their performance tends to take a quite dip. This where Neural Netowrks does their magic , which have proven to be so effective and useful. They perform exceptionally well on structured data.

<img width="720" height='580' alt="" src="https://user-images.githubusercontent.com/99672298/186084176-5830c906-fb90-4fb6-ba05-66496934f10b.png">

As the Amount of data icreases, the performance of data learning algorithms, like SVM and logistic regression, does not improve infacts it tends to plateau after a certain point. In the case of neural networks as the amount of data increases the performance of the model increases.

## 2 Some Basic Concepts Related to Neural Networks<a class="anchor" id="2"></a>
![15 08 2022_20 31 52_REC](https://user-images.githubusercontent.com/99672298/186165818-1c943321-7dd7-456a-9a1d-b39198fc8dc7.png)

### 2.1 Different layers of a Neural Network<a class="anchor" id="2.1"></a>

<img width="820" height='580' alt="627d1225cb1b3d197840427a_60f040a887535b932a3b2b6e_cnn-hero%20(1)" src="https://user-images.githubusercontent.com/99672298/186167806-5aef0ecd-88bb-46df-87b7-89c2e02854cf.png">

#### **`Input layers:-`** Also known as Input Nodes are the input information from the dataset the features $x_1$, $x_2$, ..... $x_n$ that is provided to the model to learn and derive conclusions from. Input nodes pass the information to the next layer i.e Hidden layers.

#### **`Hidden layers:-`** It is the set of neurons where all the computations are performed on the input data. There can be any number of hidden layers in a neural network. The simplest network consists of a single hidden layer.

This is the layer where complex computations happen. The more your model has hidden layers, the more complex the model will be. This is kind of like a black box of the neural network where the model learns complex relations present in the data.

#### **`Output Layer:-`** From the diagram given above ther is only on node in the ouput layer, but don't think that is always like that in every neural network model. The number of nodes in the output layer completely depends upon the problem that we have taken. If we have a binary classification problem then the output node is going to be 1 but in the case of multi-class classification, the output nodes can be more than 1.

#### Step by Step Working of the Artificial Neural Network

![image](https://user-images.githubusercontent.com/99672298/186180131-164203c5-dabd-466e-a6d9-b0874dd7edd0.png)
![image](https://user-images.githubusercontent.com/99672298/186179926-b40a240c-90aa-4cc8-a1e4-82b7392515c2.png)
![image](https://user-images.githubusercontent.com/99672298/186179961-02f7183d-85ea-41dd-bd5a-4bd3acfb39b6.png)

+ **1.) In the first step, Input units are passed i.e data is passes with some weights attached to it to the hidden layers. WE can have any number of hidden layers.**
+ **2.) Each hidden layers consists of neurons. All the inputs are connected to neuron (each).**
+ **3.) After passing on the inputs, all the the computation is performed in the hidden layers.**


