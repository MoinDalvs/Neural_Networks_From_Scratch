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

[Table of Content](#0.1)

## 2. Some Basic Concepts Related to Neural Networks<a class="anchor" id="2"></a>
![15 08 2022_20 31 52_REC](https://user-images.githubusercontent.com/99672298/186165818-1c943321-7dd7-456a-9a1d-b39198fc8dc7.png)

### 2.1 Different layers of a Neural Network<a class="anchor" id="2.1"></a>

<img width="820" height='580' alt="627d1225cb1b3d197840427a_60f040a887535b932a3b2b6e_cnn-hero%20(1)" src="https://user-images.githubusercontent.com/99672298/186167806-5aef0ecd-88bb-46df-87b7-89c2e02854cf.png">

#### **`Input layers:-`** Also known as Input Nodes are the input information from the dataset the features $x_1$, $x_2$, ..... $x_n$ that is provided to the model to learn and derive conclusions from. Input nodes pass the information to the next layer i.e Hidden layers.

#### **`Hidden layers:-`** It is the set of neurons where all the computations are performed on the input data. There can be any number of hidden layers in a neural network. The simplest network consists of a single hidden layer.

This is the layer where complex computations happen. The more your model has hidden layers, the more complex the model will be. This is kind of like a black box of the neural network where the model learns complex relations present in the data.

#### **`Output Layer:-`** From the diagram given above ther is only on node in the ouput layer, but don't think that is always like that in every neural network model. The number of nodes in the output layer completely depends upon the problem that we have taken. If we have a binary classification problem then the output node is going to be 1 but in the case of multi-class classification, the output nodes can be more than 1.

### 2.2 Step by Step Working of the Artificial Neural Network<a class="anchor" id="2.1"></a>

![image](https://user-images.githubusercontent.com/99672298/186180131-164203c5-dabd-466e-a6d9-b0874dd7edd0.png)
![image](https://user-images.githubusercontent.com/99672298/186179926-b40a240c-90aa-4cc8-a1e4-82b7392515c2.png)
![image](https://user-images.githubusercontent.com/99672298/186179961-02f7183d-85ea-41dd-bd5a-4bd3acfb39b6.png)

+ **1.) In the first step, Input units are passed i.e data is passes with some weights attached to it to the hidden layers. WE can have any number of hidden layers.**
+ **2.) Each hidden layers consists of neurons. All the inputs are connected to neuron (each).**
+ **3.) After passing on the inputs, all the the computation is performed in the hidden layers.**

#### Computation performed in hidden layers are done in two steps which are as follows:-

##### First of all, all the inputs are multiplied by their respective weights assigned. Weights are the gradient of each variable. It shows the strength of the particular input. After assigning the weights, a bias variable is added. Bias is coefficient of each variable and it is constant that helps the model to fit in the best way possible.

##### Then in the second step, the activation function is applied to the linear equation 'y'. The activation function is a non-linear transformation that is applied to the input before sending it to the next layer of neuron. The importance of the activation function is to incubate non-linearity i the model.

##### The whole process described in point 3 is performed in each hidden layers. After passing through every hidden layers we move to the last layer i.e our output layer which gives us the final output. 
##### **`This process explained above is known as forward Propogation.`**
##### After getting the predictions from the output layers, the error is calculated i.e the difference between the actual and the predicted output. If the error is large then steps are take to minimize the error and for the same purpose Back propogation is performed.

![12203Schematic-diagram-of-backpropagation-training-algorithm-and-typical-neuron-model_W640](https://user-images.githubusercontent.com/99672298/186192225-d825db1d-cbde-4176-9f65-566a24308904.jpg)
![21 08 2022_16 11 48_REC](https://user-images.githubusercontent.com/99672298/186192374-d465b7bb-b318-4525-834d-6021eb919209.png)
![21 08 2022_16 12 32_REC](https://user-images.githubusercontent.com/99672298/186192472-05b6502a-b717-4f17-b1b8-fb77e3105e82.png)
![21 08 2022_16 14 54_REC](https://user-images.githubusercontent.com/99672298/186192598-32234ca5-849b-4fe9-92bd-994f216a5817.png)
![21 08 2022_16 13 45_REC](https://user-images.githubusercontent.com/99672298/186192523-52532cfb-807f-48a0-9b9d-da4bdbed6c60.png)
![21 08 2022_16 25 14_REC](https://user-images.githubusercontent.com/99672298/186192689-98928ce5-a99a-4f13-9fd6-0c86590a5e93.png)
![21 08 2022_16 24 42_REC](https://user-images.githubusercontent.com/99672298/186192794-78b6992c-d389-49fd-b7d3-87c5557cf351.png)
![21 08 2022_16 25 58_REC](https://user-images.githubusercontent.com/99672298/186192721-6aedb39d-d42e-48b0-8254-2c202c03f4d1.png)
![21 08 2022_16 26 30_REC](https://user-images.githubusercontent.com/99672298/186193081-504eb1f6-739b-48ae-adb7-4d1d9c4df65a.png)

#### So long story short

Artificial Neural Network (ANN) are comprised of node layers. containing an input layer, one or more hidden layers, and a output layer. Each node or artificial neuron, connects to another and has an associated weigths and threshold.

If the output of any individual node is above the specified threshold value, that ndoe is activated, sending data to the next layer of the network otherwise, no data is passed along to the next layer of the network.

Once an input layer is determined, weight are assigned. These weights help determine the importance of any given variable, with larger ones contributing more significantly to the output compared to other inputs

All inputs are then multiplied by their respective weights and then summed. Afterwards, the output is passed through an activation function, which determines the output. If the ouput exceeds a given threshold, it 'fires' (or activates) the node, passing data to the next layer in the network. This results in the output of one node becoming in the input of the next node.

#### 

