### `Read and catch up on content:`
- [Gradient Descent article](https://github.com/MoinDalvs/Gradient_Descent_For_beginners/blob/main/README.md) :books:

## 0.1 Table of Contents<a class="anchor" id="0.1"></a>
1. [Let's talk about Neural Networks.](#1)
2. [Some Basic Concepts Related to Neural Networks](#2)
	- 2.1 [Different layers of a Neural Network](#2.1)
3. [Weight and Bias initialization](#3)
	- 3.1 [Why Weight Initialization?](#3.1)
	- 3.2 [Zero initialization](#3.2)
	- 3.3 [Random initialization (Initialized weights randomly)](#3.3)
	- 3.4 [Weight Initialization for ReLU](#3.4)
	- 3.5 [Weight Initialization for Sigmoid and Tanh](#3.5)
	- 3.6 [Best Practices for Weight Initialization](#3.6)
4. [Activation Functions](#4)
 	- 4.1 [Why do we need Activation Function in Neural Network?](#4.1)
 	- 4.2 [Basic Types of Neural Network Activation Function](#4.2)
 	- 4.3 [Non-Linear Neural Networks Activation Functions](#4.3)
 	- 4.4 [How to choose the right Activation Function?](#4.4)
5. [Why are deep neural networks hard to train?](#5)
	- 5.1 [Vanishing Gradient Descent](#5.1)
	- 5.2 [Exploding Gradient Descent](#5.2)
	- 5.3 [Why do the gradients even vanish/explode?](#5.3)
	- 5.4 [How to know if our model is suffering from the Exploding/Vanishing gradient problem?](#5.4)
6. [How to avoid Overfitting of Neural Networks?](#6)
	- 6.1 [What is Regularization?](#6.1)
	- 6.2 [What is Dropout?](#6.2)
	- 6.3 [What is Neural Network Pruning?](#6.3)
	- 6.4 [Early stopping](#6.4)
8. [Step by Step Working of the Artificial Neural Network](#7)
   
   
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

#### **`Backpropoation:`**
It is the process of updating and finding the optimal values of weights or ceofficients which helps the model to minimize the error (loss function). The weights are updated with the help of optimizers we talked about in [Gradient Descent article](https://github.com/MoinDalvs/Gradient_Descent_For_beginners/blob/main/README.md). The weights of the network connections are repeatedly adjusted to minimize the difference between tha actual and the observed values. It aims to minimize the cost function by adjusting the network weights and biases. The cost funciton gradient determine the level of adjustment with respect to parameters like activation funciton , weights, biases etc.

![12203Schematic-diagram-of-backpropagation-training-algorithm-and-typical-neuron-model_W640](https://user-images.githubusercontent.com/99672298/186192225-d825db1d-cbde-4176-9f65-566a24308904.jpg)

+ **`After propagating the input features forward to the output layer through the various hidden layers consisting of different/same activation functions, we come up with a predicted probability of a sample belonging to the positive class ( generally, for classification tasks).`**
+ **`Now, the backpropagation algorithm propagates backward from the output layer to the input layer calculating the error gradients on the way.`**
+ **`Once the computation for gradients of the cost function w.r.t each parameter (weights and biases) in the neural network is done, the algorithm takes a gradient descent step towards the minimum to update the value of each parameter in the network using these gradients.`**

![21 08 2022_16 11 48_REC](https://user-images.githubusercontent.com/99672298/186192374-d465b7bb-b318-4525-834d-6021eb919209.png)
![21 08 2022_16 12 32_REC](https://user-images.githubusercontent.com/99672298/186192472-05b6502a-b717-4f17-b1b8-fb77e3105e82.png)
![21 08 2022_16 14 54_REC](https://user-images.githubusercontent.com/99672298/186192598-32234ca5-849b-4fe9-92bd-994f216a5817.png)
![21 08 2022_16 13 45_REC](https://user-images.githubusercontent.com/99672298/186192523-52532cfb-807f-48a0-9b9d-da4bdbed6c60.png)
![21 08 2022_16 25 14_REC](https://user-images.githubusercontent.com/99672298/186192689-98928ce5-a99a-4f13-9fd6-0c86590a5e93.png)
![21 08 2022_16 24 42_REC](https://user-images.githubusercontent.com/99672298/186192794-78b6992c-d389-49fd-b7d3-87c5557cf351.png)
![21 08 2022_16 25 58_REC](https://user-images.githubusercontent.com/99672298/186192721-6aedb39d-d42e-48b0-8254-2c202c03f4d1.png)
![21 08 2022_16 26 30_REC](https://user-images.githubusercontent.com/99672298/186193081-504eb1f6-739b-48ae-adb7-4d1d9c4df65a.png)
___
#### **`Output Layer:-`** From the diagram given above ther is only on node in the ouput layer, but don't think that is always like that in every neural network model. The number of nodes in the output layer completely depends upon the problem that we have taken. If we have a binary classification problem then the output node is going to be 1 but in the case of multi-class classification, the output nodes can be more than 1.

![image](https://user-images.githubusercontent.com/99672298/186179926-b40a240c-90aa-4cc8-a1e4-82b7392515c2.png)

[Table of Content](#0.1)

## 3 Weight and Bias initialization<a class="anchor" id="3"></a>

### 3.1 Why Weight Initialization?<a class="anchor" id="3.1"></a>

Its main objective is to prevent layer activation outputs from exploding or vanishing gradients during the forward propagation. If either of the problems occurs, loss gradients will either be too large or too small, and the network will take more time to converge if it is even able to do so at all.

If we initialized the weights correctly, then our objective i.e, optimization of loss function will be achieved in the least time otherwise converging to a minimum using gradient descent will be impossible.

#### Weight initialization is an important consideration in the design of a neural network model.

Building even a simple neural network can be a confusing task and upon that tuning it to get a better result is extremely tedious. But, the first step that comes in consideration while building a neural network is the initialization of parameters, if done correctly then optimization will be achieved in the least time otherwise converging to a minima using gradient descent will be impossible.

Basic notations
Consider an L layer neural network, which has L-1 hidden layers and 1 input and output layer each. The parameters (weights and biases) for layer l are represented as

![image](https://user-images.githubusercontent.com/99672298/186379787-d673f2b1-73d7-4218-8c05-3f815289c22e.png)

The nodes in neural networks are composed of parameters referred to as weights used to calculate a weighted sum of the inputs.

Neural network models are fit using an optimization algorithm called stochastic gradient descent that incrementally changes the network weights to minimize a loss function, hopefully resulting in a set of weights for the mode that is capable of making useful predictions.

This optimization algorithm requires a starting point in the space of possible weight values from which to begin the optimization process. Weight initialization is a procedure to set the weights of a neural network to small random values that define the starting point for the optimization (learning or training) of the neural network model.

`training deep models is a sufficiently difficult task that most algorithms are strongly affected by the choice of initialization. The initial point can determine whether the algorithm converges at all, with some initial points being so unstable that the algorithm encounters numerical difficulties and fails altogether.`
	
Each time, a neural network is initialized with a different set of weights, resulting in a different starting point for the optimization process, and potentially resulting in a different final set of weights with different performance characteristics.

+ **Zero initialization**
+ **Random initialization**

[Table of Content](#0.1)

### + **`3.2 Zero initialization :`**<a class="anchor" id="3.2"></a>

In general practice biases are initialized with 0 and weights are initialized with random numbers, what if weights are initialized with 0?

In order to understand let us consider we applied sigmoid activation function for the output layer.

![image](https://user-images.githubusercontent.com/99672298/186380171-febdf940-b5d6-4ee1-91a5-dfe643215971.png)

If we initialized all the weights with 0, then what happens is that the derivative wrt loss function is the same for every weight, thus all weights have the same value in subsequent iterations. This makes hidden layers symmetric and this process continues for all the n iterations. Thus initialized weights with zero make your network no better than a linear model. An important thing to keep in mind is that biases have no effect what so ever when initialized with 0. It is important to note that setting biases to 0 will not create any problems as non-zero weights take care of breaking the symmetry and even if bias is 0, the values in every neuron will still be different.

W[l] = np.random.zeros((l-1,l))

let us consider a neural network with only three hidden layers with ReLu activation function in hidden layers and sigmoid for the output layer.

Using the above neural network on the dataset “make circles” from sklearn.datasets, the result obtained as the following :

for 15000 iterations, loss = 0.6931471805599453, accuracy = 50 %

![image](https://user-images.githubusercontent.com/99672298/186380403-222b7c53-2e45-4fe5-83d0-4ccfa2e7037c.png)

clearly, zero initialization isn’t successful in classification.

We cannot initialize all weights to the value 0.0 as the optimization algorithm results in some asymmetry in the error gradient to begin searching effectively.

Historically, weight initialization follows simple heuristics, such as:

+ Small random values in the range [-0.3, 0.3]
+ Small random values in the range [0, 1]
+ Small random values in the range [-1, 1]
+ These heuristics continue to work well in general.

`We almost always initialize all the weights in the model to values drawn randomly from a Gaussian or uniform distribution. The choice of Gaussian or uniform distribution does not seem to matter very much, but has not been exhaustively studied. The scale of the initial distribution, however, does have a large effect on both the outcome of the optimization procedure and on the ability of the network to generalize.`

[Table of Content](#0.1)

### +  **`3.3 Random initialization (Initialized weights randomly):`**<a class="anchor" id="3.3"></a>

– This technique tries to address the problems of zero initialization since it prevents neurons from learning the same features of their inputs since our goal is to make each neuron learn different functions of its input and this technique gives much better accuracy than zero initialization.

– In general, it is used to break the symmetry. It is better to assign random values except 0 to weights.

– Remember, neural networks are very sensitive and prone to overfitting as it quickly memorizes the training data.

Now, after reading this technique a new question comes to mind: **“What happens if the weights initialized randomly can be very high or very low?”**

(a) Vanishing gradients :

 For any activation function, abs(dW) will get smaller and smaller as we go backward with every layer during backpropagation especially in the case of deep neural networks. So, in this case, the earlier layers’ weights are adjusted slowly.
Due to this, the weight update is minor which results in slower convergence.
This makes the optimization of our loss function slow. It might be possible in the worst case, this may completely stop the neural network from training further.
More specifically, in the case of the sigmoid and tanh and activation functions, if your weights are very large, then the gradient will be vanishingly small, effectively preventing the weights from changing their value. This is because abs(dW) will increase very slightly or possibly get smaller and smaller after the completion of every iteration.
So, here comes the use of the RELU activation function in which vanishing gradients are generally not a problem as the gradient is 0 for negative (and zero) values of inputs and 1 for positive values of inputs.
(b) Exploding gradients : 

This is the exact opposite case of the vanishing gradients, which we discussed above.
Consider we have weights that are non-negative, large, and having small activations A. When these weights are multiplied along with the different layers, they cause a very large change in the value of the overall gradient (cost). This means that the changes in W, given by the equation W= W — ⍺ * dW, will be in huge steps, the downward moment will increase.
Problems occurred due to exploding gradients:

– This problem might result in the oscillation of the optimizer around the minima or even overshooting the optimum again and again and the model will never learn!

– Due to the large values of the gradients, it may cause numbers to overflow which results in incorrect computations or introductions of NaN’s (missing values).

Assigning random values to weights is better than just 0 assignment. But there is one thing to keep in my mind is that what happens if weights are initialized high values or very low values and what is a reasonable initialization of weight values.

a) If weights are initialized with very high values the term np.dot(W,X)+b becomes significantly higher and if an activation function like sigmoid() is applied, the function maps its value near to 1 where the slope of gradient changes slowly and learning takes a lot of time.

b) If weights are initialized with low values it gets mapped to 0, where the case is the same as above.

This problem is often referred to as the vanishing gradient.

To see this let us see the example we took above but now the weights are initialized with very large values instead of 0 :

W[l] = np.random.randn(l-1,l)*10

Neural network is the same as earlier, using this initialization on the dataset “make circles” from sklearn.datasets, the result obtained as the following :

for 15000 iterations, loss = 0.38278397192120406, accuracy = 86 %

![image](https://user-images.githubusercontent.com/99672298/186380638-9c4352f0-b4fd-4c23-8cef-7fddbd6ceba6.png)

This solution is better but doesn’t properly fulfil the needs so, let us see a new technique.

### New Initialization techniques 

### + **3.4 Weight Initialization for ReLU**<a class="anchor" id="3.4"></a>
 
As we saw above that with large or 0 initialization of weights(W), not significant result is obtained even if we use appropriate initialization of weights it is probable that training process is going to take longer time. There are certain problems associated with it :

a) If the model is too large and takes many days to train then what

b) What about vanishing/exploding gradient problem

The “xavier” weight initialization was found to have problems when used to initialize networks that use the rectified linear (ReLU) activation function.

As such, a modified version of the approach was developed specifically for nodes and layers that use ReLU activation, popular in the hidden layers of most multilayer Perceptron and convolutional neural network models.

The current standard approach for initialization of the weights of neural network layers and nodes that use the rectified linear (ReLU) activation function is called “he” initialization.

These were some problems that stood in the path for many years but in 2015, He et al. (2015) proposed activation aware initialization of weights (for ReLu) that was able to resolve this problem. ReLu and leaky ReLu also solves the problem of vanishing gradient.

He initialization: we just simply multiply random initialization with

![image](https://user-images.githubusercontent.com/99672298/186381495-05b8773d-882b-44b3-bc66-d89b8bcf0a4d.png)

To see how effective this solution is, let us use the previous dataset and neural network we took for above initialization and results are :

for 15000 iterations, loss =0.07357895962677366, accuracy = 96 %

![image](https://user-images.githubusercontent.com/99672298/186381560-ab064e4e-b796-45aa-b710-d8219d09abd3.png)

Surely, this is an improvement over the previous techniques.

There are also some other techniques other than He initialization in use that is comparatively better than old techniques and are used frequently.

### + **3.5 Weight Initialization for Sigmoid and Tanh**<a class="anchor" id="3.5"></a>

The current standard approach for initialization of the weights of neural network layers and nodes that use the Sigmoid or TanH activation function is called “glorot” or “xavier” initialization.

There are two versions of this weight initialization method, which we will refer to as “xavier” and “normalized xavier.”

`Glorot and Bengio proposed to adopt a properly scaled uniform distribution for initialization. This is called “Xavier” initialization […] Its derivation is based on the assumption that the activations are linear. This assumption is invalid for ReLU and PReLU.`

Both approaches were derived assuming that the activation function is linear, nevertheless, they have become the standard for nonlinear activation functions like Sigmoid and Tanh, but not ReLU.

Xavier initialization: It is same as He initialization but it is used for Sigmoid and tanh() activation function, in this method 2 is replaced with 1.

![image](https://user-images.githubusercontent.com/99672298/186381877-664cc64a-6a5b-40d0-a48e-812a24d16737.png)

Some also use the following technique for initialization :

![image](https://user-images.githubusercontent.com/99672298/186381925-5af62ac0-d74e-4fb4-ab33-fdee411215cd.png)

These methods serve as good starting points for initialization and mitigate the chances of exploding or vanishing gradients. They set the weights neither too much bigger than 1, nor too much less than 1. So, the gradients do not vanish or explode too quickly. **They help avoid slow convergence, also ensuring that we do not keep oscillating off the minima.**

[Table of Content](#0.1)

### 3.6 Best Practices for Weight Initialization<a class="anchor" id="3.6"></a>

👉 Use RELU or leaky RELU as the activation function, as they both are relatively robust to the vanishing or exploding gradient problems (especially for networks that are not too deep). In the case of leaky RELU, they never have zero gradients. Thus they never die and training continues.

👉 Use Heuristics for weight initialization: For deep neural networks, we can use any of the following heuristics to initialize the weights depending on the chosen non-linear activation function.

While these heuristics do not completely solve the exploding or vanishing gradients problems, they help to reduce it to a great extent. The most common heuristics are as follows:

(a) For RELU activation function: This heuristic is called He-et-al Initialization.

In this heuristic, we multiplied the randomly generated values of W by:

![image](https://user-images.githubusercontent.com/99672298/186389673-b4a392f1-369f-457d-b1bf-eebffa24ed48.png)

b) For tanh activation function : This heuristic is known as Xavier initialization.

In this heuristic, we multiplied the randomly generated values of W by:

![image](https://user-images.githubusercontent.com/99672298/186389731-d569ba97-fbac-4a7a-9dc6-a73d3d138702.png)

+ **Benefits of using these heuristics:**

+ All these heuristics serve as good starting points for weight initialization and they reduce the chances of exploding or vanishing gradients.
+ All these heuristics do not vanish or explode too quickly, as the weights are neither too much bigger than 1 nor too much less than 1.
+ They help to avoid slow convergence and ensure that we do not keep oscillating off the minima.

+ 👉 Gradient Clipping:  It is another way for dealing with the exploding gradient problem. In this technique, we set a threshold value, and if our chosen function of a gradient is larger than this threshold, then we set it to another value.

**NOTE: In this article, we have talked about various initializations of weights, but not the biases since gradients wrt bias will depend only on the linear activation of that layer, but not depend on the gradients of the deeper layers. Thus, there is not a problem of diminishing or explosion of gradients for the bias terms. So, Biases can be safely initialized to 0.**

#### Conclusion
+ 👉 Zero initialization causes the neuron to memorize the same functions almost in each iteration.

+ 👉 Random initialization is a better choice to break the symmetry. However, initializing weight with much high or low value can result in slower optimization.

+ 👉 Using an extra scaling factor in Xavier initialization, He-et-al Initialization, etc can solve the above issue to some extent. That’s why these are the more recommended weight initialization methods among all.

#### Key Points to Remember
+ **`Weights should be small`**
+ **`Weights should not be same`**
+ **`Weights should have good amount of variance`**

[Table of Content](#0.1)

## 4 Activation Functions<a class="anchor" id="4"></a>

#### Activation Function: 

![image](https://user-images.githubusercontent.com/99672298/186180131-164203c5-dabd-466e-a6d9-b0874dd7edd0.png)
![image](https://user-images.githubusercontent.com/99672298/186179961-02f7183d-85ea-41dd-bd5a-4bd3acfb39b6.png)

An activation function in a neural network defines how the weighted sum of the input is transformed into an output from a node or nodes in a layer of the network.

It decides whether a neuron should be activated or not. This means that it will decide whether the neurons input to the network is important or not in the process of prediction.

Sometimes the activation function is called a “transfer function.” If the output range of the activation function is limited, then it may be called a “squashing function.” Many activation functions are nonlinear and may be referred to as the “nonlinearity” in the layer or the network design.

The choice of activation function has a large impact on the capability and performance of the neural network, and different activation functions may be used in different parts of the model.

Technically, the activation function is used within or after the internal processing of each node in the network, although networks are designed to use the same activation function for all nodes in a layer.

A network may have three types of layers: input layers that take raw input from the domain, hidden layers that take input from another layer and pass output to another layer, and output layers that make a prediction.

All hidden layers typically use the same activation function. The output layer will typically use a different activation function from the hidden layers and is dependent upon the type of prediction required by the model.

Activation functions are also typically differentiable, meaning the first-order derivative can be calculated for a given input value. This is required given that neural networks are typically trained using the backpropagation of error algorithm that requires the derivative of prediction error in order to update the weights of the model.

There are many different types of activation functions used in neural networks, although perhaps only a small number of functions used in practice for hidden and output layers.

Let’s take a look at the activation functions used for each type of layer in turn.

### 4.1 Why do we need Activation Function in Neural Network?<a class="anchor" id="4.1"></a>
Well, the purpose of an activation function is to add non-linearity to the neural network. If we have a neural network working without the activation functions. In that case, every neuron will only be performed a linear transformation on the inputs using weights and biases. It's because it doesn't matter how many hidden layers we attach in the neural networks, all layers will behave in the same way because the composition of two linear functions is a linear function itself.

Although, the nerual network becomes simpler learning model any complex taks is impossible and our model would be just a linear regression model.

![image](https://user-images.githubusercontent.com/99672298/186457519-261e2f28-a975-4865-a3dd-86aaf496f9e7.png)

#### Activation for Hidden Layers
A hidden layer in a neural network is a layer that receives input from another layer (such as another hidden layer or an input layer) and provides output to another layer (such as another hidden layer or an output layer).

A hidden layer does not directly contact input data or produce outputs for a model, at least in general.

A neural network may have more hidden than 1 layers.

Typically, a differentiable nonlinear activation function is used in the hidden layers of a neural network. This allows the model to learn more complex functions than a network trained using a linear activation function.

[Table of Content](#0.1)

### 4.2 Basic Types of Neural Network Activation Function<a class="anchor" id="4.2"></a>

#### Types of Activation Functions

We have divided all the essential neural networks in three major parts:

A. Binary step function

B. Linear function

C. Non linear activation function

#### A. Binary Step Neural Network Activation Function
 
1. Binary Step Function
 
This activation function very basic and it comes to mind every time if we try to bound output. It is basically a threshold base classifier, in this, we decide some threshold value to decide output that neuron should be activated or deactivated.

Binary step function depends on a threshold value that decides whether a neuron should be activated or not. 

The input fed to the activation function is compared to a certain threshold; if the input is greater than it, then the neuron is activated, else it is deactivated, meaning that its output is not passed on to the next hidden layer.

![image](https://user-images.githubusercontent.com/99672298/186428859-9f4c9b96-7da5-4a1c-bb47-5f1ad424d526.png)
![image](https://user-images.githubusercontent.com/99672298/186428922-9bbf30bb-13a5-46ed-9cb5-620fce35bb28.png)

In this, we decide the threshold value to 0. It is very simple and useful to classify binary problems or classifier.

Here are some of the limitations of binary step function:

+ It cannot provide multi-value outputs—for example, it cannot be used for multi-class classification problems. 
+ The gradient of the step function is zero, which causes a hindrance in the backpropagation process.

#### B. Linear Neural Network Activation Function
 
2. Linear Function
 
It is a simple straight line activation function where our function is directly proportional to the weighted sum of neurons or input. Linear activation functions are better in giving a wide range of activations and a line of a positive slope may increase the firing rate as the input rate increases.

In binary, either a neuron is firing or not. If you know gradient descent in deep learning then you would notice that in this function derivative is constant.

Y = mZ

![image](https://user-images.githubusercontent.com/99672298/186444216-138f162e-dfe3-49c4-a5d2-f65d9f64b5ac.png)

Where derivative with respect to Z is constant m. The meaning gradient is also constant and it has nothing to do with Z. In this, if the changes made in backpropagation will be constant and not dependent on Z so this will not be good for learning. 

In this, our second layer is the output of a linear function of previous layers input. Wait a minute, what have we learned in this that if we compare our all the layers and remove all the layers except the first and last then also we can only get an output which is a linear function of the first layer.

In this the activation is proportional to the input which means the function doesn't do anything to the weighted sum of the input, it simply spits out the value it was given.

The linear activation function, also known as "no activation," or "identity function" (multiplied x1.0), is where the activation is proportional to the input.

![image](https://user-images.githubusercontent.com/99672298/186430449-7b4e123f-b15c-454b-9484-3d314f70dd16.png)

Range : (-infinity to infinity)

Mathematically it can be represented as:

![image](https://user-images.githubusercontent.com/99672298/186430668-2ecf6c6c-22b5-418b-b035-bbbb002c12ad.png)

However, a linear activation function has two major problems :

+ It’s not possible to use backpropagation as the derivative of the function is a constant and has no relation to the input x. 
+ All layers of the neural network will collapse into one if a linear activation function is used. No matter the number of layers in the neural network, the last layer will still be a linear function of the first layer. So, essentially, a linear activation function turns the neural network into just one layer.

It doesn’t help with the complexity or various parameters of usual data that is fed to the neural networks.

#### C. Non Linear Neural Network Activation Function

The linear activation function shown above is simply a linear regression model. 

Because of its limited power, this does not allow the model to create complex mappings between the network’s inputs and outputs. 

**Non-linear activation function solve the following limitations of linear activation functions:**

+ They allow backpropagation because now the derivative function would be related to the input, and it’s possible to go back and understand which weights in the input neurons can provide a better prediction.
+ They allow the stacking of multiple layers of neurons as the output would now be a non-linear combination of input passed through multiple layers. Any output can be represented as a functional computation in a neural network.

The Nonlinear Activation Functions are the most used activation functions. Nonlinearity helps to makes the graph look something like this

![image](https://user-images.githubusercontent.com/99672298/186443936-5913993f-5228-4321-8f77-ade869b1b0a7.png)

3. ReLU( Rectified Linear unit) Activation function
 
Rectified linear unit or ReLU is most widely used activation function right now which ranges from 0 to infinity, All the negative values are converted into zero, and this conversion rate is so fast that neither it can map nor fit into data properly which creates a problem, but where there is a problem there is a solution.

![image](https://user-images.githubusercontent.com/99672298/186431603-d57c4793-ecf0-4e63-ab11-c92db66418a4.png)

We use Leaky ReLU function instead of ReLU to avoid this unfitting, in Leaky ReLU range is expanded which enhances the performance.

[Table of Content](#0.1)
___
### 4.3 Non-Linear Neural Networks Activation Functions<a class="anchor" id="4.3"></a>
	
#### Sigmoid / Logistic Activation Function 

![30 06 2022_16 21 10_REC](https://user-images.githubusercontent.com/99672298/186486535-bb15b142-e027-4e83-b9ec-8765caf31114.png)

The sigmoid activation function is used mostly as it does its task with great efficiency, it basically is a probabilistic approach towards decision making and ranges in between 0 to 1, so when we have to make a decision or to predict an output we use this activation function because of the range is the minimum, therefore, prediction would be more accurate.

The function is differentiable.That means, we can find the slope of the sigmoid curve at any two points.

The function is monotonic but function’s derivative is not.

The logistic sigmoid function can cause a neural network to get stuck at the training time.

This function takes any real value as input and outputs values in the range of 0 to 1. 

The larger the input (more positive), the closer the output value will be to 1.0, whereas the smaller the input (more negative), the closer the output will be to 0.0, as shown below.

![image](https://user-images.githubusercontent.com/99672298/186431840-9cc7635b-b597-474c-a8a2-6beb821997cf.png)

Mathematically it can be represented as:

![image](https://user-images.githubusercontent.com/99672298/186431932-8fabc628-6d55-4789-bf7d-3ac22b2c8f8a.png)

Here’s why sigmoid/logistic activation function is one of the most widely used functions:

+ It is commonly used for models where we have to predict the probability as an output. Since probability of anything exists only between the range of 0 and 1, sigmoid is the right choice because of its range.
+ The function is differentiable and provides a smooth gradient, i.e., preventing jumps in output values. This is represented by an S-shape of the sigmoid activation function. 

The limitations of sigmoid function are discussed below:

The sigmoid function causes a problem mainly termed as vanishing gradient problem which occurs because we convert large input in between the range of 0 to 1 and therefore their derivatives become much smaller which does not give satisfactory output. To solve this problem another activation function such as ReLU is used where we do not have a small derivative problem.

As we can see from the above Figure, the gradient values are only significant for range -3 to 3, and the graph gets much flatter in other regions. 

It implies that for values greater than 3 or less than -3, the function will have very small gradients. As the gradient value approaches zero, the network ceases to learn and suffers from the Vanishing gradient problem.

The output of the logistic function is not symmetric around zero. So the output of all the neurons will be of the same sign. This makes the training of the neural network more difficult and unstable.

The derivative of the function is f'(x) = sigmoid(x)*(1-sigmoid(x)). 

![15 08 2022_20 30 01_REC](https://user-images.githubusercontent.com/99672298/186432936-96eb2ec6-38f0-4aae-b5a7-0bd094d69759.png)
![image](https://user-images.githubusercontent.com/99672298/186433142-6aacecdd-3dc2-454b-a82d-a8598ea77dc5.png)
![15 08 2022_20 42 50_REC](https://user-images.githubusercontent.com/99672298/186433960-edca5375-13d9-4f17-aa56-2a8e28087aba.png)
![image](https://user-images.githubusercontent.com/99672298/186434148-cc9a2c8c-ec65-4bc4-8baf-5a14c0f4d149.png)

The softmax function is a more generalized logistic activation function which is used for multiclass classification.
___
#### Tanh Function (Hyperbolic Tangent)

![image](https://user-images.githubusercontent.com/99672298/186445494-e48d8e01-9799-424f-a388-9896b20d3842.png)
![30 06 2022_19 27 44_REC](https://user-images.githubusercontent.com/99672298/186486571-d263d21f-d021-4338-a343-2d07caed2cc8.png)

Tanh function is very similar to the sigmoid/logistic activation function, and even has the same S-shape with the difference in output range of -1 to 1. In Tanh, the larger the input (more positive), the closer the output value will be to 1.0, whereas the smaller the input (more negative), the closer the output will be to -1.0.

This activation function is slightly better than the sigmoid function, like the sigmoid function it is also used to predict or to differentiate between two classes but it maps the negative input into negative quantity only and ranges in between -1 to  1.

![image](https://user-images.githubusercontent.com/99672298/186433618-329390d6-e265-4466-a857-50cf189f6ef2.png)
![image](https://user-images.githubusercontent.com/99672298/186434653-a56990e3-961e-4afc-a20a-a747934cba52.png)

As you can see— it also faces the problem of vanishing gradients similar to the sigmoid activation function. Plus the gradient of the tanh function is much steeper as compared to the sigmoid function.

![15 08 2022_20 47 58_REC](https://user-images.githubusercontent.com/99672298/186434316-120da1d4-bb7c-4014-9f1d-0cdb1e8ea60a.png)
![15 08 2022_20 50 39_REC](https://user-images.githubusercontent.com/99672298/186446590-3a74561a-436a-4b0a-bc3b-ab0aaae725bb.png)

Mathematically it can be represented as:

![image](https://user-images.githubusercontent.com/99672298/186434452-c6e5be4b-b3e4-44d3-bff1-f5353194ebb3.png)

**Advantages of using this activation function are:**

+ The output of the tanh activation function is Zero centered; hence we can easily map the output values as strongly negative, neutral, or strongly positive.
+ Usually used in hidden layers of a neural network as its values lie between -1 to; therefore, the mean for the hidden layer comes out to be 0 or very close to it. It helps in centering the data and makes learning for the next layer much easier.
+ The tanh functions have been used mostly in RNN for natural language processing and speech recognition tasks
+ The advantage is that the negative inputs will be mapped strongly negative and the zero inputs will be mapped near zero in the tanh graph.
+ The function is differentiable.
+ The function is monotonic while its derivative is not monotonic.
+ The tanh function is mainly used classification between two classes.
+ Both tanh and logistic sigmoid activation functions are used in feed-forward nets.

Have a look at the gradient of the tanh activation function to understand its limitations.

	💡 Note:  Although both sigmoid and tanh face vanishing gradient issue, tanh is zero centered, and the gradients are not restricted to move in a certain direction. Therefore, in practice, tanh nonlinearity is always preferred to sigmoid nonlinearity.
___
#### ReLU Function
#### ReLU stands for Rectified Linear Unit. 

![15 08 2022_20 52 09_REC](https://user-images.githubusercontent.com/99672298/186446640-b91ecb44-1653-4965-9770-b4d79ce097af.png)
![15 08 2022_22 32 34_REC](https://user-images.githubusercontent.com/99672298/186446717-20b03278-b045-4583-83fe-3c93430e5985.png)
![15 08 2022_22 35 43_REC](https://user-images.githubusercontent.com/99672298/186446754-0ff06be1-2465-4d00-ad00-6fcc29209b1d.png)
![25 07 2022_15 44 31_REC](https://user-images.githubusercontent.com/99672298/186486411-0f662157-7108-4a29-9c34-d9b098b36fc2.png)

The rectified linear activation function, or ReLU activation function, is perhaps the most common function used for hidden layers.

It is common because it is both simple to implement and effective at overcoming the limitations of other previously popular activation functions, such as Sigmoid and Tanh. Specifically, it is less susceptible to vanishing gradients that prevent deep models from being trained, although it can suffer from other problems like saturated or “dead” units.

Along with the overall speed of computation enhanced, ReLU provides faster computation since it does not compute exponentials and divisions

· It easily overfits compared to the sigmoid function and is one of the main limitations. Some techniques like dropout are used to reduce the overfitting

Although it gives an impression of a linear function, ReLU has a derivative function and allows for backpropagation while simultaneously making it computationally efficient. 

The main catch here is that the ReLU function does not activate all the neurons at the same time. 

The ReLU function is calculated as follows:

max(0.0, x)

This means that if the input value (x) is negative, then a value 0.0 is returned, otherwise, the value is returned.

The neurons will only be deactivated if the output of the linear transformation is less than 0.

![image](https://user-images.githubusercontent.com/99672298/186445887-4d7c8c5c-7e41-405a-8837-8997154de4b2.png)

As you can see, the ReLU is half rectified (from bottom). f(z) is zero when z is less than zero and f(z) is equal to z when z is above or equal to zero.

Range: [ 0 to infinity)

The function and its derivative both are monotonic.

But the issue is that all the negative values become zero immediately which decreases the ability of the model to fit or train from the data properly. That means any negative input given to the ReLU activation function turns the value into zero immediately in the graph, which in turns affects the resulting graph by not mapping the negative values appropriately.

Mathematically it can be represented as:

![image](https://user-images.githubusercontent.com/99672298/186446036-6e57941d-31b0-45e4-8264-d8fdf9880da0.png)

**The advantages of using ReLU as an activation function are as follows:**

+ Since only a certain number of neurons are activated, the ReLU function is far more computationally efficient when compared to the sigmoid and tanh functions.
+ ReLU accelerates the convergence of gradient descent towards the global minimum of the loss function due to its linear, non-saturating property.

**The limitations faced by this function are:**

+ The Dying ReLU problem, which I explained below.

![image](https://user-images.githubusercontent.com/99672298/186446339-880780df-7de2-4108-a732-fc4245203bd9.png)

The negative side of the graph makes the gradient value zero. Due to this reason, during the backpropagation process, the weights and biases for some neurons are not updated. This can create dead neurons which never get activated. 

All the negative input values become zero immediately, which decreases the model’s ability to fit or train from the data properly. 
___
#### Leaky ReLU Function
Leaky ReLU is an improved version of ReLU function to solve the Dying ReLU problem as it has a small positive slope in the negative area.

![15 08 2022_22 38 17_REC](https://user-images.githubusercontent.com/99672298/186446907-59de1579-70bd-4126-af2d-65e3868e3f9d.png)
![15 08 2022_22 39 33_REC](https://user-images.githubusercontent.com/99672298/186446942-6bb11bff-4748-4c1c-afeb-36e336c641b7.png)
![25 07 2022_15 58 11_REC](https://user-images.githubusercontent.com/99672298/186486465-09d77226-91b9-4f5c-bfe9-14d0dcd1bce3.png)

It is an attempt to solve the dying ReLU problem

![image](https://user-images.githubusercontent.com/99672298/186447034-adeebfc2-f833-4358-b877-d66a6302ac9c.png)
![image](https://user-images.githubusercontent.com/99672298/186447125-84ea6086-8878-4e37-91d7-02ede7ac824a.png)

Mathematically it can be represented as:

![image](https://user-images.githubusercontent.com/99672298/186447280-5556a5b3-557d-477d-83f3-6711c67641f2.png)

The leak helps to increase the range of the ReLU function. Usually, the value of a is 0.01 or so.

When a is not 0.01 then it is called Randomized ReLU.

Therefore the range of the Leaky ReLU is (-infinity to infinity).

Both Leaky and Randomized ReLU functions are monotonic in nature. Also, their derivatives also monotonic in nature.

**The advantages of Leaky ReLU are same as that of ReLU, in addition to the fact that it does enable backpropagation, even for negative input values.**

+ By making this minor modification for negative input values, the gradient of the left side of the graph comes out to be a non-zero value. Therefore, we would no longer encounter dead neurons in that region. 

Here is the derivative of the Leaky ReLU function. 

![image](https://user-images.githubusercontent.com/99672298/186447446-fd4f58cd-ab4e-4bbf-b61c-c4bdd75fd086.png)

**The limitations that this function faces include:**

+ The predictions may not be consistent for negative input values. 
+ The gradient for negative values is a small value that makes the learning of model parameters time-consuming.
___
#### Parametric ReLU Function
Parametric ReLU is another variant of ReLU that aims to solve the problem of gradient’s becoming zero for the left half of the axis. 

This function provides the slope of the negative part of the function as an argument a. By performing backpropagation, the most appropriate value of a is learnt.

![15 08 2022_22 56 07_REC](https://user-images.githubusercontent.com/99672298/186448159-591a0b0f-84ef-4be1-96e2-b6d371b7843a.png)
![15 08 2022_22 58 18_REC](https://user-images.githubusercontent.com/99672298/186448202-01e2ffde-c847-4b78-b903-447397eb2558.png)
![15 08 2022_22 59 38_REC](https://user-images.githubusercontent.com/99672298/186448252-2581171b-6758-4c73-8d3b-b42c9655f6af.png)
![15 08 2022_23 00 14_REC](https://user-images.githubusercontent.com/99672298/186448292-8dd1633c-e685-45df-a2a1-f54e60a73580.png)

Mathematically it can be represented as:

![image](https://user-images.githubusercontent.com/99672298/186448391-1582341d-29a2-4a26-8194-b0f0d52162ba.png)

Where "a" is the slope parameter for negative values.

The parameterized ReLU function is used when the leaky ReLU function still fails at solving the problem of dead neurons, and the relevant information is not successfully passed to the next layer. 

This function’s limitation is that it may perform differently for different problems depending upon the value of slope parameter a.
___
#### Exponential Linear Units (ELUs) Function
Exponential Linear Unit, or ELU for short, is also a variant of ReLU that modifies the slope of the negative part of the function. 

ELU uses a log curve to define the negativ values unlike the leaky ReLU and Parametric ReLU functions with a straight line.

![15 08 2022_22 45 16_REC](https://user-images.githubusercontent.com/99672298/186448565-250cbede-53fe-4e22-a92d-29c1e3f0bb13.png)
![15 08 2022_22 48 46_REC](https://user-images.githubusercontent.com/99672298/186448589-845fa684-b8cf-490d-ab82-9f1b46a998e0.png)
![15 08 2022_22 52 05_REC](https://user-images.githubusercontent.com/99672298/186448626-a5cf3136-9896-44ea-b206-340adfba95f8.png)

Mathematically it can be represented as:

![image](https://user-images.githubusercontent.com/99672298/186448700-30691467-8a9a-4eed-9f4f-f10911369dbd.png)

**ELU is a strong alternative for f ReLU because of the following advantages:**

+ ELU becomes smooth slowly until its output equal to -α whereas RELU sharply smoothes.
+ Avoids dead ReLU problem by introducing log curve for negative values of input. It helps the network nudge weights and biases in the right direction.

**The limitations of the ELU function are as follow:**

+ It increases the computational time because of the exponential operation included
+ No learning of the ‘a’ value takes place
+ Exploding gradient problem

![image](https://user-images.githubusercontent.com/99672298/186449094-1a240dcd-8e61-4656-815f-301e3f42f9f5.png)

Mathematically it can be represented as:

![image](https://user-images.githubusercontent.com/99672298/186449165-43115b5f-3b4f-4d67-9c3f-20835e459db8.png)

#### Argmax Function
Argmax Function
The argmax, or “arg max,” mathematical function returns the index in the list that contains the largest value.

Think of it as the meta version of max: one level of indirection above max, pointing to the position in the list that has the max value rather than the value itself.

![image](https://user-images.githubusercontent.com/99672298/186483749-83d04449-f002-4f0a-a39a-d5d3fc3d1937.png)
![22 08 2022_16 55 26_REC](https://user-images.githubusercontent.com/99672298/186483862-933931e4-fa67-4b30-9bda-b7df7fc8e856.png)
![22 08 2022_16 55 56_REC](https://user-images.githubusercontent.com/99672298/186483887-be975c24-0027-475b-bdbf-9002a4233702.png)
___
#### Softmax Function
Before exploring the ins and outs of the Softmax activation function, we should focus on its building block—the sigmoid/logistic activation function that works on calculating probability values. 

The output of the sigmoid function was in the range of 0 to 1, which can be interpreted of as Predicted "probabilities". 

But— **`💡Note: The word "probabilities is in quotes because we should not put a lot of trust in their accuracy, is that they are in part dependent on the Weights and Biases in the Neural Network and these factors in turn, depends on the randomly selected initial values and if we change those values, we can end up with different such factors that give us a Neural Network that is just as good at classifying the data and different raw output values give us different SoftMax Output values`**

**`In other words, the predicted "probabilities" don't just depend on the input values but also on the random initial values for the Weights and Biases. So don't put a lot of trust in the accuracy of these predicted "probabilities"`**

This function faces certain problems.

Let’s suppose we have five output values of 0.8, 0.9, 0.7, 0.8, and 0.6, respectively. How can we move forward with it?

The answer is: We can’t.

The above values don’t make sense as the sum of all the classes/output probabilities should be equal to 1. 

You see, the Softmax function is described as a combination of multiple sigmoids. 

It calculates the relative probabilities. Similar to the sigmoid/logistic activation function, the SoftMax function returns the probability of each class. 

It is most commonly used as an activation function for the last layer of the neural network in the case of multi-class classification. 

![15 08 2022_22 53 26_REC](https://user-images.githubusercontent.com/99672298/186453040-6ea78f2b-f944-4a2b-9802-ccaa04224e9e.png)
![15 08 2022_22 53 01_REC](https://user-images.githubusercontent.com/99672298/186452983-c953c6ee-105d-4a82-8844-a9bd5b33dc65.png)
![image](https://user-images.githubusercontent.com/99672298/186453207-4397da2f-dc13-4498-808b-def315605f80.png)

Mathematically it can be represented as:

![image](https://user-images.githubusercontent.com/99672298/186453254-b004ef32-ba41-49ea-9741-cd2f0a3bf946.png)
![22 08 2022_19 25 29_REC](https://user-images.githubusercontent.com/99672298/186486317-ace0c63a-218d-459c-81d5-e0b469d692dd.png)

Softmax is used mainly at the last layer i.e output layer for decision making the same as sigmoid activation works, the softmax basically gives value to the input variable according to their weight and the sum of these weights is eventually one.

For Binary classification, both sigmoid, as well as softmax, are equally approachable but in case of multi-class classification problem we generally use softmax and cross-entropy along with it.
___
#### Swish
It is a self-gated activation function developed by researchers at Google. 

Swish consistently matches or outperforms ReLU activation function on deep networks applied to various challenging domains such as image classification, machine translation etc. 

![image](https://user-images.githubusercontent.com/99672298/186453700-11364e81-a36c-4a27-9191-50e681181dde.png)
![15 08 2022_23 00 43_REC](https://user-images.githubusercontent.com/99672298/186453720-2ad1eecf-2edc-4596-aca5-d5384a65aa44.png)
![15 08 2022_23 02 08_REC](https://user-images.githubusercontent.com/99672298/186453764-c91ba61a-8a99-44cb-9a34-7382af60693e.png)

This function is bounded below but unbounded above i.e. Y approaches to a constant value as X approaches negative infinity but Y approaches to infinity as X approaches infinity.

Mathematically it can be represented as:

![image](https://user-images.githubusercontent.com/99672298/186453838-26ec90f6-76cb-4787-a178-c3dd5ccc87d0.png)

Here are a few advantages of the Swish activation function over ReLU:

+ Swish is a smooth function that means that it does not abruptly change direction like ReLU does near x = 0. Rather, it smoothly bends from 0 towards values < 0 and then upwards again.
+ Small negative values were zeroed out in ReLU activation function. However, those negative values may still be relevant for capturing patterns underlying the data. Large negative values are zeroed out for reasons of sparsity making it a win-win situation.
+ The swish function being non-monotonous enhances the expression of input data and weight to be learnt.
+  This function does not suffer from vanishing gradient problems
___
#### Gaussian Error Linear Unit (GELU)
The Gaussian Error Linear Unit (GELU) activation function is compatible with BERT, ROBERTa, ALBERT, and other top NLP models. This activation function is motivated by combining properties from dropout, zoneout, and ReLUs. 

ReLU and dropout together yield a neuron’s output. ReLU does it deterministically by multiplying the input by zero or one (depending upon the input value being positive or negative) and dropout stochastically multiplying by zero. 

RNN regularizer called zoneout stochastically multiplies inputs by one. 

We merge this functionality by multiplying the input by either zero or one which is stochastically determined and is dependent upon the input. We multiply the neuron input x by 

m ∼ Bernoulli(Φ(x)), where Φ(x) = P(X ≤x), X ∼ N (0, 1) is the cumulative distribution function of the standard normal distribution. 

This distribution is chosen since neuron inputs tend to follow a normal distribution, especially with Batch Normalization.

![image](https://user-images.githubusercontent.com/99672298/186454075-46f1e02d-c6ea-4742-9acf-b698104ea7ac.png)

Mathematically it can be represented as:

![image](https://user-images.githubusercontent.com/99672298/186454124-1515ca0a-fe6d-414d-87e1-7744f3cbd63d.png)

GELU nonlinearity is better than ReLU and ELU activations and finds performance improvements across all tasks in domains of computer vision, natural language processing, and speech recognition.
___
#### Scaled Exponential Linear Unit (SELU)
SELU was defined in self-normalizing networks and takes care of internal normalization which means each layer preserves the mean and variance from the previous layers. SELU enables this normalization by adjusting the mean and variance. 

SELU has both positive and negative values to shift the mean, which was impossible for ReLU activation function as it cannot output negative values. 

Gradients can be used to adjust the variance. The activation function needs a region with a gradient larger than one to increase it.

![image](https://user-images.githubusercontent.com/99672298/186454403-8b62e7a4-79d8-4cb0-8600-35d765f5de2b.png)

Mathematically it can be represented as:

![image](https://user-images.githubusercontent.com/99672298/186454447-9d997c5c-2a5d-4d9b-9425-6fd823e147eb.png)

SELU has values of alpha α and lambda λ predefined. 

Here’s the main advantage of SELU over ReLU:

Internal normalization is faster than external normalization, which means the network converges faster.
SELU is a relatively newer activation function and needs more papers on architectures such as CNNs and RNNs, where it is comparatively explored.
___
#### Maxout 

![15 08 2022_23 02 35_REC](https://user-images.githubusercontent.com/99672298/186458363-3fb7d19d-93f0-4cfc-988c-a6d0290144ea.png)
___
#### Softplus Function
· Softplus was proposed by Dugas in 2001, given by the relationship,

f(x)=log (1+e^x)

· Softplus has smoothing and nonzero gradient properties, thereby enhancing the stabilization and performance of DNN designed with soft plus units

· The comparison of the Softplus function with the ReLU and Sigmoid functions showed improved performance with lesser epochs to convergence during training

![image](https://user-images.githubusercontent.com/99672298/186454960-2945434a-6fd1-4402-b1cc-b25e8bb64133.png)
![15 08 2022_23 07 21_REC](https://user-images.githubusercontent.com/99672298/186458423-bb91f073-9dde-4257-8e12-1d82febcbfad.png)
![62b18a8dc83132e1a479b65d_neural-network-activation-function-cheat-sheet](https://user-images.githubusercontent.com/99672298/186478248-1b743493-b770-4646-b7da-eed1ffa7be0c.jpeg)
![26 07 2022_16 06 11_REC](https://user-images.githubusercontent.com/99672298/186486769-1d7f37c8-f1b4-42c1-b729-b10e0f99f4af.png)

[Table of Content](#0.1)

___
### 4.4 How to choose the right Activation Function?<a class="anchor" id="4.4"></a>

#### Activation for Output Layers
The output layer is the layer in a neural network model that directly outputs a prediction.

All feed-forward neural network models have an output layer.

There are perhaps three activation functions you may want to consider for use in the output layer; they are:

+ Linear
+ Logistic (Sigmoid)
+ Softmax

This is not an exhaustive list of activation functions used for output layers, but they are the most commonly used.

You need to match your activation function for your output layer based on the type of prediction problem that you are solving—specifically, the type of predicted variable.

Here’s what you should keep in mind.

As a rule of thumb, you can begin with using the ReLU activation function and then move over to other activation functions if ReLU doesn’t provide optimum results.

And here are a few other guidelines to help you out.

+ ReLU activation function should only be used in the hidden layers.
+ Sigmoid/Logistic and Tanh functions should not be used in hidden layers as they make the model more susceptible to problems during training (due to vanishing gradients).
+ Swish function is used in neural networks having a depth greater than 40 layers.

Finally, a few rules for choosing the activation function for your output layer based on the type of prediction problem that you are solving:

+ Regression - Linear Activation Function
+ Binary Classification—Sigmoid/Logistic Activation Function
+ Multiclass Classification—Softmax
+ Multilabel Classification—Sigmoid
+ Due to the vanishing gradient problem ‘Sigmoid’ and ‘Tanh’ activation functions are avoided sometimes in deep neural network architectures

The activation function used in hidden layers is typically chosen based on the type of neural network architecture.

+ Convolutional Neural Network (CNN): ReLU activation function.
+ Recurrent Neural Network: Tanh and/or Sigmoid activation function.

![image](https://user-images.githubusercontent.com/99672298/186456174-73bbeef6-ed14-46c9-b839-4f744df8ac4f.png)
![image](https://user-images.githubusercontent.com/99672298/186456403-3fe911da-913b-4890-a435-0a4a30b06e00.png)

	And hey—use this cheatsheet to consolidate all the knowledge on the Neural Network Activation Functions that you've just acquired :)
___
#### Why derivative/differentiation is used?
When updating the curve, to know in which direction and how much to change or update the curve depending upon the slope.That is why we use differentiation in almost every part of Machine Learning and Deep Learning.

![image](https://user-images.githubusercontent.com/99672298/186456509-eb80052d-81db-4a4a-bb32-fb9775dc13a5.png)
![image](https://user-images.githubusercontent.com/99672298/186456528-02f52e9c-b87d-459a-9229-169756894059.png)

[Table of Content](#0.1)

## 5 Why are deep neural networks hard to train?<a class="anchor" id="5"></a>
There are two challenges you might encounter when training your deep neural networks.

Let's discuss them in more detail.

### 5.1 Vanishing Gradients<a class="anchor" id="5.1"></a>
Vanishing –
As the backpropagation algorithm advances downwards(or backward) from the output layer towards the input layer, the gradients often get smaller and smaller and approach zero which eventually leaves the weights of the initial or lower layers nearly unchanged. As a result, the gradient descent never converges to the optimum. This is known as the vanishing gradients problem.
Like the sigmoid function, certain activation functions squish an ample input space into a small output space between 0 and 1. 

Therefore, a large change in the input of the sigmoid function will cause a small change in the output. Hence, the derivative becomes small. For shallow networks with only a few layers that use these activations, this isn’t a big problem. 

However, when more layers are used, it can cause the gradient to be too small for training to work effectively. 

### 5.2 Exploding Gradients<a class="anchor" id="5.2"></a>
Exploding –
On the contrary, in some cases, the gradients keep on getting larger and larger as the backpropagation algorithm progresses. This, in turn, causes very large weight updates and causes the gradient descent to diverge. This is known as the exploding gradients problem.
Exploding gradients are problems where significant error gradients accumulate and result in very large updates to neural network model weights during training. 

An unstable network can result when there are exploding gradients, and the learning cannot be completed. 

The values of the weights can also become so large as to overflow and result in something called NaN values. 

### 5.3 Why do the gradients even vanish/explode?<a class="anchor" id="5.3"></a>
Certain activation functions, like the logistic function (sigmoid), have a very huge difference between the variance of their inputs and the outputs. In simpler words, they shrink and transform a larger input space into a smaller output space that lies between the range of [0,1].

![image](https://user-images.githubusercontent.com/99672298/186479457-51c30945-40bf-4302-8be1-1bcded533b64.png)

Observing the above graph of the Sigmoid function, we can see that for larger inputs (negative or positive), it saturates at 0 or 1 with a derivative very close to zero. Thus, when the backpropagation algorithm chips in, it virtually has no gradients to propagate backward in the network, and whatever little residual gradients exist keeps on diluting as the algorithm progresses down through the top layers. So, this leaves nothing for the lower layers.

Similarly, in some cases suppose the initial weights assigned to the network generate some large loss. Now the gradients can accumulate during an update and result in very large gradients which eventually results in large updates to the network weights and leads to an unstable network. The parameters can sometimes become so large that they overflow and result in NaN values.

[Table of Content](#0.1)

### 5.4 How to know if our model is suffering from the Exploding/Vanishing gradient problem?<a class="anchor" id="5.4"></a>

![image](https://user-images.githubusercontent.com/99672298/186481678-13391e6e-e56c-47cf-80c8-1d9b63bf8d85.png)

#### Solutions 
Now that we are well aware of the vanishing/exploding gradients problems, it’s time to learn some techniques that can be used to fix the respective problems.

#### 1. Proper Weight Initialization 
+ The variance of outputs of each layer should be equal to the variance of its inputs.
+ The gradients should have equal variance before and after flowing through a layer in the reverse direction.

Although it is impossible for both conditions to hold for any layer in the network until and unless the number of inputs to the layer ( fanin ) is equal to the number of neurons in the layer ( fanout ), but they proposed a well-proven compromise that works incredibly well in practice: randomly initialize the connection weights for each layer in the network as described in the following equation which is popularly known as Xavier initialization (after the author’s first name) or Glorot initialization (after his last name).

where  fanavg = ( fanin + fanout ) / 2

+ Normal distribution with mean 0 and variance σ2 = 1/ $fan_avg$
+ Or a uniform distribution between -r  and +r , with r = sqrt( 3 / $fan_avg$ )         

Following are some more very popular weight initialization strategies for different activation functions, they only differ by the scale of variance and by the usage of either $fan_avg$ or $fan_in$ 

![image](https://user-images.githubusercontent.com/99672298/186491040-2fe3a27c-042f-4c79-9436-df0aea1adfbb.png)

Using the above initialization strategies can significantly speed up the training and increase the odds of gradient descent converging at a lower generalization error. 

![image](https://user-images.githubusercontent.com/99672298/186491279-c3651e9c-74e3-4d87-b519-5648f670ac59.png)

	he_avg_init = keras.initializers.VarianceScaling(scale=2., mode='fan_avg', distribution='uniform')

#### 2. Using Non-saturating Activation Functions
 
In an earlier section, while studying the nature of sigmoid activation function, we observed that its nature of saturating for larger inputs (negative or positive) came out to be a major reason behind the vanishing of gradients thus making it non-recommendable to use in the hidden layers of the network.

So to tackle the issue regarding the saturation of activation functions like sigmoid and tanh, we must use some other non-saturating functions like ReLu and its alternatives.

ReLU ( Rectified Linear Unit )

![image](https://user-images.githubusercontent.com/99672298/186491425-cc4ebc69-c3d7-4a7b-8b3a-c863e1cde0b0.png)

+ Relu(z) = max(0,z)
+ Outputs 0 for any negative input.
+ Range: [0, infinity]

Unfortunately, the ReLu function is also not a perfect pick for the intermediate layers of the network “in some cases”. It suffers from a problem known as dying ReLus wherein some neurons just die out, meaning they keep on throwing 0 as outputs with the advancement in training.

Some popular alternative functions of the ReLU that mitigates the problem of vanishing gradients when used as activation for the intermediate layers of the network  are LReLU, PReLU, ELU, SELU :

LReLU (Leaky ReLU)

+ LeakyReLUα(z) = max(αz, z)
+ The amount of “leak” is controlled by the hyperparameter α, it is the slope of the function for z < 0.
+ The smaller slope for the leak ensures that the neurons powered by leaky Relu never die; although they might venture into a state of coma for a long training phase they always have a chance to eventually wake up.
+ α can also be trained, that is, the model learns the value of α during training. This variant wherein α is now considered a parameter rather than a hyperparameter is called parametric leaky ReLu (PReLU).

ELU (Exponential Linear Unit)

+ For z < 0, it takes on negative values which allow the unit to have an average output closer to 0 thus alleviating the vanishing gradient problem
+ For z < 0, the gradients are non zero. This avoids the dead neurons problem.
+ For α = 1, the function is smooth everywhere, this speeds up the gradient descent since it does not bounce right and left around z=0.
+ A scaled version of this function ( SELU: Scaled ELU ) is also used very often in Deep Learning.

#### 3. Batch Normalization
 
Using He initialization along with any variant of the ReLU activation function can significantly reduce the chances of vanishing/exploding problems at the beginning. However, it does not guarantee that the problem won’t reappear during training.

The Following key points explain the intuition behind BN and how it works:

It consists of adding an operation in the model just before or after the activation function of each hidden layer.
This operation simply zero-centers and normalizes each input, then scales and shifts the result using two new parameter vectors per layer: one for scaling, the other for shifting.
In other words, the operation lets the model learn the optimal scale and mean of each of the layer’s inputs.
To zero-center and normalize the inputs, the algorithm needs to estimate each input’s mean and standard deviation.
It does so by evaluating the mean and standard deviation of the input over the current mini-batch (hence the name “Batch Normalization”).

![image](https://user-images.githubusercontent.com/99672298/186492785-589797f1-56a3-4f6e-93a3-8e963ee27af1.png)
![image](https://user-images.githubusercontent.com/99672298/186492862-e80b3d5f-7792-4524-970d-3ac790527ee3.png)

#### Gradient Clipping

![image](https://user-images.githubusercontent.com/99672298/186493069-6f44c8a3-013c-4597-837a-05c21205f260.png)

[Table of Content](#0.1)

## 6. How to avoid Overfitting of Neural Networks?<a class="anchor" id="6"></a>

One of the most important aspects when training neural network is avoiding overfitting and also one of the most common problems data science professionals face. Have you come across a situation where your model performed exceptionally well on train data but was not able to predict test data. 

![image](https://user-images.githubusercontent.com/99672298/186583811-e6b289cf-a745-4dd3-a48f-00b3ee28bd9c.png)

Have you seen this image before? As we move towards the right in this image, our model tries to learn too well the details and the noise from the training data, which ultimately results in poor performance on the unseen data.

In other words, while going towards the right, the complexity of the model increases such that the training error reduces but the testing error doesn’t. This is shown in the image below.
Overfitting refers to the phenomenon where a neural network models the training data very well but fails when it sees new data from the same problem. Overfitting is caused by noise in the training data that the neural network picks up during training and learns it as an underlying concept of the data. The model on the right side above is with a high complexity is able to pick up and learn patterns even noise in the data that are just caused by some random fluctuation or error.

On the other hand, the lower complexity network on the left side models the distribution much better by not trying too hard to model each data pattern individually.

![image](https://user-images.githubusercontent.com/99672298/186583922-2b5016d6-68b2-456b-b291-12c2f259b420.png)

Overfitting causes the neurla network model to perform very well during training phase but the performance gets much worse during inference time whrn faced with new data. Less complex neural networks are less susceptible to overfitting. To prevent overfitting or a high variance we must use something

### 6.1 What is Regularization?<a class="anchor" id="6.1"></a>

A universal problem in machine learning has been making an algorithm that performs equally well on training data and any new samples or test dataset. Techniques used in machine learning that have specifically been designed to cater to reducing test error, mostly at the expense of increased training error, are globally known as regularization.

Regularization is a technique which makes slight modifications to the learning algorithm such that the model generalizes better. This in turn improves the model’s performance on the unseen data as well.

![Filter_Method](https://miro.medium.com/max/1296/1*Fy1M8W2VAiDOae65Glxj6A.gif)

Regularization may be defined as any modification or change in the learning algorithm that helps reduce its error over a test dataset, commonly known as generalization error but not on the supplied or training dataset.

In learning algorithms, there are many variants of regularization techniques, each of which tries to cater to different challenges. These can be listed down straightforwardly based on the kind of challenge the technique is trying to deal with:

+ Some try to put extra constraints on the learning of an ML model, like adding restrictions on the range/type of parameter values.
+ Some add more terms in the objective or cost function, like a soft constraint on the parameter values. More often than not, a careful selection of the right constraints and penalties in the cost function contributes to a massive boost in the model's performance, specifically on the test dataset.
+ These extra terms can also be encoded based on some prior information that closely relates to the dataset or the problem statement.
+ One of the most commonly used regularization techniques is creating ensemble models, which take into account the collective decision of multiple models, each trained with different samples of data.

The main aim of regularization is to reduce the over-complexity of the machine learning models and help the model learn a simpler function to promote generalization.

![Filter Method](https://miro.medium.com/max/1400/1*9qFJk_wPzoSD_6c0jLsNzw.gif)

#### Bias-Variance Tradeoff

In order to understand how the deviation of the function is varied, bias and variance can be adopted. Bias is the measurement of deviation or error from the real value of function (Train data), variance is the measurement of deviation in the response variable function while estimating it over a different training sample of the dataset (Test Data)

Therefore, for a generalized data model, we must keep bias possibly low while modelling that leads to high accuracy. And, one should not obtain greatly varied results from output, therefore, low variance is recommended for a model to perform good.

The underlying association between bias and variance is closely related to the overfitting, underfitting and capacity in machine learning such that while calculating the generalization error (where bias and variance are crucial elements) increase in the model capacity can lead to increase in variance and decrease in bias.

The trade-off is the tension amid error introduced by the bias and the variance.

Bias vs variance tradeoff graph here sheds a bit more light on the nuances of this topic and demarcation:

![image](https://user-images.githubusercontent.com/99672298/186588607-d6505209-5c2c-4c8e-9153-b5aeafedf968.png)
![image](https://user-images.githubusercontent.com/99672298/186588944-d4726b22-3faf-4aef-9ece-af8b6f29b3bb.png)
![image](https://user-images.githubusercontent.com/99672298/186589069-0cd31bc1-b35b-4ed0-b959-627acccf8f3f.png)

Regularization of an estimator works by trading increased bias for reduced variance. An effective regularize will be the one that makes the best trade between bias and variance, and the end-product of the tradeoff should be a significant reduction in variance at minimum expense to bias. In simpler terms, this would mean low variance without immensely increasing the bias value.

#### How does Regularization help reduce Overfitting?
Let’s consider a neural network which is overfitting on the training data as shown in the image below.

![image](https://user-images.githubusercontent.com/99672298/186585470-2c550c57-672a-4d4b-8973-193cb3eadf14.png)

If you have studied the concept of regularization **in machine learning, you will have a fair idea that regularization penalizes the coefficients. In deep learning, it actually penalizes the weight matrices of the nodes.**

Assume that our regularization coefficient is so high that some of the weight matrices are nearly equal to zero.

![image](https://user-images.githubusercontent.com/99672298/186585521-0f9d8f0a-00c9-4182-a1dd-e688d6a42c61.png)

This will result in a much simpler linear network and slight underfitting of the training data.

Such a large value of the regularization coefficient is not that useful. We need to optimize the value of regularization coefficient in order to obtain a well-fitted model as shown in the image below.

![image](https://user-images.githubusercontent.com/99672298/186585551-a843464a-083f-4aad-a8d0-7ee48b4f91c3.png)

#### Different Regularization Techniques in Deep Learning
Now that we have an understanding of how regularization helps in reducing overfitting, we’ll learn a few different techniques in order to apply regularization in deep learning.

#### L2 & L1 regularization
L1 and L2 are the most common types of regularization. These update the general cost function by adding another term known as the regularization term.

Cost function = Loss (say, binary cross entropy) + Regularization term

Due to the addition of this regularization term, the values of weight matrices decrease because it assumes that a neural network with smaller weight matrices leads to simpler models. Therefore, it will also reduce overfitting to quite an extent.

However, this regularization term differs in L1 and L2.

#### In L2, we have:

The Regression model that uses L2 regularization is called Ridge Regression.

![image](https://user-images.githubusercontent.com/99672298/186591282-b71ef870-5848-4f71-87d8-3d5557e5ca6c.png)

![image](https://user-images.githubusercontent.com/99672298/186586182-1b8f58f6-31b3-45d7-a0e9-adea8643029b.png)

Regularization adds the penalty as model complexity increases. The regularization parameter (lambda) penalizes all the parameters except intercept so that the model generalizes the data and won’t overfit. Ridge regression adds **“squared magnitude of the coefficient”** as penalty term to the loss function. Here the box part in the above image represents the L2 regularization element/term.

Lambda is a hyperparameter.

If lambda is zero, then it is equivalent to OLS.

Ordinary Least Square or OLS, is a stats model which also helps us in identifying more significant features that can have a heavy influence on the output.

But if lambda is very large, then it will add too much weight, and it will lead to under-fitting. Important points to be considered about L2 can be listed below:

+ Ridge regularization forces the weights to be small but does not make them zero and does not give the sparse solution.
+ Ridge is not robust to outliers as square terms blow up the error differences of the outliers, and the regularization term tries to fix it by penalizing the weights.
+ Ridge regression performs better when all the input features influence the output, and all with weights are of roughly equal size.
+ L2 regularization can learn complex data patterns

Here, lambda is the regularization parameter. It is the hyperparameter whose value is optimized for better results. **`L2 regularization is also known as weight decay as it forces the weights to decay towards zero (but not exactly zero).`**

#### In L1, we have:

Lasso Regression (Least Absolute Shrinkage and Selection Operator) adds “Absolute value of magnitude” of coefficient, as penalty term to the loss function.

Lasso shrinks the less important feature’s coefficient to zero; thus, removing some feature altogether. So, this works well for feature selection in case we have a huge number of features.

![image](https://user-images.githubusercontent.com/99672298/186589224-d9de4588-db1d-4db3-805c-b7d47ec15ba5.png)

![image](https://user-images.githubusercontent.com/99672298/186586233-498f471b-3331-4853-9f50-07660595a5db.png)

+ **`L1 regularization is that it is easy to implement and can be trained as a one-shot thing, meaning that once it is trained you are done with it and can just use the parameter vector and weights.`**
+ **`L1 regularization is robust in dealing with outliers. It creates sparsity in the solution (most of the coefficients of the solution are zero), which means the less important features or noise terms will be zero. It makes L1 regularization robust to outliers.`**

To understand the above mentioned point, let us go through the following example and try to understand what it means when an algorithm is said to be sensitive to outliers

+ For instance we are trying to classify images of various birds of different species and have a neural network with a few hundred parameters.
+ We find a sample of birds of one species, which we have no reason to believe are of any different species from all the others.
+ We add this image to the training set and try to train the neural network. This is like throwing an outlier into the mix of all the others. By looking at the edge of the hyperspace where the hyperplane is closest to, we pick up on this outlier, but by the time we’ve got to the hyperplane it’s quite far from the plane and is hence an outlier.
+ The solution in such cases is to perform iterative dropout. L1 regularization is a one-shot solution, but in the end we are going to have to make some kind of hard decision on where to cut off the edges of the hyperspace.
+ Iterative dropout is a method of deciding exactly where to cut off. It is a little more expensive in terms of training time, but in the end it might give us an easier decision about how far the hyperspace edges are.

**`Along with shrinking coefficients, the lasso performs feature selection, as well. (Remember the ‘selection‘ in the lasso full-form?) Because some of the coefficients become exactly zero, which is equivalent to the particular feature being excluded from the model.`**

**`In this, we penalize the absolute value of the weights. Unlike L2, the weights may be reduced to zero here. Hence, it is very useful when we are trying to compress our model. Otherwise, we usually prefer L2 over it.`**

![image](https://user-images.githubusercontent.com/99672298/186586553-6b31764f-562e-4473-a8b7-2ced4d49f5d7.png)
![image](https://user-images.githubusercontent.com/99672298/186600743-36d9ff0e-b57f-4934-9fc6-53dac55dea3e.png)

#### Intuitively Speaking
Smaller weights reduce the impact of the hidden neurons. In that case, those neurons becomes negligible and the overall complexity of the neural network gets reduced.
But we have to be careful. When chossing the regularization term $alpha$. The goal is to strike the right balance between low complexity of the model and accuracy.

+ **`If our $alpha$ value is to high, our model is too simple, but you run the risk of underfitting our data. Our model won't learn enough about the training data to make useful predictions.`**
+ **`If our $alpha$ value is too low, our model will be more complex and we run to the risk of overfitting our data. Our model will learn too much about the particularities of the training data will even pick up the noise in the data and then the model won't even be able to generalize to new data.`**

[Table of Content](#0.1)

### 6.2 What is Dropout?<a class="anchor" id="6.2"></a>

#### How to Dropout
Dropout is implemented per-layer in a neural network.

It can be used with most types of layers, such as dense fully connected layers, convolutional layers, and recurrent layers such as the long short-term memory network layer.

Dropout may be implemented on any or all hidden layers in the network as well as the visible or input layer. It is not used on the output layer.

Dropout is not used after training when making a prediction with the fit network.

The weights of the network will be larger than normal because of dropout. Therefore, before finalizing the network, the weights are first scaled by the chosen dropout rate. The network can then be used as per normal to make predictions.

If a unit is retained with probability p during training, the outgoing weights of that unit are multiplied by p at test time.

The rescaling of the weights can be performed at training time instead, after each weight update at the end of the mini-batch. This is sometimes called “inverse dropout” and does not require any modification of weights during training. Both the Keras and PyTorch deep learning libraries implement dropout in this way.

At test time, we scale down the output by the dropout rate. […] Note that this process can be implemented by doing both operations at training time and leaving the output unchanged at test time, which is often the way it’s implemented in practice

Dropout works well in practice, perhaps replacing the need for weight regularization (e.g. weight decay) and activity regularization (e.g. representation sparsity).

… dropout is more effective than other standard computationally inexpensive regularizers, such as weight decay, filter norm constraints and sparse activity regularization. Dropout may also be combined with other forms of regularization to yield a further improvement.

#### Dropout Regularization
The term 'dropout' refers to dropping out units (both hidden and visible) (neurons) in a neural network. Simply put, dropout refers to ignoring units (i.e. neurons) during the training phase of certain set of neurons which is chosen at random. By “ignoring”, I mean these units are not considered during a particular forward or backward pass. At each training phase, individual nodes are either dropout of the net with probability (1-p) or kept with probability p, so that a shallow network is left (less dense)

#### Why do we need Dropout?
Given that we know a bit about dropout, a question arises — why do we need dropout at all? Why do we need to literally shut-down parts of a neural networks?

#### **`The answer to these questions is “to prevent over-fitting”.`**

A fully connected layer occupies most of the parameters and hence, neurons develop co- dependency amongst each other during training which curbs the individual power of each neuron leading to over-fitting of training data.

+ Training Phase:

Training Phase: For each hidden layer, for each training sample, for each iteration, ignore (zero out) a random fraction, p, of nodes (and corresponding activations).

+ Testing Phase:

Use all activations, but reduce them by a factor p (to account for the missing activations during training).

![image](https://user-images.githubusercontent.com/99672298/186606791-9548d398-9746-4b71-b6ac-2da89f5afef0.png)

Dropout means, that during training with some probability "p" a number of neurons of the neural networks gets turned off during training.

Let say p=0.5 you can observe on right approx half of the neurons are not active.

![Filter Method](https://cdn.analyticsvidhya.com/wp-content/uploads/2018/04/1IrdJ5PghD9YoOyVAQ73MJw.gif)

#### Experiment in Keras

Let’s try this theory in practice. To see how dropout works, I build a deep net in Keras and tried to validate it on the CIFAR-10 dataset. The deep network is built had three convolution layers of size 64, 128 and 256 followed by two densely connected layers of size 512 and an output layer dense layer of size 10 (number of classes in the CIFAR-10 dataset).

I took ReLU as the activation function for hidden layers and sigmoid for the output layer (these are standards, didn’t experiment much on changing these). Also, I used the standard categorical cross-entropy loss.

Finally, I used dropout in all layers and increase the fraction of dropout from 0.0 (no dropout at all) to 0.9 with a step size of 0.1 and ran each of those to 20 epochs. The results look like this:

![image](https://user-images.githubusercontent.com/99672298/186608357-7c217992-35f5-4e76-8ccf-426ae0ece39f.png)
![image](https://user-images.githubusercontent.com/99672298/186608375-41061cb0-03a3-4df9-a50d-92f8cafc46bb.png)

From the above graphs we can conclude that with increasing the dropout, there is some increase in validation accuracy and decrease in loss initially before the trend starts to go down.
There could be two reasons for the trend to go down if dropout fraction is 0.2:

+ 0.2 is actual minima for the this dataset, network and the set parameters used
+ More epochs are needed to train the networks.

This probability of choosing how many nodes should be dropped is the hyperparameter of the dropout function. As seen in the image above, dropout can be applied to both the hidden layers as well as the input layers.

#### Tips for Using Dropout Regularization
This section provides some tips for using dropout regularization with your neural network.

#### Use With All Network Types
Dropout regularization is a generic approach.

It can be used with most, perhaps all, types of neural network models, not least the most common network types of Multilayer Perceptrons, Convolutional Neural Networks, and Long Short-Term Memory Recurrent Neural Networks.

In the case of LSTMs, it may be desirable to use different dropout rates for the input and recurrent connections.

#### Dropout Rate
The default interpretation of the dropout hyperparameter is the probability of training a given node in a layer, where 1.0 means no dropout, and 0.0 means no outputs from the layer.

A good value for dropout in a hidden layer is between 0.5 and 0.8. Input layers use a larger dropout rate, such as of 0.8.

### Use a Larger Network
It is common for larger networks (more layers or more nodes) to more easily overfit the training data.

When using dropout regularization, it is possible to use larger networks with less risk of overfitting. In fact, a large network (more nodes per layer) may be required as dropout will probabilistically reduce the capacity of the network.

A good rule of thumb is to divide the number of nodes in the layer before dropout by the proposed dropout rate and use that as the number of nodes in the new network that uses dropout. For example, a network with 100 nodes and a proposed dropout rate of 0.5 will require 200 nodes (100 / 0.5) when using dropout.

**`If n is the number of hidden units in any layer and p is the probability of retaining a unit […] a good dropout net should have at least n/p units`**

#### Use a Weight Constraint
Network weights will increase in size in response to the probabilistic removal of layer activations.

Large weight size can be a sign of an unstable network.

To counter this effect a weight constraint can be imposed to force the norm (magnitude) of all weights in a layer to be below a specified value. For example, the maximum norm constraint is recommended with a value between 3-4.

[…] we can use max-norm regularization. This constrains the norm of the vector of incoming weights at each hidden unit to be bound by a constant c. Typical values of c range from 3 to 4.

This does introduce an additional hyperparameter that may require tuning for the model.

#### Use With Smaller Datasets
Like other regularization methods, dropout is more effective on those problems where there is a limited amount of training data and the model is likely to overfit the training data.

Problems where there is a large amount of training data may see less benefit from using dropout.

For very large datasets, regularization confers little reduction in generalization error. In these cases, the computational cost of using dropout and larger models may outweigh the benefit of regularization.

#### **`Large neural nets trained on relatively small datasets can overfit the training data.`**

[Table of Content](#0.1)

### 6.3 What is Neural Network Pruning?<a class="anchor" id="6.3"></a>

Much of the success of deep learning has come from building larger and larger neural networks. This allows these models to perform better on various tasks, but also makes them more expensive to use. Larger models take more storage space which makes them harder to distribute. Larger models also take more time to run and can require more expensive hardware. This is especially a concern if you are productionizing a model for a real-world application.

Model compression aims to reduce the size of models while minimizing loss in accuracy or performance. Neural network pruning is a method of compression that involves removing weights from a trained model. In agriculture, pruning is cutting off unnecessary branches or stems of a plant. In machine learning, pruning is removing unnecessary neurons or weights. We will go over some basic concepts and methods of neural network pruning.

#### Need of Inference optimization
As we know that an efficient model is that model which optimizes memory usage and performance at the inference time. Deep Learning model inference is just as crucial as model training, and it is ultimately what determines the solution’s performance metrics. Once the deep learning model has been properly trained for a given application, the next stage is to guarantee that the model is deployed into a production-ready environment, which requires both the application and the model to be efficient and dependable. 

Maintaining a healthy balance between model correctness and inference time is critical. The running cost of the implemented solution is determined by the inference time. It’s crucial to have memory-optimized and real-time (or lower latency) models since the system where your solution will be deployed may have memory limits.

Developers are looking for novel and more effective ways to reduce the computing costs of neural networks as image processing, finance, facial recognition, facial authentication, and voice assistants all require real-time processing. Pruning is one of the most used procedures.

#### What is Neural Network Pruning?
Pruning is the process of deleting parameters from an existing neural network, which might involve removing individual parameters or groups of parameters, such as neurons. This procedure aims to keep the network’s accuracy while enhancing its efficiency. This can be done to cut down on the amount of computing power necessary to run the neural network.

![image](https://user-images.githubusercontent.com/99672298/186634503-63b33500-175b-408c-adf7-c7fe98aaa91a.png)

#### Types of Pruning
Pruning can take many different forms, with the approach chosen based on our desired output. In some circumstances, speed takes precedence over memory, whereas in others, memory is sacrificed. The way sparsity structure, scoring, scheduling, and fine-tuning are handled by different pruning approaches.

![image](https://user-images.githubusercontent.com/99672298/186634887-84db438a-8b0c-4673-8977-e760164e6201.png)

+ **Structured and Unstructured Pruning**

+ + **Remove weights or neurons?**

There are different ways to prune a neural network. 

Individual parameters are pruned using an unstructured pruning approach. This results in a sparse neural network, which, while lower in terms of parameter count, may not be configured in a way that promotes speed improvements. 

Randomly zeroing out the parameters saves memory but may not necessarily improve computing performance because we end up conducting the same number of matrix multiplications as before. 

Because we set specific weights in the weight matrix to zero, this is also known as Weight Pruning.

(1) You can prune weights. This is done by setting individual parameters to zero and making the network sparse. This would lower the number of parameters in the model while keeping the architecture the same. 

Weight-based pruning is more popular as it is easier to do without hurting the performance of the network. However, it requires sparse computations to be effective. This requires hardware support and a certain amount of sparsity to be efficient.

(2) You can remove entire nodes from the network. This would make the network architecture itself smaller, while aiming to keep the accuracy of the initial larger network.

Pruning nodes will allow dense computation which is more optimized. This allows the network to be run normally without sparse computation. This dense computation is more often better supported on hardware. However, removing entire neurons can more easily hurt the accuracy of the neural network.

#### What to prune?
#### **`A major challenge in pruning is determining what to prune. If you are removing weights or nodes from a model, you want the parameters you remove to be less useful. There are different heuristics and methods of determining which nodes are less important and can be removed with minimal effect on accuracy. You can use heuristics based on the weights or activations of a neuron to determine how important it is for the model’s performance. The goal is to remove more of the less important parameters.`**

#### **`One of the simplest ways to prune is based on the magnitude of the weight. Removing a weight is essentially setting it to zero. You can minimize the effect on the network by removing weights that are already close to zero, meaning low in magnitude. This can be implemented by removing all weights below a certain threshold. To prune a neuron based on weight magnitude you can use the L2 norm of the neuron’s weights.`**

#### **`Rather than just weights, activations on training data can be used as a criteria for pruning. When running a dataset through a network, certain statistics of the activations can be observed. You may observe that some neurons always outputs near-zero values. Those neurons can likely be removed with little impact on the model. The intuition is that if a neuron rarely activates with a high value, then it is rarely used in the model’s task.`**

#### **`In addition to the magnitude of weights or activations, redundancy of parameters can mean a neuron can be removed. If two neurons in a layer have very similar weights or activations, it can mean they are doing the same thing. By this intuition, we can remove one of the neurons and preserve the same functionality.`**

#### **`Ideally in a neural network, all the neurons have unique parameters and output activations that are significant in magnitude and not redundant. We want all the neurons are doing something unique, and remove those that are not.`**

#### When to prune?
A major consideration in pruning is where to put it in the training/testing machine learning timeline. If you are using a weight magnitude-based pruning approach, as described in the previous section, you would want to prune after training. However, after pruning, you may observe that the model performance has suffered. This can be fixed by fine-tuning, meaning retraining the model after pruning to restore accuracy.

![image](https://user-images.githubusercontent.com/99672298/186636651-bd960b61-1dac-413e-ba16-c05f180700c4.png)

The usage of pruning can change depending on the application and methods used. Sometimes fine-tuning or multiple iterations of pruning are not necessary. This depends on how much of the network is pruned.

#### How to evaluate pruning?
There multiple metrics to consider when evaluating a pruning method: accuracy, size, and computation time. Accuracy is needed to determine how the model performs on its task. Model size is how much bytes of storage the model takes. To determine computation time, you can use FLOPs (Floating point operations) as a metric. This is more consistent to measure than inference time and it does not depend on what system the model runs on.

With pruning, there is a tradeoff between model performance and efficiency. You can prune heavily and have a smaller more efficient network, but also less accurate. Or you could prune lightly and have a highly performant network, that is also large and expensive to operate. This trade-off needs to be considered for different applications of the neural network.

#### Steps to be followed while pruning:

 + Determine the significance of each neuron.
 + Prioritize the neurons based on their value (assuming there is a clearly defined measure for “importance”).
 + Remove the neuron that is the least significant.
 + Determine whether to prune further based on a termination condition (to be defined by the user).
 + If unanticipated adjustments in data distribution may occur during deployment, don’t prune.
 + If you only have a partial understanding of the distribution shifts throughout training and pruning, prune moderately.
 + If you can account for all movements in the data distribution throughout training and pruning, prune to the maximum extent possible.
 + When retraining, specifically consider data augmentation to maximize the prune potential.
 
### Early stopping<a class="anchor" id="6.4"></a>
Early stopping is a kind of cross-validation strategy where we keep one part of the training set as the validation set. When we see that the performance on the validation set is getting worse, we immediately stop the training on the model. This is known as early stopping.

![image](https://user-images.githubusercontent.com/99672298/186644374-708a8c7b-a962-4799-ac38-3f74d4de1225.png)

In the above image, we will stop training at the dotted line since after that our model will start overfitting on the training data.

In keras, we can apply early stopping using the callbacks function. Below is the sample code for it.

![image](https://user-images.githubusercontent.com/99672298/186644437-dfbaece0-24ef-4e0c-943b-0a5dfa7881c2.png)

Here, monitor denotes the quantity that needs to be monitored and ‘val_err’ denotes the validation error.

Patience denotes the number of epochs with no further improvement after which the training will be stopped. For better understanding, let’s take a look at the above image again. After the dotted line, each epoch will result in a higher value of validation error. Therefore, 5 epochs after the dotted line (since our patience is equal to 5), our model will stop because no further improvement is seen.

Note: It may be possible that after 5 epochs (this is the value defined for patience in general), the model starts improving again and the validation error starts decreasing as well. Therefore, we need to take extra care while tuning this hyperparameter.

[Table of Content](#0.1)

## 7. Step by Step Working of the Artificial Neural Network<a class="anchor" id="7"></a>

#### Steps of Training a Neural Network
Training a neural network consists of the following basic steps:

+ **Step-1:** Initialization of Neural Network: Initialize weights and biases.

+ **Step-2:** Forward propagation: Using the given input X, weights W, and biases b, for every layer we compute a linear combination of inputs and weights (Z)and then apply activation function to linear combination (A). At the final layer, we compute f(A(l-1)) which could be a sigmoid (for binary classification problem), softmax (for multi-class classification problem), and this gives the prediction y_hat.

+ **Step-3:** Compute the loss function: The loss function includes both the actual label y and predicted label y_hat in its expression. It shows how far our predictions from the actual target, and our main objective is to minimize the loss function.

+ **Step-4:** Backward Propagation: In backpropagation, we find the gradients of the loss function, which is a function of y and y_hat, and gradients wrt A, W, and b called dA, dW, and db. By using these gradients, we update the values of the parameters from the last layer to the first layer. 
+ **Step-5:** Repeat steps 2–4 for n epochs till we observe that the loss function is minimized, without overfitting the train data.

For Example,

For a neural network having 2 layers, i.e. one hidden layer. (Here bias term is not added just for the simplicity)

#### **`Forward Propogation`**
![image](https://user-images.githubusercontent.com/99672298/186383610-8bdd9799-dd13-4861-a595-ba790970d193.png)
#### **`Backward Propogation`**
![image](https://user-images.githubusercontent.com/99672298/186383655-a975f9e3-41a5-4c0b-86f4-d77573f2a574.png)

+ **1.) In the first step, Input units are passed i.e data is passes with some weights attached to it to the hidden layers. WE can have any number of hidden layers.**
+ **2.) Each hidden layers consists of neurons. All the inputs are connected to neuron (each).**
+ **3.) After passing on the inputs, all the the computation is performed in the hidden layers.**

#### Computation performed in hidden layers are done in two steps which are as follows:-

##### First of all, all the inputs are multiplied by their respective weights assigned. Weights are the gradient of each variable. It shows the strength of the particular input. After assigning the weights, a bias variable is added. Bias is coefficient of each variable and it is constant that helps the model to fit in the best way possible.

##### Then in the second step, the activation function is applied to the linear equation 'y'. The activation function is a non-linear transformation that is applied to the input before sending it to the next layer of neuron. The importance of the activation function is to incubate non-linearity i the model.

##### The whole process described in point 3 is performed in each hidden layers. After passing through every hidden layers we move to the last layer i.e our output layer which gives us the final output. 
##### **`This process explained above is known as forward Propogation.`**
##### After getting the predictions from the output layers, the error is calculated i.e the difference between the actual and the predicted output. If the error is large then steps are take to minimize the error and for the same purpose **`Back propogation is performed.`**

#### So long story short

Artificial Neural Network (ANN) are comprised of node layers. containing an input layer, one or more hidden layers, and a output layer. Each node or artificial neuron, connects to another and has an associated weigths and threshold.

If the output of any individual node is above the specified threshold value, that ndoe is activated, sending data to the next layer of the network otherwise, no data is passed along to the next layer of the network.

Once an input layer is determined, weight are assigned. These weights help determine the importance of any given variable, with larger ones contributing more significantly to the output compared to other inputs

All inputs are then multiplied by their respective weights and then summed. Afterwards, the output is passed through an activation function, which determines the output. If the ouput exceeds a given threshold, it 'fires' (or activates) the node, passing data to the next layer in the network. This results in the output of one node becoming in the input of the next node.

___

[Table of Content](#0.1)

<div style="display:fill;
            border-radius: false;
            border-style: solid;
            border-color:#000000;
            border-style: false;
            border-width: 2px;
            color:#CF673A;
            font-size:15px;
            font-family: Georgia;
            background-color:#E8DCCC;
            text-align:center;
            letter-spacing:0.1px;
            padding: 0.1em;">

**<h2>♡ Thank you for taking the time ♡**
