### `Read and catch up on content:`
- [Gradient Descent article](https://github.com/MoinDalvs/Gradient_Descent_For_beginners/blob/main/README.md) :books:

## 0.1 Table of Contents<a class="anchor" id="0.1"></a>
1. [Let's talk about Neural Networks.](#1)
2. [Some Basic Concepts Related to Neural Networks](#2)
    - 2.1 [Different layers of a Neural Network](#2.1)
    - 2.2 [Step by Step Working of the Artificial Neural Network](#2.2)
    - 2.3 [Weight and Bias initialization](#2.3)    

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

### 2.2 Step by Step Working of the Artificial Neural Network<a class="anchor" id="2.2"></a>

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
##### After getting the predictions from the output layers, the error is calculated i.e the difference between the actual and the predicted output. If the error is large then steps are take to minimize the error and for the same purpose **`Back propogation is performed.`**

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

### 2.3 Weight and Bias initialization<a class="anchor" id="2.3"></a>

Weight initialization is an important consideration in the design of a neural network model.

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
+ **`Zero initialization :`**

In general practice biases are initialized with 0 and weights are initialized with random numbers, what if weights are initialized with 0?

In order to understand let us consider we applied sigmoid activation function for the output layer.

![image](https://user-images.githubusercontent.com/99672298/186380171-febdf940-b5d6-4ee1-91a5-dfe643215971.png)

If all the weights are initialized with 0, the derivative with respect to loss function is the same for every w in W[l], thus all weights have the same value in subsequent iterations. This makes hidden units symmetric and continues for all the n iterations i.e. setting weights to 0 does not make it better than a linear model. An important thing to keep in mind is that biases have no effect what so ever when initialized with 0.

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

+  **`Random initialization :`**

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

#### New Initialization techniques

+ **Weight Initialization for ReLU**
 
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

+ **Weight Initialization for Sigmoid and Tanh**

The current standard approach for initialization of the weights of neural network layers and nodes that use the Sigmoid or TanH activation function is called “glorot” or “xavier” initialization.

There are two versions of this weight initialization method, which we will refer to as “xavier” and “normalized xavier.”

`Glorot and Bengio proposed to adopt a properly scaled uniform distribution for initialization. This is called “Xavier” initialization […] Its derivation is based on the assumption that the activations are linear. This assumption is invalid for ReLU and PReLU.`

Both approaches were derived assuming that the activation function is linear, nevertheless, they have become the standard for nonlinear activation functions like Sigmoid and Tanh, but not ReLU.

Xavier initialization: It is same as He initialization but it is used for Sigmoid and tanh() activation function, in this method 2 is replaced with 1.

![image](https://user-images.githubusercontent.com/99672298/186381877-664cc64a-6a5b-40d0-a48e-812a24d16737.png)

Some also use the following technique for initialization :

![image](https://user-images.githubusercontent.com/99672298/186381925-5af62ac0-d74e-4fb4-ab33-fdee411215cd.png)

These methods serve as good starting points for initialization and mitigate the chances of exploding or vanishing gradients. They set the weights neither too much bigger than 1, nor too much less than 1. So, the gradients do not vanish or explode too quickly. **They help avoid slow convergence, also ensuring that we do not keep oscillating off the minima.**

