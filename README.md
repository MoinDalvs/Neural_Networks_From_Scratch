## 0.1 Table of Contents<a class="anchor" id="0.1"></a>
1. [Let's talk about Differential Derivative and Partial Derivative](#1)
    - 1.1 [What is Differential?](#1.1)
    - 1.2 [What is Derivative?](#1.2)
    - 1.3 [Partial Derivation (Differentiation)](#1.3)
2. [Gradient Descent](#2)
    - 2.1 [What is Gradient Descent?](#2.1)
    - 2.2 [The Cost function](#2.2)
    - 2.3 [Linear Regression using Gradient Descent](#2.3)
    - 2.4 [Initilization](#2.4)
    - 2.5 [Direction and learning Rate](#2.5)
    - 2.6 [Challenges with Gradient Descent](#2.6)
    - 2.7 [Types of Gradient Descent](#2.7)

## 1. Let's talk about Differential Derivative and Partial Derivative<a class="anchor" id="1"></a>
+ To better understand the difference between the differential and derivative of a function. You need to understand the concept of a function.
  
+ A function is one of the basic concepts in mathematics that defines a relationship between a set of inputs and a set of possible outputs where each input is related to one output. Where Input variables are the independent variables and the other which is output variable is the dependent variable.
  
+ Take for example the statement " 'y' is a function of $x$ i.e y=$f(x)$ which means something related to y is directly related to x by some formula.
  
+ The Calculus as a tool defines the derivative of a function as the limit of a particular kind.
#### 1.1 What is Differential?<a class="anchor" id="1.1"></a>
It is one of the fundamentals divisions of calculus, along with integral calculus. It is a subfield of calculus that deals with infinitesimal change in some varying quantity. The world we live in is full of interrelated quantities that change periodically.

For example, the area of a circular body which changes as the radius changes or a projectile which changes with the velocity. These changing entities, in mathematical terms, are called as variables and **the rate of change of one variable with respect to another is a `derivative`**. **And the equation which represents the relationship between these variables is called a `differential equation.`**
#### 1.2 What is Derivative?<a class="anchor" id="1.2"></a>
  The derivative of a function represents an instantaneous rate of change in the value of a dependent variable with respect to the change in value of the independent variable. **`It’s a fundamental tool of calculus which can also be interpreted as the slope of the tangent line.`** It measures how steep the graph of a function is at some given point on the graph.
  
+ It measures how steep the graph of a function is at some given point on the graph
    
+ Equations which define relationship between there variables and their derivatives are called Differential Equations.
    
+ Differentiation is the process of finding a derivative.
    
+ The derivative of a function is the rate of change of the output value with respect to its input value, where as differential is the actual change of function.
    
+ Differentials are represented as $d_x$, $d_y$, $d_t$, and so on, where $d_x$ represents a small change in $x$, etc.
    
+ When comparing changes in related quantities where y is the function of x, the differential $d_y$ is written as y = $f$(x)
+ $\frac{d_y}{d_x}=f(x)$
+ The derivative of a function is the slope of the function at any point and is written as $\frac{d}{d_x}$.
+ For example, the derivative of $sin(x)$ can be written as $\frac{d_{sin(x)}}{d_x}=cos(x)$
### 1.3 Partial Derivation (Differentiation)<a class="anchor" id="1.3"></a>
Partial differentiation is used to differentiate mathematical functions having more than one variable in them. In ordinary differentiation, we find derivative with respect to one variable only, as function contains only one variable. So partial differentiation is more general than ordinary differentiation. **`Partial differentiation is used for finding maxima and minima in optimization problems`**.

It is a derivative where we hold some independent variable as constant and find derivative with respect to another independent variable.

For example, suppose we have an equation of a curve with X and Y coordinates in it as 2 independent variables. To find the slope in the direction of X, while keeping Y fixed, we will find the partial derivative. Similarly, we can find the slope in the direction of Y (keeping X as Fixed).

Example: consider $f$ = 4$x^2$ + 3y + z 
The partial derivative of the above equation with respect to x is $f^{'}$ = 8$x$

In ordinary differentition the same equation goes like $f_{'}$ = 8$x$ + 3$\frac{d_y}{d_x}$ + $\frac{d_z}{d_x}$

![image](https://user-images.githubusercontent.com/99672298/181267289-b7699632-b5ec-41db-9419-473e2075f248.png)

## 2. Gradient Descent<a class="anchor" id="2"></a>

+ Optimization is the core of every machine learning algorithm

### 2.1 What is Gradient Descent?<a class="anchor" id="2.1"></a>

To explain Gradient Descent I’ll use the classic mountaineering example.

Suppose you are at the top of a mountain, and you have to reach a lake which is at the lowest point of the mountain (a.k.a valley). A twist is that you are blindfolded and you have zero visibility to see where you are headed. So, what approach will you take to reach the lake?

The best way is to check the ground near you and observe where the land tends to descend. This will give an idea in what direction you should take your first step. If you follow the descending path, it is very likely you would reach the lake.

To represent this graphically, notice the below graph.

![image](https://user-images.githubusercontent.com/99672298/181270129-ad968808-f5d8-4e1c-9f43-87ecbe73ef1c.png)

Let us now map this scenario in mathematical terms.

Suppose we want to find out the best parameters (θ1) and (θ2) for our learning algorithm. Similar to the analogy above, we see we find similar mountains and valleys when we plot our “cost space”. Cost space is nothing but how our algorithm would perform when we choose a particular value for a parameter.

So on the y-axis, we have the cost J(θ) against our parameters θ1 and θ2 on x-axis and z-axis respectively. Here, hills are represented by red region, which have high cost, and valleys are represented by blue region, which have low cost.

![image](https://user-images.githubusercontent.com/99672298/181286388-e82f5cec-6c54-442a-af27-6a84d49ed69e.png)


### 2.2 The Cost function<a class="anchor" id="2.2"></a>
It is defind as the measurement of difference of error between actual values and expected values at the current position

The slight difference between the loss fucntion and the cost function is about the error within the training of machine learning models, as loss function refers to the errors of one training example, while a cost function calculates the average error across on entire training set.

+ Gradient descent is an optimization algorithm that works iteratively to find the model parameters with minimal cost or error values. If we go through a formal definition of Gradient descent

+ Gradient descent is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function.Gradient descent is an optimization algorithm used to optimize neural networks and many other machine learning algorithms. Our main goal in optimization is to find the local minima, and gradient descent helps us to take repeated steps in the direction opposite of the gradient of the function at the current point. This method is commonly used in machine learning (ML) and deep learning (DL) to minimize a cost/loss function

![Filter_Method](https://editor.analyticsvidhya.com/uploads/25665ezgif.com-gif-maker.gif)

### 2.3 Linear Regression using Gradient Descent<a class="anchor" id="2.3"></a>

![Filter Method](https://miro.medium.com/max/1400/1*CjTBNFUEI_IokEOXJ00zKw.gif)

A straight line is represented by using the formula y = m$x$ + c
+ y is the dependent variable
+ $x$ is independent variable
+ m is the slope of the line
+ c is the y intercept

![image](https://user-images.githubusercontent.com/99672298/181279168-8949b883-3cb1-4a6f-a7d6-70a506a67ff8.png)

#### (MSE) Loss Function
We will use the Mean Squared Error function to calculate the loss. There are three steps in this function:

+ Find the difference between the actual y and predicted y value(y = mx + c), for a given x.
+ Square this difference.
+ Find the mean of the squares for every value in X.

![image](https://user-images.githubusercontent.com/99672298/181279524-875c3382-4732-42f8-bf25-699bb33a5e91.png)

Here yᵢ is the actual value and ȳᵢ is the predicted value. Lets substitute the value of ȳᵢ:

![image](https://user-images.githubusercontent.com/99672298/181279573-e5b8c097-87c3-4527-b249-98cb45d30dd6.png)

**The loss function helps to increase and improve machine learning efficiency by providing feedback to the model, so that it can minimize error and find the global minima. Further, it continuously iterates along the direction of the negative gradient until the cost function approaches zero.**

Calculating the partial derivative of the cost function with respect to slope (m), let the partial derivative of the cost function with respect to m be $D_m$

Let’s try applying gradient descent to m and c and approach it step by step:

+ Initially let m = 0 and c = 0. Let L be our learning rate. This controls how much the value of m changes with each step. L could be a small value like 0.0001 for good accuracy.
+ Calculate the partial derivative of the loss function with respect to m, and plug in the current values of x, y, m and c in it to obtain the derivative value D.

![image](https://user-images.githubusercontent.com/99672298/181281437-486385c4-62fe-40c5-9e31-5fbd754f72d9.png)

Dₘ is the value of the partial derivative with respect to m. Similarly lets find the partial derivative with respect to c, Dc :

![image](https://user-images.githubusercontent.com/99672298/181281479-5faf1ffd-cb17-42c0-afd7-9a755a1cef02.png)

+ Now we update the current value of m and c using the following equation:

![image](https://user-images.githubusercontent.com/99672298/181281560-2742cbd0-cc4c-4d80-966f-3697d230ed79.png)

+ We repeat this process until our loss function is a very small value or ideally 0 (which means 0 error or 100% accuracy). The value of m and c that we are left with now will be the optimum values.

### 2.4 Initilization<a class="anchor" id="2.4"></a>
 The Starting point is a randomly selected value of slope and intercept (m & c). It is used as a starting point, to derive the first derivative or slope and then uses the tangent line to calculate the steepness of the slope. Further, this slope will inform the update to the parameters. The slope becomees steeper at eht starting point, but whenever new paramters are generated, then the steepness gradually reduces, the closer we get to the optimal value, the closer the slope of the curve gets to zero. This means that wehtn the slope of the cruve is close to zero, which is claled as that we are near to the point of convergence.
 
### 2.5 Direction and learning Rate<a class="anchor" id="2.5"></a>
 
These two factors are used to determine the partial derivative calculation of future iteration and allow it to the point of convergence or glabal minima.
 
#### Learning Rate 
 
It is defined as the step size taken to reach the minima or lowest point. It has a strong influence on performance. It controls how much the value of m (slope) and c (intercept) changes withe each step.
 
Let "L" be the learning rate
 
+ step_size = $D_c$ x L
+ step_size = $D_m$ x learning_rate(L)
+ New_slope = old_slope - stepsize
+ New_intercept = old_intercept - stepsize
 
+ **Smaller learning rate:** the model will take too much time before it reaches minima might even exhaust the max iterations specified.

![](https://editor.analyticsvidhya.com/uploads/26393ezgif.com-gif-maker%20(5).gif)
+ **Large (Big learning rate):** the steps taken will be large and we can even miss the minima the algorithm may not converge to the optimal point.
+ Because the steps size being too big, it simply jumping back and forth between the convex function of gradient descent. In our case even if we continue till ith iteration we will not reach the local minima.

![f](https://editor.analyticsvidhya.com/uploads/81207ezgif.com-gif-maker%20(4).gif)

+ **The learning rate should be an optimum value.**
+ If the learning rate is high, it results in larger steps but also leads to risk of overshooting the minimum. At the same time, a low (small) learning rate shows the small step size, which compromises overall efficeincy but gives the advantage of more precision.

![f](https://editor.analyticsvidhya.com/uploads/94236ezgif.com-gif-maker%20(7).gif)
![image](https://user-images.githubusercontent.com/99672298/181450732-7108af01-d47c-4688-9438-38948430f16c.png)


#### In summary, Gradient Descent methods steps are
1.) Choose starting point (initialisation).
2.) Calculate gradient at that point.
3.) Make a scaled step in the opposite direction to the gradient
#### The closer we get to the optimal value the closer the slope of the curve gets to 0 and when this happens which means we should take small steps, because we are close to the optimal value and when the case is opposite of this we should take big steps.
4.) We repeat this process until our loss function is a very small value. The value of m and c that we are left with now will be the optimum values.

### 2.6 Challenges with Gradient Descent<a class="anchor" id="2.6"></a>
Gradient Descent works fine in most of the cases, but there are many cases where gradient descent doesn't work properly or fails to work altogether

#### Gradient descent algorithm does not work for all functions. There are specific requirements. A function has to be:
+ **Continues variable and Differentiable**
  First, what does it mean it has to be differentiable?
If a function is differentiable it has a derivative for each point in its domain.not all functions meet these criteria. First, let’s see some examples of functions meeting this criterion:

![image](https://user-images.githubusercontent.com/99672298/181440157-886e7c5a-8d2e-466f-945d-b05fb784ac90.png)

Typical non-differentiable functions have a step a jump a cusp discontinuity of or infinte discontinuity:

![image](https://user-images.githubusercontent.com/99672298/181440208-9bd2365c-f18c-4e5b-bd3c-e2ccd34d2ca6.png)

+ **Next requirement — function has to be convex.**

![image](https://user-images.githubusercontent.com/99672298/181440345-5e8a10c9-0bd0-419f-8fb2-556d5c151384.png)

If the data is arranges in a way that it poses a non-convex optimization problem. It is very difficult to perform optimization problem. It is very difficult to perform
optimization using gradient descent. Gradient Descent only works for problems which have a well defined `convex optimization problems.`

Even when optimizing a convex optimization problem, there may be numerous minimal points. The lowest points is called `Global minima`, where as rest of the points ratger than the global minima are called local minima. Our main aim is go to the global minima avoiding local minima. There is also a **`Saddle Point`** problem. This is a point in the data where the gradient is zero but is not an optimal point. Whenever the slope of the cost function is at zero or just close to zero, the model stops learning further.

![image](https://user-images.githubusercontent.com/99672298/181446138-02faa76f-e083-47f6-b9c7-71a190d9449f.png)

Apart from the global minima, there occurs some scenarios that can show this slope, which is saddle Point and local minima generates the shape similar to the global minima, where the slope of the cost funcstions increase on both side of the current ponts. The name of the saddle pont is taken by that of a horse's saddle.

#### Another challenge

In a deep learning neural network often model is trained with gradient descent and back propagation there can occur two more issues other than lcoal minima and saddle points.

![image](https://user-images.githubusercontent.com/99672298/181450854-cca55386-f974-42e5-9d47-4c674050a5fd.png)

#### Vanishing and Exploding Gradients

These problems occur when the gradient is too large or too small and because of this problem the algorithms do no converge

+ Vanishing Gradients:
Vanishing Gradient occurs when the gradient is smaller than expected. During backpropagation, this gradient becomes smaller that causing the decrease in the learning rate of earlier layers than the later layer of the network. Once this happens, the weight parameters update until they become insignificant.
+ Exploding Gradient:
Exploding gradient is just opposite to the vanishing gradient as it occurs when the Gradient is too large and creates a stable model. Further, in this scenario, model weight increases, and they will be represented as NaN. This problem can be solved using the dimensionality reduction technique, which helps to minimize complexity within the model.

### 2.7 Types of Gradient Descent<a class="anchor" id="2.7"></a>
Based on the error in various training models, the Gradient Descent learning algorithm can be divided into Batch gradient descent, stochastic gradient descent, and mini-batch gradient descent. Let's understand these different types of gradient descent:

+ **Batch Gardient Descent**
Batch gradient descent (BGD) is used to find the error for each point in the training set and update the model after evaluating all training examples. This procedure is known as the training **`epoch.`** In simple words, it is a greedy approach where we have to sum over all examples for each update.

Advantages of Batch gradient descent:
+ It produces less noise in comparison to other gradient descent.
+ It produces stable gradient descent convergence.
+ It is Computationally efficient as all resources are used for all training samples.

+ **Stochastic gradient descent**
