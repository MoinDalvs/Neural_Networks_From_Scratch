## 0.1 Table of Contents<a class="anchor" id="0.1"></a>
1. [Let's talk about Differential Derivative and Partial Derivative](#1)
    - 1.1 [What is Differential?](#1.1)
    - 1.2 [What is Derivative?](#1.2)
    - 1.3 [Partial Derivation (Differentiation)](#1.3)
2. [Gradient Descent](#2)
    - 2.1 [What is Gradient Descent?](#2.1)
    - 2.2 [The Cost function](#2.2)
    - 2.3 [Linear Regression using Gradient Descent](#2.3)
    - 2.4 [Initialization](#2.4)
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

[Table of Content](#0.1)
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

### 2.4 Initialization<a class="anchor" id="2.4"></a>
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
![26 07 2022_16 09 00_REC](https://user-images.githubusercontent.com/99672298/181470460-29085cba-cda1-4238-a4b0-e7591a57b1b2.png)

#### Vanishing and Exploding Gradients

These problems occur when the gradient is too large or too small and because of this problem the algorithms do no converge

![04 07 2022_20 05 34_REC](https://user-images.githubusercontent.com/99672298/181470768-e2af8e65-7b04-4f1b-ab99-f00162cc7930.png)

+ Vanishing Gradients:
Vanishing Gradient occurs when the gradient is smaller than expected. During backpropagation, this gradient becomes smaller that causing the decrease in the learning rate of earlier layers than the later layer of the network. Once this happens, the weight parameters update until they become insignificant.
+ Exploding Gradient:
Exploding gradient is just opposite to the vanishing gradient as it occurs when the Gradient is too large and creates a stable model. Further, in this scenario, model weight increases, and they will be represented as NaN. This problem can be solved using the dimensionality reduction technique, which helps to minimize complexity within the model.

![26 07 2022_16 09 30_REC](https://user-images.githubusercontent.com/99672298/181470534-006f49ed-3ce4-4e1f-a76c-fe5599a75ad9.png)

### 2.7 Types of Gradient Descent<a class="anchor" id="2.7"></a>
Based on the error in various training models, the Gradient Descent learning algorithm can be divided into Batch gradient descent, stochastic gradient descent, and mini-batch gradient descent. Let's understand these different types of gradient descent:

![26 07 2022_16 07 56_REC](https://user-images.githubusercontent.com/99672298/181470298-c33c0863-7914-41ff-90ac-b06fad97cea4.png)

#### **Batch Gardient Descent**
Batch gradient descent (BGD) is used to find the error for each point in the training set and update the model after evaluating all training examples. This procedure is known as the training **`epoch.`** In simple words, it is a greedy approach where we have to sum over all examples for each update.

Advantages of Batch gradient descent:
+ It produces less noise in comparison to other gradient descent.
+ It produces stable gradient descent convergence.
+ It is Computationally efficient as all resources are used for all training samples.

Disadvantages of Batch gradient descent:
+ Sometimes a stable error gradient can lead to a local minima and unlike stochastic gradient descent no noisy steps are there to help get out of the local minima
+ The entire training set can be too large to process in the memory due to which additional memory might be needed
+ Depending on computer resources it can take too long for processing all the training samples as a batch

#### **Stochastic gradient descent**
Stochastic gradient descent (SGD) is a type of gradient descent that runs one training example per iteration. Or in other words, it processes a training epoch for each example within a dataset and updates each training example's parameters one at a time. As it requires only one training example at a time, hence it is easier to store in allocated memory. However, it shows some computational efficiency losses in comparison to batch gradient systems as it shows frequent updates that require more detail and speed. 

Further, due to frequent updates, it is also treated as a noisy gradient. However, sometimes it can be helpful in finding the global minimum and also escaping the local minimum.

Advantages of Stochastic gradient descent:

In Stochastic gradient descent (SGD), learning happens on every example, and it consists of a few advantages over other gradient descent.

+ It is easier to allocate in desired memory.
+ It is relatively fast to compute than batch gradient descent.
+ It is more efficient for large datasets.

Disadvantages of Stochastic Gradient descent:

+ Updating the model so frequently is more computationally expensive than other configurations of gradient descent, taking significantly longer to train models on large datasets.
+ The frequent updates can result in a noisy gradient signal, which may cause the model parameters and in turn the model error to jump around (have a higher variance over training epochs).
+ The noisy learning process down the error gradient can also make it hard for the algorithm to settle on an error minimum for the model.

#### **MiniBatch Gradient Descent:**
Mini Batch gradient descent is the combination of both batch gradient descent and stochastic gradient descent. It divides the training datasets into small batch sizes then performs the updates on those batches separately. 

Splitting training datasets into smaller batches make a balance to maintain the computational efficiency of batch gradient descent and speed of stochastic gradient descent. Hence, we can achieve a special type of gradient descent with higher computational efficiency and less noisy gradient descent.

Advantages of Mini Batch gradient descent:

+ It is easier to fit in allocated memory.
+ It is computationally efficient.
+ It produces stable gradient descent convergence.

### Variants of Gradient Descent Algorithm

#### **Vanilla Gradient Descent**
This is the simplest form of gradient descent technique. Here, vanilla means pure / without any adulteration. Its main feature is that we take small steps in the direction of the minima by taking gradient of the cost function.

Let’s look at its pseudocode.
+ update = learning_rate * gradient_of_parameters
+ parameters = parameters - update

Here, we see that we make an update to the parameters by taking gradient of the parameters. And multiplying it by a learning rate, which is essentially a constant number suggesting how fast we want to go the minimum. Learning rate is a hyper-parameter and should be treated with care when choosing its value.

![image](https://user-images.githubusercontent.com/99672298/181469887-dd53f973-c13c-42d9-bcd8-a32f5cfc2f84.png)

#### **Gradient Descent with Momentum**
Here, we tweak the above algorithm in such a way that we pay heed to the prior step before taking the next step.

Here’s a pseudocode.
+ update = learning_rate * gradient
+ velocity = previous_update * momentum
+ parameter = parameter + velocity – update

Here, our update is the same as that of vanilla gradient descent. But we introduce a new term called velocity, which considers the previous update and a constant which is called momentum.

#### **ADAGRAD**
ADAGRAD uses adaptive technique for learning rate updation. In this algorithm, on the basis of how the gradient has been changing for all the previous iterations we try to change the learning rate.

Here’s a pseudocode
+ grad_component = previous_grad_component + (gradient * gradient)
+ rate_change = square_root(grad_component) + epsilon
+ adapted_learning_rate = learning_rate * rate_change
+ update = adapted_learning_rate * gradient
+ parameter = parameter – update
#### **ADAM**
ADAM is one more adaptive technique which builds on adagrad and further reduces it downside. In other words, you can consider this as momentum + ADAGRAD.

Here’s a pseudocode.

+ adapted_gradient = previous_gradient + ((gradient – previous_gradient) * (1 – beta1))
+ gradient_component = (gradient_change – previous_learning_rate)
+ adapted_learning_rate =  previous_learning_rate + (gradient_component * (1 – beta2))
+ update = adapted_learning_rate * adapted_gradient
+ parameter = parameter – update
+ Here beta1 and beta2 are constants to keep changes in gradient and learning rate in check

### The problem with gradient descent is that the weight update at a moment (t) is governed by the learning rate and gradient at that moment only. It doesn't take into account the past steps.

It leads to the following problems:

1.) The Saddle point Problem (plateau)

![image](https://user-images.githubusercontent.com/99672298/181585064-594d61de-cc66-459b-ad54-dd082396a133.png)

2.) Noisy - The path followed by the Gradient descent

![image](https://user-images.githubusercontent.com/99672298/181584434-6b0f391e-a021-4f1e-88db-a4dd72e68839.png)
![](https://miro.medium.com/max/992/1*t-kykynrtQ0olmFeNgIB0w.gif)

A problem with the gradient descent algorithm is that the progression of the search space based on the gradient. For example, the search may progress downhill towards the minima, but during this progression, it may move in another direction, even uphill, this can slow down the progress of the search.

![image](https://user-images.githubusercontent.com/99672298/181585689-2b0cd426-4425-413e-9f89-bc9c4d646c05.png)

Another prblem let's assume the intial weight of the network under consideration corresponds to point 'P1' (Look at the below figure)  with gradient descent, the loss function decreases rapidly along the slope 'P1' to 'P2' as the gradient along this slope is high. But as soon as it reaches 'P2' the gradient becomes very low. The weight updates around 'P2' is very small, Even after many iterations, the cost moves very slowly before getting stuck at a point where the gradient eventually becomes zero. In the below case as you can see in the figure, ideally cost should have moved to the global minima point 'P3' but because the gradient disappear at point 'B', we are stuck

![image](https://user-images.githubusercontent.com/99672298/181586827-0e55e9f5-9372-4344-8dfb-b44b32d8ce17.png)

One approach to the problem is to add history to the parameter update equation based on the gradient encountered in the previous updates. 

"If I am repeatedly being asked to move in the same direction then I should probably gain some confidence and start taking bigger steps in that direction. Just as a ball gains momentum while rolling down a slope." This changes is based on the metaphor of momentum from physics where accelaration in a direction can be acculmulated from past updates.

### Momentum
Momentum is an extension to the gradient descent optimization algorithm, often referred to as gradient descent with momentum.
#### How come momentum is going to help us fix our earlier two problems?

#### 1.) Saddle point problem 

#### 2.) Noisy path followed

![image](https://user-images.githubusercontent.com/99672298/181620407-5b9bbd7f-7f51-4747-b644-fcc002586100.png)

Now, Imagine you have a ball rolling from point A. The ball starts rolling down slowly and gathers some momentum across the slope AB. When the ball reaches point B, it has accumulated enough momentum to push itself across the plateau region B and finally following slope BC to land at the global minima C.

#### How can this be used and applied to Gradient Descent?
We can use a moving average over the past gradients. In regions where the gradient is high like AB, weight updates will be large. Thus, in a way we are gathering momentum by taking a moving average over these gradients.

But there is a problem with this method, it considers all the gradients over iterations with equal weightage. The gradient at t=0 has equal weightage as that of the gradient at current iteration t. We need to use some sort of weighted average of the past gradients such that the recent gradients are given more weightage.

This can be done by using an Exponential Moving Average(EMA). An exponential moving average is a moving average that assigns a greater weight on the most recent values.

**`Momentum can be interpreted as a ball rolling down a nearly horizontal incline. The ball naturally gathers momentum as gravity causes it to accelerate.`**

This to some amount addresses our second problem. Gradient Descent with Momentum takes small steps in directions where the gradients oscillate and take large steps along the direction where the past gradients have the same direction(same sign). This problem with momentum is that acceleration can sometimes overshoot the search and run past our goal other side of the minima valley. While making a lot og U-turns before finally converging.

#### Can we do something to reduce the oscillation/ U-turns / overshooting the minima?
#### YES!

### Nesterov Momentum

![image](https://user-images.githubusercontent.com/99672298/181622949-fd237ff6-6edb-49d9-b00f-5667c348e2b4.png)

A limitation of gradient descent is that it can get stuck in flat areas or bounce around if the objective function returns noisy gradients. Momentum is an approach that accelerates the progress of the search to skim across flat areas and smooth out bouncy gradients.

In some cases, the acceleration of momentum can cause the search to miss or overshoot the minima at the bottom of basins or valleys. Nesterov momentum is an extension of momentum that involves calculating the decaying moving average of the gradients of projected positions in the search space rather than the actual positions themselves.

While `Momentum` first computes the current gradient and then take a big jump in the direction of the updated accumulated gradient, where **`Nesterov`** first makes a big jump in the direction of the previous accumulated gradient, measures the gradient and then complete Nesterov update. This anticipatory updates  prevents us from going to fast and results in increased responsiveness and reduces oscillation.

This has the effect of harnessing the accelerating benefits of momentum whilst allowing the search to slow down when approaching the optima and reduce the likelihood of missing or overshooting it.

**`Look ahead before you leap`**
### Adaptive Gradient Descent (ADAGrad)

A problem with the gradient descent algorithm is that the step size (learning rate) is the same for each variable or dimension in the search space. It is possible that better performance can be achieved using a step size that is tailored to each variable, allowing larger movements in dimensions with a consistently steep gradient and smaller movements in dimensions with less steep gradients.

AdaGrad is designed to specifically explore the idea of automatically tailoring the step size for each dimension in the search space.

An internal variable is then maintained for each input variable that is the sum of the squared partial derivatives for the input variable observed during the search.

This sum of the squared partial derivatives is then used to calculate the step size for the variable by dividing the initial step size value (e.g. hyperparameter value specified at the start of the run) divided by the square root of the sum of the squared partial derivatives.

One of Adagrad’s main benefits is that it eliminates the need to manually tune the learning rate. But, Adagrad’s main weakness is its accumulation of the squared gradients in the denominator: Since every added term is positive, the accumulated sum keeps growing during training. This in turn causes the learning rate to shrink and eventually become infinitesimally small, at which point the algorithm is no longer able to acquire additional knowledge. This has the effect of stopping the search too soon, before the minima can even be located.

**`Adaptive Gradients, or AdaGrad for short, is an extension of the gradient descent optimization algorithm that allows the step size in each dimension used by the optimization algorithm to be automatically adapted based on the gradients seen for the variable (partial derivatives) seen over the course of the search.`**

## Root Mean Squared Propogation (RMSProp)



### Overview:
Gradient descent refers to a minimization optimization algorithm that follows the negative of the gradient downhill of the target function to locate the minimum of the function.

A downhill movement is made by first calculating how far to move in the input space, calculated as the steps size (called alpha or the learning rate) multiplied by the gradient. This is then subtracted from the current point, ensuring we move against the gradient, or down the target function.

The steeper the objective function at a given point, the larger the magnitude of the gradient, and in turn, the larger the step taken in the search space. The size of the step taken is scaled using a step size hyperparameter.
+ Step Size (alpha): Hyperparameter that controls how far to move in the search space against the gradient each iteration of the algorithm.
+ Gradient descent is an optimization algorithm that follows the negative gradient of an objective function in order to locate the minimum of the function.

A downhill movement is made by first calculating how far to move in the input space, calculated as the steps size (called alpha or the learning rate) multiplied by the gradient. This is then subtracted from the current point, ensuring we move against the gradient, or down the target function.

x(t+1) = x(t) – step_size * f'(x(t))

The steeper the objective function at a given point, the larger the magnitude of the gradient, and in turn, the larger the step taken in the search space. The size of the step taken is scaled using a step size hyperparameter.

Step Size (alpha): Hyperparameter that controls how far to move in the search space against the gradient each iteration of the algorithm.
If the step size is too small, the movement in the search space will be small, and the search will take a long time. If the step size is too large, the search may bounce around the search space and skip over the optima.
