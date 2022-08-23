### `Read and catch up on content:`
- [Gradient Descent article](https://github.com/MoinDalvs/Gradient_Descent_For_beginners/blob/main/README.md) :books:

## 0.1 Table of Contents<a class="anchor" id="0.1"></a>
1. [Let's talk about Neural Networks.](#1)
    - 1.1 [Some Basic Concepts Related to Neural Networks](#1.1)
    - 1.2 [What is Derivative?](#1.2)
    - 1.3 [Partial Derivation (Differentiation)](#1.3)
2. [Gradient Descent](#2)
    - 2.1 [What is Gradient Descent?](#2.1)
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

Supervised learning can be used on both structured and unstructered data. For example of a structured data let's take house prices prediction data, let's also assume the given data tells us the size and the number of bedrooms. This is what is called a well structured data, where each features, such as the size of the house, the number of bedrooms, etc has a very well defined meaning.
	
In contrast, unstructured data refers to things like audio, raw audio, or images where you might want to recognize what's in the image or text (like object detection and OCR Optical character recognition). Here, the features might be the pixel values in an image or the individual words in a piece of text. It's not really clear what each pixel of the image represents and therefore this falls under the unstructured data.
	
Simple machine learning algorithms work well with structured data. But when it comes to unstructured data, their performance tends to take a quite dip. This where Neural Netowrks does their magic , which have proven to be so effective and useful. They perform exceptionally well on structured data.

![image](https://user-images.githubusercontent.com/99672298/186084176-5830c906-fb90-4fb6-ba05-66496934f10b.png)

As the Amount of data icreases, the performance of data learning algorithms, like SVM and logistic regression, does not improve infacts it tends to plateau after a certain point. In the case of neural networks as the amount of data increases the performance of the model increases.

### 1.1 Some Basic Concepts Related to Neural Networks<a class="anchor" id="1.1"></a>
