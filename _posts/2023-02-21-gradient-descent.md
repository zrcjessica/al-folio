---
layout: post
title:  Gradient descent
date: 2023-02-21 13:41:00
description: A Python-coded implementation of gradient descent for a linear regression on simulated data
tags: machine_learning gradient_descent backpropagation jupyter code
categories: tutorials code
---

Gradient descent is an optimization algorithm used to find the optimal weights for a model. While it is popularly used to tune the weights in neural networks, in this tutorial we will be using it to try to recover the coefficients used to simulate a toy dataset. We will also demonstrate the differences between batch, stochastic, and mini-batch gradient descent and defining some of the relevant terms.

# Simulating the data

We will begin by simulating the data for this example. Our simulated dataset will contain $$n=10000$$ samples using a linear function defined as:

$$
f(x_1, x_2, x_3) = ax_1 + bx_2 + cx_3 + \upepsilon
$$ 

Here, $${x_1,x_2,x_3}$$ represent an input vector of size 3 and $$\upepsilon$$ is noise.

To do this, we will first simulate the coefficients of the model, $$a,b,c$$ by picking 3 random integers between 1 and 10.
 
{% highlight python linenos %}

 import numpy as np
 np.random.seed(1)
 
 # simulate coeffs 
 coeffs = np.random.randint(1,10,size = 3)
 
 print("Coefficients: a = %d, b = %d, c = %d" % (coeffs[0], coeffs[1], coeffs[2]))
{% endhighlight %}

Next, we will generate the inputs and the noise for each sample by randomly sampling from the standard normal distribution, and then calculate the outputs based on the equation defined above.

{% highlight python linenos %}

 # define number of data points
 n = 10000

 # define inputs in dataset
 X = np.random.randn(n,3)

 # define noise
 noise = np.random.randn(n)

 # get outputs for dataset
 y = np.sum(X*coeffs, axis = 1) + noise
{% endhighlight %}  

Just to get a better idea of what the data looks like, let's coerce the dataset we've simulated into a data frame format and take a look at it.

{% highlight python linenos %}
 import pandas as pd

 pd.DataFrame(np.hstack((X, np.reshape(y, (-1,1)))), columns = ["x1","x2","x3","y"])
{% endhighlight %}  

Now let's cover the concepts that will be important for this tutorial.

# Gradient descent
Gradient descent is an optimization method that is popularly used in machine learning to find the best parameters (more often referred to as _weights_ in the case of neural networks) for a given model. It works quite like how it sounds - by following gradients to descend towards the minimum of a **cost function**. Cost functions are a function of the difference between the true and predicted outputs of a given model. There are a number of different cost functions out there. For example, cross entropy cost functions are popularly used for classification problems while squared error cost functions are popular used for regression problems. Conceptually, gradient descent looks something like this:

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/gradient_descent/gradient-descent.jpeg" class="img-fluid" width=600 %}
    </div>
</div>
[Image source](https://saugatbhattarai.com.np/what-is-gradient-descent-in-machine-learning/)
