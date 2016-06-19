# Machine learning by Andrew Ng
# Week01


Machine learning is everywhere:

* search
* NPL, handwriting recognition, self-customising programmes

###Supervised learning

Most common type. We train computer with a right answer given. Used for the following problems:

* regression problem - fit model to our data
* classification problem - choose (0,1), divide data into two groups
	* - oo o  oo o xo xxx o x ->
	* SVM approach
* use multiple features to divide data into multiple groups

###Un-supervised learning

We are relaying on algorithm to learn how the data works

* clustering algorithm
	* market segmentation
	* gene types
	* organise computing clusters
	* social network analysis
	* astronomical data analysis

* cocktail party problem - recording voice at different microphones
	* we can solve this problem with one line of code
	* [W,s,v] =  svd((repmat(sum(x.*x,1),size(x,1),1).*x)*x');

## Cost function

* m - no of training examples
* Xs - input variable/features
* Ys - output variable

* We use training sets to train model $f(x^i)=y^i$ where $(x^i,y^i)$ known.
* We are using hypothesis functions to map x (features) to y (output)
	* For example linear regression with one variable (univariate linear regression) would be $H_0(x)= \theta_0 + \theta_1 x$
* To train we are defining our cost function. Example here is LSA - called here squared error function.
	* for linear model we would use $\tau (\theta_0 , \theta_1)= \frac{1}{n}\textstyle \sum_{i=1}^{m}(h_0(x^i)-y^i)^2$
	* it is also called sq error function, so it is L2 function
* hypothesis is function of x, cost function is function of parameter $\theta_i$
	* while the hypothesis function is linear, the cost function is always a [convex function](https://en.wikipedia.org/wiki/Convex_function), so we will look for minimum function to solve for it
	* with two parameters this is bow-shape function.
* To mineralise cost function, or to find minimum
	* we look at direction for steepest descent
	* we will repeat $\theta_j := \theta_j - \alpha \frac{\partial J( \theta_0, \theta_1) }{\partial \theta_j}$ until convergence
	* we are using $\alpha$ to define how quick we learn - we have to balance between speed and overshooting minimum
	* we can end up with local minimum, and then our cost function will get stuck as $\alpha \frac{\partial J( \theta_0, \theta_1) }{\partial \theta_j}=0$
	* for the specific choice of cost function, used in linear regression, there are no local optima other than the global optimum
	* as we approach minimum, our derivative get smaller so we can keep $\alpha$ constant
	* With multiple values make sure you do **simultaneous update** all variables - always use last epoch values, that is \theta_0^2, \theta_1^2 := f(\theta_0^0, \theta_1^0)
	* batch gradient descent - on each step we use of all training examples, is the most basic approach


