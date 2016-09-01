Machine learning by Andrew Ng - Week03
===


## Classification

* Using linear regression
	* we predict {0,1}
	* we can use liner regression to solve this problem by defining threshold, for ex if $h_\theta(x) \geq 0.5$ y=1 else y=0
	* outlayer hinge point will create a problem as it will shift threshold and create miss classification
	*  $h_\theta(x)$ can also be >1 and <0 which created another problem
* solution here is logistic regression - this is classification algorythm
	* we will get value $0 \leq h_\theta(x) \leq 1$
	* $h_\theta(x) = \frac{1}{1+e^{-\theta^Tx}}$
		* $g(z) = \frac{1}{1+e^{-z}}$ - sygmoid (logistic)
	* interpretation - $h\theta(x) is the probability of y=1 output given input x as parametrised by $\theta$, otherwords $h_\theta(x) = P(y=1 | x;\theta)$

## Decision Boundary

* $h_\theta(x) \geq 0.5$ if $\theta^Tx \geq 0$
* decision boundary will be the line breaking the groups
* parameters $\theta$ define decision boundary
* training set is used to find optimal parameters $\theta$
* high order polinomial features will give much complex shapes

## cost function

* training set of m examples is used
	* we use linear regression $J(\theta)$ to find parameters $\theta$
	* otherwise we can use cost function $J(\theta) = \frac{\sum cost(h_\theta(x^{(i)},y) }{m} $ so $ost(h_\theta(x^{(i)},y) = \frac{1}{2}(h_\theta(x)-y)^2
		* if we use above $J(\theta)$ function for logistic regression we will end up with non-convex function, with many local minima making our problem very complicated
		* we need to find different $J(\theta)$ that is a convex functon
		* we can re-define our function as combination of two -log functions

# Gradient descent

* $cost(h_\theta(x),y) = -y  log(h_\theta(x)) -  (1-y)  log(1-h_\theta(x))$

cae6fe2986