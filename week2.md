# Machine learning by Andrew Ng
# Week02


## Multiple features

* we can use multiple features to estimate our output
* hypothesis $h_\theta(x) = \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3 + ... + \theta_n x_n$ or $\theta^Tx$
* with  gradient descent we iterate through all the values x


## feature modeling

* make sure that features got the same scale of values
* as gradient descent will be more close to circle
* scale feature so each feature $-1 \leq x_i \leq 1$
	* be within $\pme^2$
	* feature scalning and mean normalisation
	* $\frac{x-\mu}{value_range}$
* learning rate
	* debugging gradient descent - plot $J(\theta)$ over iterations
	* $J(\theta)$ should decrease after each elevation
	* otherards use threshold level
			* it is diffficult to determine value
	* if $J(\theta)$ increase or if it jumps up and down use smaller $\alpha$
	* if $\alpha$ is too small convergence will be slow
	* try choose $\alpha$ by using values x3 (0.001, 0.003, 0.01 ect)
* creating new features by combining other ones
* polynomial regression
		* or cubic function ect
		* if you do this make sure to apply feature scaling

## Normal equation

* solve data analytically instead of gradient descent
* calculate $\frac{dJ(\theta)}{dx} = 0$ for each parameter $\theta$
* to minmalise we use $(X^TX)^{-1}X^Ty$
* should need less iterations than gradient decent
* gradient descent works well even if n is large, while normal eq slow if n is large
* need to compute $A^{-1}$ $O(n^3)$
* normal equation will not work for more complex algorithms
* normal equation non-invertibility (degenerated matrix)
	* use sudo invertion instead (pinv)
	* redudant features
	* too many features
		* remove some features
		* use regularisation

