########################################################################################
Convex Optimisation
########################################################################################
.. note::
	* The functions we're dealing with are of the form :math:`f(\mathbf{x}):\mathbb{R}^d\mapsto\mathbb{R}`
	* Taylor expansion of the function around a fixed point :math:`\mathbf{x}_0\in\mathbb{R}^d` if the function is infinitely differentiable

		.. math:: f(\mathbf{x})=f(\mathbf{x}_0)+\langle(\nabla_\mathbf{x}f|_{\mathbf{x}=\mathbf{x}_0}), (\mathbf{x}-\mathbf{x}_0)\rangle+(\mathbf{x}-\mathbf{x}_0)^T(\nabla^2_\mathbf{x}f|_{\mathbf{x}=\mathbf{x}_0})(\mathbf{x}-\mathbf{x}_0)+\cdots
	* We use the notation :math:`\mathbf{g}:\mathbb{R}^d\mapsto\mathbb{R}^d:=\nabla_\mathbf{x}f` for the gradient and :math:`\mathbf{H}:\mathbb{R}^d\mapsto\mathbb{R}^d:=\nabla_\mathbf{x}f` for the Hessian.
	* The Taylor expansion is then written as

		.. math:: f(\mathbf{x})=f(\mathbf{x}_0)+\mathbf{g}(\mathbf{x}_0)^T(\mathbf{x}-\mathbf{x}_0)+(\mathbf{x}-\mathbf{x}_0)^T\mathbf{H}(\mathbf{x}_0)(\mathbf{x}-\mathbf{x}_0)+\cdots

****************************************************************************************
Unconstrained Optimization
****************************************************************************************
.. note::
	* We're interested in finding the minimum :math:`\min_{\mathbf{x}\in\mathbb{R}^d}f(\mathbf{x})`.

First-order Methods
========================================================================================
.. note::
	* First-order methods use the first-order Taylor's approximation
	* It is assumed that the function :math:`f(\mathbf{x})` behaves locally linear around a point :math:`\mathbf{x}\in\mathbb{R}^d` and therefore can be approximated by

		.. math:: f(\mathbf{x})\approx f(\mathbf{x}_0)+\mathbf{g}(\mathbf{x}_0)^T(\mathbf{x}-\mathbf{x}_0)

Gradient Descent
----------------------------------------------------------------------------------------
.. note::
	* [TODO: proof] The direction of the gradient gives the direction of steepest ascent (i.e. the opposite of gradient provides the direction of steepest descent).
	* We start with an abritrary starting point :math:`\mathbf{x}_t=\mathbf{x}_0\in\mathbb{R}^d`.
	* We take a small step along the direction of the gradient as

		.. math:: \mathbf{x}_{t+1}=\mathbf{x}_t-\varepsilon\mathbf{g}(\mathbf{x}_t)
	* Here, the scalar :math:`\varepsilon>0` is a parameter that determines the stepsize.
	* We evaluate the function value at this new point as

		.. math:: f(\mathbf{x}_{t+1})=f(\mathbf{x}_t)+\mathbf{g}(\mathbf{x}_t)^T(\mathbf{x}_{t+1}-\mathbf{x}_t)+\frac{1}{2!}(\mathbf{x}_{t+1}-\mathbf{x}_t)^T\mathbf{H}(\mathbf{x}_t)(\mathbf{x}_{t+1}-\mathbf{x}_t)\cdots=f(\mathbf{x}_t)-\varepsilon\mathbf{g}(\mathbf{x}_t)^T\mathbf{g}(\mathbf{x}_t)+\frac{1}{2!}\varepsilon^2\mathbf{g}(\mathbf{x}_t)^T\mathbf{H}(\mathbf{x}_t)\mathbf{g}(\mathbf{x}_t)+\cdots

Code example for Linear Regression
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
.. code-block:: python

	import numpy as np
	import matplotlib.pyplot as plot
	import pandas as pd
	import seaborn as seaborn
	
	# create the function as linear with random normal noise
	def define_function(d):
		return np.random.randn(d)

	def create_dataset(w, noise_sigma, N=1000):
		d = w.shape[0]
		X = [np.random.rand(d).tolist() for i in np.arange(N)] # N rows and d columns
		return pd.DataFrame([(*x, w.dot(x) + np.random.randn() * noise_sigma) for x in X])

	def compute_loss(X, y, wt):
		return np.linalg.norm(y-X*wt)

	def compute_gradient(X, y, wt):
		return -2*X.T*(y-X*wt)

	def gradient_descent(X, y, lr=0.0001, eps=1e-5, max_iter=100):
		wt = np.matrix([[np.random.randn()], [np.random.randn()]])
		loss = compute_loss(X, y, wt)
		i = 0
		print(f'iter={i}')
		print(f'wt={wt}')
		print(f'loss={loss}')
		loss_values = []
    
		while loss > eps and i < max_iter:
			print(f'iter={i}')
			g = compute_gradient(X, y, wt)
			wt = wt - lr*g
			loss = compute_loss(X, y, wt)
			i = i+1
			print(f'wt={wt}')
			print(f'loss={loss}')
			loss_values.append([loss])
        
		return wt, loss_values

	w = define_function(2)
	df = create_dataset(w, noise_sigma=0.01, N=1000)
	X = np.asarray(df.iloc[:,:2])
	y = np.asarray(df.iloc[:,2])

	# direct estimator from least square
	w_hat = (np.linalg.inv(X.T * X)) * X.T * y

	X = np.asmatrix(X)
	y = np.asmatrix(y).T
	w_gd, loss_values = gradient_descent(X, y, lr=0.001, eps=1e-5, max_iter=50)

	plot.plot(np.arange(len(loss_values)), loss_values)
	plot.show()

Second-order Methods
========================================================================================
Newton's Method
----------------------------------------------------------------------------------------
.. code-block:: python

	def compute_loss(X, y, wt):
		return np.linalg.norm(y-X*wt)

	def compute_gradient(X, y, wt):
		return -2*X.T*(y-X*wt)

	def compute_hessian(X):
		return 2*X.T*X

	def newton_method(X, y, eps=1e-5, max_iter=5):
		wt = np.matrix([[np.random.randn()], [np.random.randn()]])
		loss = compute_loss(X, y, wt)
		i = 0
		print(f'iter={i}')
		print(f'wt={wt}')
		print(f'loss={loss}')
		loss_values = []
    
		while loss > eps and i < max_iter:
			print(f'iter={i}')
			g = compute_gradient(X, y, wt)
			H = compute_hessian(X)
			wt = wt - np.linalg.inv(H)*g
			loss = compute_loss(X, y, wt)
			i = i+1
			print(f'wt={wt}')
			print(f'loss={loss}')
			loss_values.append([loss])
        
		return wt, loss_values

	w_newt, loss_values_newt = newton_method(X, y, eps=1e-5, max_iter=2)
	plot.plot(np.arange(len(loss_values_newt)), loss_values_newt)
	plot.show()

Code example for Linear Regression
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

****************************************************************************************
Constrained Optimization
****************************************************************************************
.. note::
	* We're interested in finding the minimum :math:`\min_{\mathbf{x}\in S\subseteq \mathbb{R}^d}f(\mathbf{x})`.

