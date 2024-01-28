########################################################################################
Convex Optimisation
########################################################################################
.. note::
	* The functions we're dealing with are of the form :math:`f(\mathbf{x}):\mathbb{R}^d\mapsto\mathbb{R}`
	* Taylor expansion of the function around a fixed point :math:`\mathbf{x}_0\in\mathbb{R}^d` if the function is infinitely differentiable

		.. math:: f(\mathbf{x})=f(\mathbf{x}_0)+\langle(\nabla_\mathbf{x}f|_{\mathbf{x}=\mathbf{x}_0}), (\mathbf{x}-\mathbf{x}_0)\rangle+(\mathbf{x}-\mathbf{x}_0)^T(\nabla^2_\mathbf{x}f|_{\mathbf{x}=\mathbf{x}_0})(\mathbf{x}-\mathbf{x}_0)+\cdots
	* We use the notation :math:`\mathbf{g}:\mathbb{R}^d\mapsto\mathbb{R}^d:=\nabla_\mathbf{x}f` for the gradient and :math:`\mathbf{g}:\mathbb{R}^d\mapsto\mathbb{R}^d:=\nabla_\mathbf{x}f` for the Hessian.
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
	* Here, the scalar math:`\varepsilon>0` is a parameter that determines the stepsize.
	* We evaluate the function value at this new point as

		.. math:: f(\mathbf{x}_{t+1})=f(\mathbf{x}_t)+\mathbf{g}(\mathbf{x}_t)^T(\mathbf{x}_{t+1}-\mathbf{x}_t)+\frac{1}{2!}(\mathbf{x}_{t+1}-\mathbf{x}_t)^T\mathbf{H}(\mathbf{x}_t)(\mathbf{x}_{t+1}-\mathbf{x}_t)\cdots=f(\mathbf{x}_t)-\varepsilon\mathbf{g}(\mathbf{x}_t)^T\mathbf{g}(\mathbf{x}_t)+\frac{1}{2!}\varepsilon^2\mathbf{g}(\mathbf{x}_t)^T\mathbf{H}(\mathbf{x}_t)\mathbf{g}(\mathbf{x}_t)+\cdots

Second-order Methods
========================================================================================
Newton's Method
----------------------------------------------------------------------------------------

****************************************************************************************
Constrained Optimization
****************************************************************************************
.. note::
	* We're interested in finding the minimum :math:`\min_{\mathbf{x}\in S\subseteq \mathbb{R}^d}f(\mathbf{x})`.

