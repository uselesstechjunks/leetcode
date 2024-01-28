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
  * First-order methods use the first-order Taylor's approximation.
  * It is assumed that the function :math:`f(\mathbf{x})` behaves locally linear around a point :math:`\mathbf{x}\in\mathbb{R}^d`.
  * 

Gradient Descent
----------------------------------------------------------------------------------------

.. note::
  * 

Second-order Methods
========================================================================================
Newton's Method
----------------------------------------------------------------------------------------

****************************************************************************************
Constrained Optimization
****************************************************************************************
.. note::
  * We're interested in finding the minimum :math:`\min_{\mathbf{x}\in S\subseteq \mathbb{R}^d}f(\mathbf{x})`.

