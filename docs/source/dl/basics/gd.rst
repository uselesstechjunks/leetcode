###################################################################################
Gradient Descent
###################################################################################

.. note::
	* Taylor's series
	* Local linear approximation information captured by gradient.
	* Local quadratic approximation information captured by Hessian.
	* First order approximation of the gradient near a point

***********************************************************************************
Nature of the Error Surface near Stationary Point
***********************************************************************************
.. note::
	* Understanding the nature of local stationary point (maxima/minima/saddle point) with the help of Hessian.

.. warning::
	Largest eigenvalue = direction of slowest descent

***********************************************************************************
Gradient Descent
***********************************************************************************
Batch Gradient Descent
===================================================================================
Stochastic Gradient Descent
===================================================================================
Mini-batch Gradient Descent
===================================================================================

***********************************************************************************
Convergence
***********************************************************************************
.. warning::
	* With fixed learning rate - takes infinitely many steps to reach the minimum
	* TODO: proof

Learning-Rate Schedule
===================================================================================
.. note::
	Larger LR at the beginning, smaller towards the end.

Faster Covergence with Momentum
===================================================================================
.. note::
	Carry on a little bit extra along the previous direction before stopping and changing direction again.

Normal Momentum
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Nesterov Momentum
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Adaptive Learning Rate
===================================================================================
.. note::
	Allow different LR along different Eigen-direction (essentially simulating Newton's Method)

AdaGrad
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
RMSProp
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Adam
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

***********************************************************************************
Managing Numerical Issues with Gradients
***********************************************************************************
Weight & Bias Initialisation
===================================================================================
Input normalisation
===================================================================================
Weight normalisation
===================================================================================
Batch Normalisation
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Layer Normalisation
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
