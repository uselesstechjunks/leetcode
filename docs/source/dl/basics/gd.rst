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
	* With fixed learning rate, convergence is not guaranteed.

		* Learning-Rate should be less than (TODO: derive) a function of the condition number to ensure convergent behaviour.
		* Assuming that the LR is set accordingly, it takes infinitely many steps to reach the minimum.
		* Need to set the threshold somewhere.
	* TODO: proof

Learning-Rate Schedule
===================================================================================
.. warning::
	Key idea: Larger LR at the beginning, smaller towards the end.

.. note::
	* Linear decay
	* Exponential decay
	* Power-law decay

Faster Covergence with Momentum
===================================================================================
.. warning::
	Carry on a little bit extra along the previous direction before stopping and changing direction again.

Normal Momentum
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Nesterov Momentum
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Adaptive Learning Rate
===================================================================================
.. warning::
	Allow different LR along different Eigen-direction (making up for Newton's Method without having to compute Hessian)

AdaGrad
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
.. note::
	Keep a running weighted average of the gradient magnitudes to set up the LR

RMSProp
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
.. note::
	Keep more importance to recently computed gradients.

Adam
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
.. note::
	RMSProp with momentum

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

Resources
===================================================================================
.. note::
	* `[Prof Ganesh Ramakrishnan] CS769 Optimization in Machine Learning IIT Bombay 2024 <https://www.cse.iitb.ac.in/%7Eganesh/cs769/>`_

		* `Full Playlist on YT <https://www.youtube.com/playlist?list=PLyo3HAXSZD3yhIPf7Luk_ZHM_ss2fFCVV>`_
		* `Unified all GD variants <https://youtu.be/2QNquvof1WA?list=PLyo3HAXSZD3yhIPf7Luk_ZHM_ss2fFCVV&t=865>`_
	* `[ruder.io] An overview of gradient descent optimization algorithms <https://www.ruder.io/optimizing-gradient-descent/>`_
