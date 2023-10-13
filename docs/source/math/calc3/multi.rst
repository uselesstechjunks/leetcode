##########################################################
Multivariable Calculus
##########################################################
We consider functions from :math:`\mathbb{R}^n` to :math:`\mathbb{R}^m`.

**********************************************************
Different Forms of Multivariable Functions
**********************************************************
Parametric Surface
==========================================================
.. note::
	* If :math:`n=1` and :math:`m > 1` then the functions :math:`f:\mathbb{R}\mapsto\mathbb{R}^m` are known as parametric sufraces.
	* Example: :math:`f(x)=(x, x^2)`

Scalar field
==========================================================
.. note::
	* If :math:`n> 1` and :math:`m=1` then the functions :math:`f:\mathbb{R}^n\mapsto\mathbb{R}` are known as scalar fields.
	* Example: :math:`f(x,y)=xy`

.. image:: ../../img/1.png
  :width: 400
  :alt: Scalar Field of f(x,y)=xy

Vector field
==========================================================
.. note::
	* If :math:`n> 1` and :math:`m> 1` then the functions :math:`\mathbf{f}:\mathbb{R}^n\mapsto\mathbb{R}^m` are known as vector fields.
	* Example: :math:`f(x,y)=(x^2,\sin(y),x+y)^\top`

**********************************************************
Directional Derivative
**********************************************************
We have a function :math:`\mathbf{f}`, from an open set :math:`E\in\mathbb{R}^n` into :math:`\mathbb{R}^m`. We want to find a proper definition of derivative of :math:`\mathbf{f}` at some point :math:`\mathbf{x}\in E`.

.. note::
	* If the domain was in :math:`\mathbb{R}`, then there is a single direction along which we can approach a point :math:`x\in\mathbb{R}`.
	* Now that the domain is in :math:`\mathbb{R}^n`, there are infinite directions along which we can approach a point :math:`\mathbf{x}\in\mathbb{R}`.
	* Along each such direction, the rate-of-change in the function can be different.
	* Therefore, extending the idea of single variable derivative, we fix our direction along a particular vector :math:`\mathbf{u}\in\mathbb{R}`.
	* For some :math:`h> 0`, we assume an :math:`h\cdot||\mathbf{u}||`-ball around :math:`\mathbf{x}`, and define the ratio

		.. math:: \frac{\mathbf{f}(\mathbf{x}+h\cdot\mathbf{u})-\mathbf{f}(\mathbf{x})}{h}
	* We define a version of derivative as :math:`\mathbf{f}'(\mathbf{x}; \mathbf{u})=\lim\limits_{h\to 0}\frac{\mathbf{f}(\mathbf{x}+h\cdot\mathbf{u})-\mathbf{f}(\mathbf{x})}{h}`

.. note::
	* If :math:`\mathbf{u}` happens to a unit-vector, then our open ball is :math:`B_h(\mathbf{x})`.
	* In this case, :math:`\mathbf{f}'(\mathbf{x}; \mathbf{u})` is called the directional derivative along :math:`\mathbf{u}`.

Partial Derivative
==========================================================

**********************************************************
Total Derivative
**********************************************************

Continuously Differentiable Functions
=========================================================

Gradient
==========================================================

Jacobian
==========================================================

**********************************************************
Higher Order Derivative
**********************************************************

Hessian
==========================================================

**********************************************************
Useful Results
**********************************************************

.. csv-table:: Table for derivatives
	:header: "Scalar derivative", "Vector derivative"
	:align: center

	:math:`f(x)\to\frac{\mathop{d}}{\mathop{dx}}f(x)`, :math:`f(\mathbf{x})\to\frac{\mathop{d}}{\mathop{d\mathbf{x}}}f(\mathbf{x})`
	:math:`bx\to b`, :math:`\mathbf{x}^\top\mathbf{b}/\mathbf{b}^\top\mathbf{x}\to \mathbf{b}`
	:math:`ax\to a`, :math:`\mathbf{x}^\top\mathbf{A}\to \mathbf{A}`
	:math:`a^2x\to a^2`, :math:`\mathbf{a}^\top\mathbf{X}^\top\mathbf{a}/\mathbf{a}^\top\mathbf{X}\mathbf{a}\to \mathbf{a}\mathbf{a}^\top`
	:math:`abx\to ab`, :math:`\mathbf{a}^\top\mathbf{X}\mathbf{b}\to \mathbf{a}\mathbf{b}^\top`	
	:math:`abx\to ab`, :math:`\mathbf{a}^\top\mathbf{X}^\top\mathbf{b}\to \mathbf{b}\mathbf{a}^\top`
	:math:`x^2\to 2x`, :math:`\mathbf{x}^\top\mathbf{x}\to 2\mathbf{x}`
	:math:`ax^2\to 2ax`, :math:`\mathbf{x}^\top\mathbf{A}\mathbf{x}\to (\mathbf{A}+\mathbf{A}^\top)\mathbf{x}`
	:math:`abx^2\to 2abx`, :math:`\mathbf{b}^\top\mathbf{X}^\top\mathbf{X}\mathbf{a}\to \mathbf{X}(\mathbf{a}\mathbf{b}^\top+\mathbf{b}\mathbf{a}^\top)`

.. seealso::
	Plethora of useful results: `Matrix Cookbook <https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf>`_
