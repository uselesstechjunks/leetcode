#########################################
Pre-requisite
#########################################

***********************************************
Binomial and Multinomial Theorem
***********************************************

(MISSISSIPPI problem): Number of ways to arrange the letters is :math:`\frac{11!}{1!4!4!2!}` (M=1,I=4,S=4,P=2).

.. note::
  * There are items of :math:`r` kinds, and within each kind they are indistinguishable. Say, we have :math:`k_i` items of :math:`x_i` kind, and so on. The number of ways of arranging a total of :math:`n` such items (i.e. :math:`\sum_{i=1}^r k_i=n`)

    .. math::
      {n\choose k_1\cdots k_r}=\frac{n!}{k_1!\cdots k_r!}
  * This is also the number of ways a set of :math:`n` indistinguishable items can be partitioned into :math:`r` subsets. To see why, think about the process where we do the partitioning first (say, :math:`C` is the number of ways we can partition) and then we rearrange each subset (:math:`k_i!` ways for subset :math:`i`) put the items in a sequence. The result is just a rearrangement of the entire sequence of :math:`n` items.

Binomial theorem: 
==============================================
* For evaluating the expression :math:`(x+y)^n=(x+y)\cdots(x+y)` (:math:`n`-times), the process involves choosing either :math:`x` or :math:`y` from each of the :math:`n`-terms. 
* We can choose :math:`x` from 0 terms to :math:`n` terms. Every time we choose :math:`x` from :math:`k` such terms, we're left with no choice but to take :math:`y` from  the remaining :math:`(n-k)` terms.
* The number of ways we can choose :math:`x` from any :math:`k` of such terms is :math:`{n\choose k}`. This is how many times we have :math:`x^k y^{n-k}`.
* Therefore

  .. math::
   (x+y)^n=\sum_{k=0}^n {n\choose k} x^k y^{n-k}

.. tip::
  * :math:`\sum_{k=0}^n {n\choose k}=2^n` (setting :math:`x=1` and :math:`y=1`).
  * :math:`\sum_{k=0}^n (-1)^k {n\choose k}=0` (setting :math:`x=-1` and :math:`y=1`).
  * (Vandermonde Identity) :math:`{m+n\choose k}=\sum_{i=0}^k {m\choose i}{n\choose k-i}`.
  * :math:`{n\choose k}={n-1\choose k-1}+{n-1\choose k}`.

Multinomial theorem:
==============================================
* For evaluating the expression :math:`(x_1+x_2+\cdots+x_r)^n=(x_1+x_2+\cdots+x_r)\cdots(x_1+x_2+\cdots+x_r)` (:math:`n`-times), the process involves choosing :math:`x_1` from :math:`k_1` such terms, :math:`x_2` from :math:`k_2` such terms, and so on.
* Regardless of how we choose, the :math:`\sum_{i=1}^r k_i=n`.
* For each values of :math:`0\leq k_1,k_2,\cdots k_r\leq n`, this correspond to the to term :math:`x_1^{k_1}\cdots x_r^{k_r}`.
* Therefore

  .. math::
   (x_1+x_2+\cdots+x_r)^n=\sum_{\sum_{i=1}^r k_i=n} {n\choose k_1\cdots k_r} x_1^{k_1}\cdots x_r^{k_r}

***********************************************
Calculus
***********************************************
Differentiation
==============================================
.. note::
 * [Product rule] :math:`\left(u(x)\cdot v(x)\right)'=u'(x)\cdot v(x)+u(x)\cdot v'(x)`
 * [Chain rule] :math:`\left(f(g(x))\right)'=f'(g(x))\cdot g'(x)`

Integration
==============================================
Integration by parts:
-------------------------------
Let :math:`u(x)` and :math:`v(x)` be two functions. We want to find out the integral of the product, :math:`\int u(x)\cdot v(x) dx`.

.. tip::
 * To derive this formula, it becomes easier if we consider :math:`w(x)=\int v(x) dx` (:math:`w'(x)=v(x)`) and consider :math:`g(x)=u(x)\cdot w(x)`.
 * Taking derivatives on both sides :math:`g'(x)=u'(x)\cdot w(x)+u(x)\cdot w'(x)` which gives

  .. math:: u(x)\cdot w'(x)=g'(x)-u'(x)\cdot w(x)
 * Taking integration on both sides and ignoring the constant

  .. math:: \int u(x)\cdot w'(x)dx=\int g'(x)dx-\int u'(x)\cdot w(x)dx=u(x)\cdot w(x)-\int u'(x)\cdot w(x)dx
 * Replacing :math:`w(x)`
  .. math:: \int u(x)\cdot v(x)dx=u(x)\cdot \int v(x)dx-\int u'(x)\left(\int v(x)dx) \right)dx

Fubini's Theorem:
-------------------------------
For double integral of a function :math:`f(x,y)` in a rectangular region :math:`R=[a,b]\times [c,d]` and :math:`\iint\limits_{R} \left|f(x,y)\right|dx dy<\infty`, we can compute it using iterated integrals as follows:

 .. math:: \iint\limits_{R} f(x,y)dx dy=\int\limits_a^b \left(\int\limits_c^d f(x,y)dy\right)dx=\int\limits_c^d \left(\int\limits_a^b f(x,y)dx\right)dy

.. seealso::
 * Calculus cheatsheet: `Notes at tutorial.math.lamar.edu <https://tutorial.math.lamar.edu/pdf/calculus_cheat_sheet_all.pdf>`_.
 * Different ways for evaluating the Gaussian integral: `YouTube video playlist by Dr Peyam <https://www.youtube.com/watch?v=HcneBkidSDQ&list=PLJb1qAQIrmmCgLyHWMXGZnioRHLqOk2bW>`_.

  * Hints (one way): Let :math:`I=\int\limits_{-\infty}^\infty e^{x^2}dx`. Try to compute :math:`I^2`, convert this into a double integral using Fubini's theorem, and then use polar co-ordinate transform.

***********************************************
Geometry
***********************************************

.. seealso::
 * On the general equation of second degree: `Notes at IMSc <https://www.imsc.res.in/~svis/eme13/kesavan-new.pdf>`_.
